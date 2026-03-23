# -*- coding: utf-8 -*-
"""PyTorch 멀티프로세스 워커.

각 워커 프로세스에서 독립적으로 모델을 로드하고 합성을 수행합니다.
GPU 메모리가 프로세스별로 분리되어 진정한 병렬 처리가 가능합니다.
"""

import json
import queue
import time
from dataclasses import dataclass, asdict
from multiprocessing import Queue, Event
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class WorkerTask:
    """워커에게 전달할 태스크."""
    task_id: str
    text: str
    voice_id: str
    prompt_wav_path: str
    prompt_text: str
    speed: float = 1.0
    num_steps: int = 16
    t_shift: float = 0.5
    guidance_scale: float = 1.0


@dataclass
class WorkerResult:
    """워커로부터 받는 결과."""
    task_id: str
    success: bool
    audio_data: Optional[bytes] = None
    sample_rate: int = 24000
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None


def pytorch_worker_main(
    worker_id: int,
    task_queue: Queue,
    result_queue: Queue,
    model_config: dict,
    shutdown_event: Event,
):
    """PyTorch 워커 프로세스 메인 함수.

    Args:
        worker_id: 워커 ID
        task_queue: 태스크 수신 큐
        result_queue: 결과 전송 큐
        model_config: 모델 설정
        shutdown_event: 종료 이벤트
    """
    # 워커 프로세스에서 SIGINT 무시 (메인 프로세스에서 shutdown_event로 제어)
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # 경고 메시지 억제
    import warnings
    import logging
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("root").setLevel(logging.ERROR)

    import torch
    import torchaudio

    worker_name = f"pytorch_worker_{worker_id}"
    device = model_config.get("device", "cuda:0")

    print(f"[{worker_name}] Starting on device {device}")

    # GPU 설정
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 모델 및 보코더
    model = None
    vocoder = None
    preprocessor = None

    try:
        # 모델 로드
        model, vocoder, preprocessor = _load_model(model_config, device)

        # GPU 메모리 사용량 출력
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated(device) / 1024**3
            print(f"[{worker_name}] Model loaded, GPU memory: {mem_gb:.2f} GB")

        # 초기화 성공 알림
        result_queue.put(WorkerResult(
            task_id="__init__",
            success=True,
        ))

    except Exception as e:
        print(f"[{worker_name}] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put(WorkerResult(
            task_id="__init__",
            success=False,
            error_message=str(e),
        ))
        return

    # 메인 루프
    try:
        while not shutdown_event.is_set():
            try:
                # 태스크 대기
                try:
                    task_data = task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if task_data is None:  # Poison pill
                    break

                # WorkerTask로 변환
                if isinstance(task_data, dict):
                    task = WorkerTask(**task_data)
                else:
                    task = task_data

                # 합성 수행
                result = _synthesize(
                    worker_name=worker_name,
                    task=task,
                    model=model,
                    vocoder=vocoder,
                    preprocessor=preprocessor,
                    device=device,
                    model_config=model_config,
                )

                result_queue.put(result)

            except Exception as e:
                if shutdown_event.is_set():
                    break
                print(f"[{worker_name}] Error in main loop: {e}")
                import traceback
                traceback.print_exc()

    except (KeyboardInterrupt, SystemExit):
        pass  # 조용히 종료

    # 정리
    print(f"[{worker_name}] Shutting down")
    _cleanup(model, vocoder, device)


def _load_model(model_config: dict, device: str):
    """모델, 보코더, 전처리기를 로드합니다."""
    import torch
    from zipvoice.models.zipvoice import ZipVoice
    from vocos import Vocos
    from zipvoice.utils.checkpoint import load_checkpoint

    model_dir = Path(model_config["model_path"])
    model_config_path = model_dir / "model.json"
    checkpoint_path = model_dir / "model.pt"

    # 모델 설정 로드
    with open(model_config_path, "r") as f:
        model_json = json.load(f)

    # 토크나이저 설정
    tokenizer_config = {
        "vocab_size": 159,
        "pad_id": 158,
    }

    # 모델 생성
    model = ZipVoice(
        **model_json["model"],
        **tokenizer_config,
    )

    # 체크포인트 로드
    load_checkpoint(filename=str(checkpoint_path), model=model, strict=True)

    # GPU로 이동
    model = model.to(device)
    model.eval()

    # FP16은 autocast로 처리 (모델 자체는 FP32 유지)
    # model.half()를 호출하면 내부 임베딩에서 dtype 불일치 발생

    # 보코더 로드 (설정 경로 우선)
    vocoder_path_str = model_config.get("vocoder_path", "./models/vocos/mel-24khz")
    vocoder_path = Path(vocoder_path_str)
    if vocoder_path.exists():
        vocoder = Vocos.from_hparams(str(vocoder_path / "config.yaml"))
        state_dict = torch.load(
            str(vocoder_path / "pytorch_model.bin"),
            weights_only=True,
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.to(device).eval()
        print(f"[Worker] Vocoder loaded from {vocoder_path}")
    else:
        # 로컬 없으면 HuggingFace에서 다운로드
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device).eval()
        print(f"[Worker] Vocoder downloaded from HuggingFace")

    # 보코더도 FP32 유지 (autocast에서 자동 처리)

    # 전처리기 (ZipVoicePreprocessor)
    from tts_engine.models.zipvoice.preprocessor import ZipVoicePreprocessor
    from tts_engine.models.zipvoice.config import ZipVoiceConfig, InferenceConfig

    # 설정 로드
    inference_config = InferenceConfig(
        num_steps=model_config.get("num_steps", 16),
        t_shift=model_config.get("t_shift", 0.5),
        guidance_scale=model_config.get("guidance_scale", 1.0),
    )
    zipvoice_config = ZipVoiceConfig(
        model_dir=str(model_dir),
        vocoder_path=vocoder_path_str,
        device=device,
        fp16=model_config.get("fp16", True),
        inference=inference_config,
    )
    preprocessor = ZipVoicePreprocessor(zipvoice_config)
    print(f"[Worker] ZipVoicePreprocessor initialized")

    return model, vocoder, preprocessor


def _synthesize(
    worker_name: str,
    task: WorkerTask,
    model,
    vocoder,
    preprocessor,
    device: str,
    model_config: dict,
) -> WorkerResult:
    """단일 합성을 수행합니다."""
    import torch

    start_time = time.time()

    try:
        # 프롬프트 전처리 (ZipVoicePreprocessor 사용)
        prompt_data = preprocessor.preprocess_prompt(
            prompt_wav_path=task.prompt_wav_path,
            prompt_text=task.prompt_text,
        )

        # 텍스트 전처리
        text_data = preprocessor.preprocess_text(
            text=task.text,
            prompt_duration=prompt_data["duration"],
            prompt_tokens_len=len(prompt_data["tokens"]),
            speed=task.speed,
        )

        # feat_scale (org 소스 기본값: 0.1)
        # 주의: preprocessor.preprocess_prompt()에서 이미 feat_scale 적용됨
        feat_scale = model_config.get("feat_scale", 0.1)
        target_rms = model_config.get("target_rms", 0.1)

        # 텐서 변환 및 GPU 이동 (FP32 유지)
        # preprocessor에서 이미 feat_scale 적용되었으므로 여기서는 적용하지 않음
        prompt_features = prompt_data["features"].unsqueeze(0).to(device).float()
        prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)

        # 프롬프트 RMS
        prompt_rms = prompt_data["original_rms"]

        # 배치 처리 (chunked_tokens 사용)
        chunked_tokens = text_data["chunked_tokens"]
        batch_tokens = chunked_tokens if chunked_tokens else [text_data["tokens_str"]]
        batch_prompt_tokens = [prompt_data["tokens"]] * len(batch_tokens)
        batch_prompt_features = prompt_features.repeat(len(batch_tokens), 1, 1)
        batch_prompt_features_lens = prompt_features_lens.repeat(len(batch_tokens))

        # 합성 (FP32 모드)
        with torch.no_grad():
            (
                pred_features,
                pred_features_lens,
                _,
                _,
            ) = model.sample(
                tokens=batch_tokens,
                prompt_tokens=batch_prompt_tokens,
                prompt_features=batch_prompt_features,
                prompt_features_lens=batch_prompt_features_lens,
                speed=task.speed,
                t_shift=task.t_shift,
                duration="predict",
                num_step=task.num_steps,
                guidance_scale=task.guidance_scale,
            )

            # 특징 후처리 (org 소스 방식)
            pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)

            # 보코더로 오디오 생성 (배치별 처리)
            batch_wavs = []
            for i in range(pred_features.size(0)):
                wav = (
                    vocoder.decode(pred_features[i][None, :, :pred_features_lens[i]])
                    .squeeze(1)
                    .clamp(-1, 1)
                )
                # 볼륨 조정 (org 소스 방식)
                if prompt_rms < target_rms:
                    wav = wav * prompt_rms / target_rms
                batch_wavs.append(wav)

            # 배치 결과 연결
            audio = torch.cat(batch_wavs, dim=-1)

        # numpy로 변환
        audio_np = audio.squeeze().cpu().float().numpy()

        # bytes로 변환 (WAV 포맷)
        import io
        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(24000)
            wav_file.writeframes((audio_np * 32767).astype(np.int16).tobytes())

        audio_bytes = buffer.getvalue()
        processing_time_ms = (time.time() - start_time) * 1000

        # CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return WorkerResult(
            task_id=task.task_id,
            success=True,
            audio_data=audio_bytes,
            sample_rate=24000,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        print(f"[{worker_name}] Synthesis error: {e}")
        import traceback
        traceback.print_exc()

        return WorkerResult(
            task_id=task.task_id,
            success=False,
            error_message=str(e),
            processing_time_ms=(time.time() - start_time) * 1000,
        )


def _cleanup(model, vocoder, device: str):
    """모델 및 GPU 메모리 정리."""
    import torch

    if model is not None:
        del model
    if vocoder is not None:
        del vocoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
