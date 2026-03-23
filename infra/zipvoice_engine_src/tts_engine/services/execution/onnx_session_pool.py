# -*- coding: utf-8 -*-
"""ONNX 세션 풀 - GPU 메모리 효율적인 ONNX 세션 관리.

N개의 ONNX 세션을 미리 로드하여 스레드 풀에서 재사용합니다.
GPU 메모리를 효율적으로 공유하며 동시 처리를 지원합니다.
"""

import io
import queue
import threading
import time
import wave
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from tts_engine.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OnnxSessionConfig:
    """ONNX 세션 설정."""
    model_dir: str
    vocoder_path: str
    device: str = "cuda:0"
    num_threads: int = 1
    # ONNX 모델 파일 경로 (기본값은 model_dir 기준)
    text_encoder_path: Optional[str] = None
    fm_decoder_path: Optional[str] = None
    # 추론 설정
    feat_scale: float = 0.1
    target_rms: float = 0.1


class OnnxSession:
    """단일 ONNX 세션 래퍼.

    text_encoder와 fm_decoder 세션을 포함합니다.
    각 세션은 별도의 CUDA 스트림을 사용하여 진정한 동시 처리를 지원합니다.
    """

    def __init__(
        self,
        session_id: int,
        config: OnnxSessionConfig,
    ):
        self.session_id = session_id
        self.config = config
        self.text_encoder = None
        self.fm_decoder = None
        self.feat_dim: int = 100
        self.vocoder = None
        self.preprocessor = None
        self._device = config.device
        self._is_loaded = False
        self._cuda_stream = None  # 세션별 CUDA 스트림

    def load(self) -> None:
        """ONNX 세션 로드."""
        import onnxruntime as ort
        from vocos import Vocos

        # ONNX Runtime 전역 로그 레벨 설정 (C++ 레벨)
        ort.set_default_logger_severity(3)  # 3 = ERROR

        model_dir = Path(self.config.model_dir)

        # ONNX 모델 경로
        text_encoder_path = self.config.text_encoder_path or str(model_dir / "text_encoder.onnx")
        fm_decoder_path = self.config.fm_decoder_path or str(model_dir / "fm_decoder.onnx")

        # 세션 옵션
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = self.config.num_threads
        session_opts.intra_op_num_threads = self.config.num_threads
        # Memcpy 경고 숨기기 (severity: 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL)
        session_opts.log_severity_level = 3  # ERROR 이상만 표시

        # GPU 프로바이더 설정
        device_id = 0
        if self._device.startswith("cuda:"):
            try:
                device_id = int(self._device.split(":")[1])
            except (IndexError, ValueError):
                device_id = 0

        available_providers = ort.get_available_providers()

        if "CUDAExecutionProvider" in available_providers:
            # 세션별 CUDA 스트림 생성 (동시 처리를 위해 필수)
            if torch.cuda.is_available():
                self._cuda_stream = torch.cuda.Stream(device=device_id)
                stream_ptr = self._cuda_stream.cuda_stream
                logger.debug(
                    f"Session {self.session_id}: Created CUDA stream",
                    stream_ptr=stream_ptr,
                    device_id=device_id,
                )
            else:
                stream_ptr = 0

            # CUDA 프로바이더 옵션 - 별도 스트림 사용
            cuda_options = {
                "device_id": device_id,
                "arena_extend_strategy": "kSameAsRequested",  # 메모리 할당 최적화
                "cudnn_conv_algo_search": "EXHAUSTIVE",  # 최적 알고리즘 탐색
                "do_copy_in_default_stream": False,  # 별도 스트림에서 복사
            }

            # CUDA 스트림 할당 (스트림이 생성된 경우)
            if stream_ptr:
                cuda_options["user_compute_stream"] = str(stream_ptr)

            providers = [
                ("CUDAExecutionProvider", cuda_options),
                "CPUExecutionProvider",
            ]
        else:
            logger.warning(f"Session {self.session_id}: CUDA not available, using CPU")
            providers = ["CPUExecutionProvider"]

        # Text Encoder 세션
        self.text_encoder = ort.InferenceSession(
            text_encoder_path,
            sess_options=session_opts,
            providers=providers,
        )

        # FM Decoder 세션
        self.fm_decoder = ort.InferenceSession(
            fm_decoder_path,
            sess_options=session_opts,
            providers=providers,
        )

        # 메타데이터에서 feat_dim 로드
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta.get("feat_dim", 100))

        # Vocoder 로드 (CPU에서 로드 후 GPU로 이동)
        vocoder_path = Path(self.config.vocoder_path)
        self.vocoder = Vocos.from_hparams(str(vocoder_path / "config.yaml"))
        state_dict = torch.load(
            str(vocoder_path / "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        self.vocoder.load_state_dict(state_dict)

        # Vocoder를 GPU로 이동
        if "cuda" in self._device:
            self.vocoder = self.vocoder.to(self._device)
        self.vocoder.eval()

        # Preprocessor 로드
        from tts_engine.models.zipvoice.preprocessor import ZipVoicePreprocessor
        from tts_engine.models.zipvoice.config import ZipVoiceConfig, InferenceConfig

        inference_config = InferenceConfig(
            feat_scale=self.config.feat_scale,
            target_rms=self.config.target_rms,
        )
        zipvoice_config = ZipVoiceConfig(
            model_dir=str(model_dir),
            vocoder_path=self.config.vocoder_path,
            device="cpu",  # 전처리는 CPU
            fp16=False,
            inference=inference_config,
        )
        self.preprocessor = ZipVoicePreprocessor(zipvoice_config)

        self._is_loaded = True
        logger.debug(f"ONNX session {self.session_id} loaded", device=self._device)

    def unload(self) -> None:
        """세션 리소스 해제."""
        if self.text_encoder:
            del self.text_encoder
            self.text_encoder = None
        if self.fm_decoder:
            del self.fm_decoder
            self.fm_decoder = None
        if self.vocoder:
            del self.vocoder
            self.vocoder = None
        if self.preprocessor:
            del self.preprocessor
            self.preprocessor = None

        # CUDA 스트림 정리
        if self._cuda_stream is not None:
            self._cuda_stream.synchronize()
            self._cuda_stream = None

        self._is_loaded = False

        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_text_encoder(
        self,
        tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_features_len: torch.Tensor,
        speed: torch.Tensor,
    ) -> torch.Tensor:
        """Text Encoder 실행."""
        out = self.text_encoder.run(
            [self.text_encoder.get_outputs()[0].name],
            {
                self.text_encoder.get_inputs()[0].name: tokens.numpy(),
                self.text_encoder.get_inputs()[1].name: prompt_tokens.numpy(),
                self.text_encoder.get_inputs()[2].name: prompt_features_len.numpy(),
                self.text_encoder.get_inputs()[3].name: speed.numpy(),
            },
        )
        return torch.from_numpy(out[0])

    def run_fm_decoder(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> torch.Tensor:
        """FM Decoder 실행."""
        out = self.fm_decoder.run(
            [self.fm_decoder.get_outputs()[0].name],
            {
                self.fm_decoder.get_inputs()[0].name: t.numpy(),
                self.fm_decoder.get_inputs()[1].name: x.numpy(),
                self.fm_decoder.get_inputs()[2].name: text_condition.numpy(),
                self.fm_decoder.get_inputs()[3].name: speech_condition.numpy(),
                self.fm_decoder.get_inputs()[4].name: guidance_scale.numpy(),
            },
        )
        return torch.from_numpy(out[0])

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded


class OnnxSessionPool:
    """ONNX 세션 풀.

    N개의 ONNX 세션을 관리하며, 스레드에서 세션을 획득/반환합니다.
    """

    def __init__(
        self,
        config: OnnxSessionConfig,
        pool_size: int = 4,
    ):
        self.config = config
        self.pool_size = pool_size
        self._sessions: List[OnnxSession] = []
        self._available: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._is_started = False

    def start(self) -> None:
        """세션 풀 시작 - 모든 세션 로드."""
        if self._is_started:
            return

        logger.info(
            "Starting ONNX session pool",
            pool_size=self.pool_size,
            device=self.config.device,
        )

        for i in range(self.pool_size):
            session = OnnxSession(session_id=i, config=self.config)
            session.load()
            self._sessions.append(session)
            self._available.put(session)

        self._is_started = True
        logger.info("ONNX session pool started", loaded_sessions=len(self._sessions))

    def stop(self) -> None:
        """세션 풀 중지 - 모든 세션 해제."""
        if not self._is_started:
            return

        logger.info("Stopping ONNX session pool")

        # 모든 세션 해제
        for session in self._sessions:
            session.unload()

        self._sessions.clear()

        # 큐 비우기
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except queue.Empty:
                break

        self._is_started = False
        logger.info("ONNX session pool stopped")

    @contextmanager
    def acquire(self, timeout: float = 30.0):
        """세션 획득 (컨텍스트 매니저).

        Args:
            timeout: 세션 대기 타임아웃 (초)

        Yields:
            OnnxSession 인스턴스

        Raises:
            TimeoutError: 타임아웃 내에 세션 획득 실패
        """
        session = None
        try:
            session = self._available.get(timeout=timeout)
            available = self._available.qsize()
            in_use = self.pool_size - available
            logger.info(
                "ONNX session acquired",
                session_id=session.session_id,
                in_use=in_use,
                available=available,
                pool_size=self.pool_size,
            )
            yield session
        except queue.Empty:
            raise TimeoutError(f"Failed to acquire ONNX session within {timeout}s")
        finally:
            if session is not None:
                self._available.put(session)
                available = self._available.qsize()
                in_use = self.pool_size - available
                logger.info(
                    "ONNX session released",
                    session_id=session.session_id,
                    in_use=in_use,
                    available=available,
                    pool_size=self.pool_size,
                )

    def get_stats(self) -> Dict[str, Any]:
        """세션 풀 통계."""
        return {
            "pool_size": self.pool_size,
            "available": self._available.qsize(),
            "in_use": self.pool_size - self._available.qsize(),
            "is_started": self._is_started,
        }


@dataclass
class SynthesisTask:
    """합성 태스크."""
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
class SynthesisResult:
    """합성 결과."""
    task_id: str
    success: bool
    audio_data: Optional[bytes] = None
    sample_rate: int = 24000
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None


def synthesize_with_session(
    session: OnnxSession,
    task: SynthesisTask,
) -> SynthesisResult:
    """ONNX 세션으로 합성 수행.

    각 세션은 별도의 CUDA 스트림에서 실행되어 진정한 동시 처리를 지원합니다.

    Args:
        session: ONNX 세션
        task: 합성 태스크

    Returns:
        합성 결과
    """
    from zipvoice.models.modules.solver import get_time_steps

    start_time = time.time()

    # 세션의 CUDA 스트림 컨텍스트에서 실행
    stream_context = (
        torch.cuda.stream(session._cuda_stream)
        if session._cuda_stream is not None
        else nullcontext()
    )

    try:
        with stream_context:
            # 전처리
            prompt_data = session.preprocessor.preprocess_prompt(
                prompt_wav_path=task.prompt_wav_path,
                prompt_text=task.prompt_text,
            )

            text_data = session.preprocessor.preprocess_text(
                text=task.text,
                prompt_duration=prompt_data["duration"],
                prompt_tokens_len=len(prompt_data["tokens"]),
                speed=task.speed,
            )

            feat_scale = session.config.feat_scale
            target_rms = session.config.target_rms

            # 텐서 준비 (CPU)
            # preprocessor에서 이미 feat_scale 적용됨
            prompt_features = prompt_data["features"].unsqueeze(0).float()
            prompt_features_len = torch.tensor(prompt_features.size(1), dtype=torch.int64)
            prompt_rms = prompt_data["original_rms"]

            # 청크 처리
            chunked_tokens = text_data["chunked_tokens"]
            if chunked_tokens:
                all_wavs = []
                for chunk_tokens in chunked_tokens:
                    wav = _run_onnx_inference(
                        session=session,
                        tokens=[chunk_tokens],
                        prompt_tokens=[prompt_data["tokens"]],
                        prompt_features=prompt_features,
                        prompt_features_len=prompt_features_len,
                        speed=task.speed,
                        t_shift=task.t_shift,
                        guidance_scale=task.guidance_scale,
                        num_steps=task.num_steps,
                        feat_scale=feat_scale,
                        target_rms=target_rms,
                        prompt_rms=prompt_rms,
                    )
                    all_wavs.append(wav)
                audio = torch.cat(all_wavs, dim=-1)
            else:
                tokens_str = text_data["tokens_str"]
                tokens = session.preprocessor.text_preprocessor.tokens_to_token_ids([tokens_str])[0]
                audio = _run_onnx_inference(
                    session=session,
                    tokens=[tokens],
                    prompt_tokens=[prompt_data["tokens"]],
                    prompt_features=prompt_features,
                    prompt_features_len=prompt_features_len,
                    speed=task.speed,
                    t_shift=task.t_shift,
                    guidance_scale=task.guidance_scale,
                    num_steps=task.num_steps,
                    feat_scale=feat_scale,
                    target_rms=target_rms,
                    prompt_rms=prompt_rms,
                )

            # WAV로 변환
            audio_np = audio.squeeze().cpu().float().numpy()

            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes((audio_np * 32767).astype(np.int16).tobytes())

            audio_bytes = buffer.getvalue()
            processing_time_ms = (time.time() - start_time) * 1000

        # 스트림 동기화 (결과 반환 전)
        if session._cuda_stream is not None:
            session._cuda_stream.synchronize()

        return SynthesisResult(
            task_id=task.task_id,
            success=True,
            audio_data=audio_bytes,
            sample_rate=24000,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"ONNX synthesis error: {e}")
        import traceback
        traceback.print_exc()

        return SynthesisResult(
            task_id=task.task_id,
            success=False,
            error_message=str(e),
            processing_time_ms=(time.time() - start_time) * 1000,
        )


def _run_onnx_inference(
    session: OnnxSession,
    tokens: List[List[int]],
    prompt_tokens: List[List[int]],
    prompt_features: torch.Tensor,
    prompt_features_len: torch.Tensor,
    speed: float,
    t_shift: float,
    guidance_scale: float,
    num_steps: int,
    feat_scale: float,
    target_rms: float,
    prompt_rms: float,
) -> torch.Tensor:
    """ONNX 추론 수행.

    세션의 CUDA 스트림에서 Vocoder도 실행하여 진정한 동시 처리를 지원합니다.
    """
    from zipvoice.models.modules.solver import get_time_steps

    # 텐서 준비
    tokens_tensor = torch.tensor(tokens, dtype=torch.int64)
    prompt_tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.int64)
    speed_tensor = torch.tensor(speed, dtype=torch.float32)

    # Text Encoder 실행
    text_condition = session.run_text_encoder(
        tokens_tensor, prompt_tokens_tensor, prompt_features_len, speed_tensor
    )

    batch_size, num_frames, _ = text_condition.shape
    feat_dim = session.feat_dim

    # Flow Matching
    timesteps = get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=num_steps,
        t_shift=t_shift,
    )
    x = torch.randn(batch_size, num_frames, feat_dim)
    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, num_frames - prompt_features.shape[1])
    )
    guidance_scale_tensor = torch.tensor(guidance_scale, dtype=torch.float32)

    for step in range(num_steps):
        v = session.run_fm_decoder(
            t=timesteps[step],
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale_tensor,
        )
        x = x + v * (timesteps[step + 1] - timesteps[step])

    # 프롬프트 부분 제거
    pred_features = x[:, prompt_features_len.item():, :]

    # 특징 후처리
    pred_features = pred_features.permute(0, 2, 1) / feat_scale

    # GPU로 이동 및 Vocoder 실행
    # 세션의 CUDA 스트림에서 실행하여 동시 처리 지원
    if session.vocoder is not None:
        device = next(session.vocoder.parameters()).device
        pred_features = pred_features.to(device)

        # 세션의 CUDA 스트림에서 Vocoder 실행 (핵심 병목 해결)
        stream_context = (
            torch.cuda.stream(session._cuda_stream)
            if session._cuda_stream is not None
            else nullcontext()
        )
        with stream_context:
            with torch.no_grad():
                wav = session.vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)
    else:
        with torch.no_grad():
            wav = pred_features.squeeze(1).clamp(-1, 1)  # Fallback

    # 볼륨 조정
    if prompt_rms < target_rms:
        wav = wav * prompt_rms / target_rms

    return wav
