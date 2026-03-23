"""
TTS 모델 어댑터 베이스 클래스
각 모델은 이 클래스를 상속하여 구현
"""
from __future__ import annotations
import time
import os
from abc import ABC, abstractmethod
from typing import Optional
from .core import BenchmarkResult, BenchmarkLogger, get_vram_mb, clear_vram, DEFAULT_TEST_TEXT


class BaseTTSAdapter(ABC):
    """모든 TTS 모델 어댑터의 베이스"""

    model_name: str = "Unknown"
    model_version: str = ""
    model_size_params: str = ""
    korean_support: bool = True
    model_hf_id: str = ""

    def __init__(self, model_dir: str, output_dir: str):
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._model = None

    # ─── 구현 필수 메서드 ─────────────────────────────────────────────────────

    @abstractmethod
    def _download(self):
        """HuggingFace에서 모델 다운로드"""
        pass

    @abstractmethod
    def _load_model(self):
        """모델 로드 (self._model 설정)"""
        pass

    @abstractmethod
    def _synthesize(self, text: str) -> tuple[any, int, float]:
        """
        텍스트 → 오디오 생성
        Returns:
            (audio_tensor_or_array, sample_rate, ttfa_ms)
        """
        pass

    @abstractmethod
    def _unload_model(self):
        """모델 언로드 및 VRAM 해제"""
        pass

    # ─── 공통 실행 로직 ──────────────────────────────────────────────────────

    def ensure_downloaded(self):
        """모델 없으면 다운로드"""
        if not self._is_downloaded():
            print(f"[DOWNLOAD] {self.model_name} → {self.model_dir}")
            self._download()
            print(f"[DOWNLOAD] 완료")

    def _is_downloaded(self) -> bool:
        return os.path.exists(self.model_dir) and len(os.listdir(self.model_dir)) > 2

    def run_benchmark(
        self,
        logger: BenchmarkLogger,
        text: str = DEFAULT_TEST_TEXT,
        n_runs: int = 1,
        load_fresh: bool = True,
    ) -> list[BenchmarkResult]:
        """모델 로드 → 추론 → 결과 기록 → 언로드"""
        results = []

        # 다운로드
        try:
            self.ensure_downloaded()
        except Exception as e:
            r = BenchmarkResult(
                model_name=self.model_name,
                model_version=self.model_version,
                model_size_params=self.model_size_params,
                korean_support=self.korean_support,
                test_text=text,
                success=False,
                error=f"Download failed: {e}",
            )
            logger.add(r)
            return [r]

        # 모델 로드
        clear_vram()
        vram_before = get_vram_mb()
        t_load_start = time.perf_counter()
        try:
            self._load_model()
            load_time = time.perf_counter() - t_load_start
            vram_after_load = get_vram_mb()
            print(f"[LOAD] {self.model_name} ({load_time:.1f}s, +{vram_after_load - vram_before:.0f}MB VRAM)")
        except Exception as e:
            r = BenchmarkResult(
                model_name=self.model_name,
                model_version=self.model_version,
                model_size_params=self.model_size_params,
                korean_support=self.korean_support,
                test_text=text,
                load_time_s=time.perf_counter() - t_load_start,
                vram_before_mb=vram_before,
                success=False,
                error=f"Load failed: {e}",
            )
            logger.add(r)
            return [r]

        # 추론 (워밍업 1회 후 n_runs 측정)
        for run_idx in range(-1, n_runs):  # -1 = warmup
            is_warmup = run_idx == -1
            label = "warmup" if is_warmup else f"run {run_idx}"
            print(f"  [{self.model_name}] {label}...")

            vram_inf_before = get_vram_mb()
            wav_path = os.path.join(
                self.output_dir,
                f"{self.model_name.replace(' ', '_').replace('/', '_')}_{max(run_idx, 0)}.wav"
            )

            try:
                t_inf_start = time.perf_counter()
                audio, sr, ttfa_ms = self._synthesize(text)
                inference_time = time.perf_counter() - t_inf_start
                vram_inf_after = get_vram_mb()

                # WAV 저장
                audio_duration = self._save_wav(audio, sr, wav_path)

                if is_warmup:
                    continue  # warmup 결과는 기록하지 않음

                r = BenchmarkResult(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    model_size_params=self.model_size_params,
                    korean_support=self.korean_support,
                    test_text=text,
                    run_index=run_idx,
                    load_time_s=load_time,
                    ttfa_ms=ttfa_ms,
                    inference_time_s=inference_time,
                    audio_duration_s=audio_duration,
                    vram_before_mb=vram_before,
                    vram_after_mb=vram_after_load,
                    sample_rate=sr,
                    output_wav=wav_path,
                    success=True,
                )
                logger.add(r)
                results.append(r)

            except Exception as e:
                import traceback
                if not is_warmup:
                    r = BenchmarkResult(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        model_size_params=self.model_size_params,
                        korean_support=self.korean_support,
                        test_text=text,
                        run_index=run_idx,
                        load_time_s=load_time,
                        vram_before_mb=vram_before,
                        vram_after_mb=vram_after_load,
                        success=False,
                        error=str(e)[:200],
                        notes=traceback.format_exc()[-500:],
                    )
                    logger.add(r)
                    results.append(r)

        # 언로드
        try:
            self._unload_model()
            clear_vram()
        except Exception as e:
            print(f"  [{self.model_name}] unload error: {e}")

        return results

    def _save_wav(self, audio, sample_rate: int, path: str) -> float:
        """WAV 저장 및 오디오 길이 반환 (soundfile 직접 사용 - torchaudio 2.10 호환)"""
        import torch
        import numpy as np
        import soundfile as sf

        if isinstance(audio, np.ndarray):
            data = audio.astype(np.float32)
        else:
            if audio.dtype != torch.float32:
                audio = audio.float()
            data = audio.cpu().numpy()

        # (channels, samples) → (samples,) or (samples, channels)
        if data.ndim == 2:
            if data.shape[0] == 1:
                data = data[0]  # mono: (1, N) → (N,)
            else:
                data = data.T   # stereo: (C, N) → (N, C)

        # 클리핑 방지
        max_val = np.abs(data).max()
        if max_val > 1.0:
            data = data / max_val

        sf.write(path, data, sample_rate, subtype='PCM_16')
        duration = data.shape[0] / sample_rate
        return duration
