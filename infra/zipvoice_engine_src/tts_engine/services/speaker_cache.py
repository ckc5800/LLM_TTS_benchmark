# -*- coding: utf-8 -*-
"""Speaker Cache - 화자 프롬프트 캐싱.

프롬프트 WAV의 특징 추출을 사전 처리하여 캐싱합니다.
ZipVoice PyTriton의 speaker_info_dict 패턴을 참조.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from tts_engine.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CachedSpeakerInfo:
    """캐시된 화자 정보."""

    voice_id: str
    prompt_tokens: List[int]
    prompt_features: torch.Tensor  # (T, F) CPU에 저장
    prompt_features_len: int
    original_rms: float
    duration: float  # 프롬프트 오디오 길이 (초)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def touch(self) -> None:
        """접근 시간 업데이트."""
        self.last_accessed = time.time()
        self.access_count += 1


class SpeakerCache:
    """화자 프롬프트 캐시.

    프롬프트 WAV의 특징 추출 결과를 캐싱하여
    동일 음성에 대한 전처리 시간을 제거합니다.

    PyTriton의 speaker_info_dict 패턴:
    ```python
    self.speaker_info_dict["default"] = (prompt_tokens, prompt_features, prompt_rms)
    ```
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 3600.0,
    ):
        """화자 캐시 초기화.

        Args:
            max_size: 최대 캐시 크기
            ttl_seconds: 캐시 유효 시간 (초)
        """
        self._cache: Dict[str, CachedSpeakerInfo] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._preprocessor = None

        # 통계
        self._hits = 0
        self._misses = 0

        logger.info(
            "SpeakerCache initialized",
            max_size=max_size,
            ttl_seconds=ttl_seconds,
        )

    def set_preprocessor(self, preprocessor: Any) -> None:
        """전처리기 설정.

        Args:
            preprocessor: ZipVoicePreprocessor 인스턴스
        """
        self._preprocessor = preprocessor

    def get(self, voice_id: str) -> Optional[CachedSpeakerInfo]:
        """캐시된 화자 정보 조회.

        Args:
            voice_id: 음성 ID

        Returns:
            CachedSpeakerInfo 또는 None
        """
        info = self._cache.get(voice_id)

        if info is None:
            self._misses += 1
            return None

        # TTL 체크
        if time.time() - info.created_at > self._ttl_seconds:
            del self._cache[voice_id]
            self._misses += 1
            return None

        info.touch()
        self._hits += 1
        return info

    def put(
        self,
        voice_id: str,
        prompt_tokens: List[int],
        prompt_features: torch.Tensor,
        original_rms: float,
        duration: float,
    ) -> CachedSpeakerInfo:
        """화자 정보 캐싱.

        Args:
            voice_id: 음성 ID
            prompt_tokens: 프롬프트 토큰
            prompt_features: 프롬프트 특징 (T, F)
            original_rms: 원본 RMS
            duration: 프롬프트 오디오 길이 (초)

        Returns:
            CachedSpeakerInfo
        """
        # LRU 정책: 캐시 가득 차면 가장 오래된 항목 제거
        if len(self._cache) >= self._max_size:
            oldest = min(self._cache.items(), key=lambda x: x[1].last_accessed)
            del self._cache[oldest[0]]
            logger.debug("Evicted oldest cache entry", voice_id=oldest[0])

        # CPU에 저장 (GPU 메모리 절약)
        if prompt_features.is_cuda:
            prompt_features = prompt_features.cpu()

        info = CachedSpeakerInfo(
            voice_id=voice_id,
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            prompt_features_len=prompt_features.size(0),
            original_rms=original_rms,
            duration=duration,
        )

        self._cache[voice_id] = info

        logger.debug(
            "Speaker info cached",
            voice_id=voice_id,
            features_shape=list(prompt_features.shape),
        )

        return info

    def preload(
        self,
        voice_id: str,
        prompt_wav_path: str,
        prompt_text: str,
    ) -> Optional[CachedSpeakerInfo]:
        """화자 정보 사전 로드.

        Args:
            voice_id: 음성 ID
            prompt_wav_path: 프롬프트 WAV 경로
            prompt_text: 프롬프트 텍스트

        Returns:
            CachedSpeakerInfo 또는 None
        """
        if self._preprocessor is None:
            logger.error("Preprocessor not set")
            return None

        # 이미 캐시된 경우 스킵
        if voice_id in self._cache:
            logger.debug("Voice already cached", voice_id=voice_id)
            return self._cache[voice_id]

        try:
            start_time = time.time()

            # 전처리 실행
            prompt_data = self._preprocessor.preprocess_prompt(
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text,
            )

            # 캐시에 저장
            info = self.put(
                voice_id=voice_id,
                prompt_tokens=prompt_data["tokens"],
                prompt_features=prompt_data["features"],
                original_rms=prompt_data["original_rms"],
                duration=prompt_data["duration"],
            )

            elapsed = (time.time() - start_time) * 1000
            logger.info(
                "Speaker preloaded",
                voice_id=voice_id,
                duration_ms=f"{elapsed:.1f}",
            )

            return info

        except Exception as e:
            logger.error(
                "Failed to preload speaker",
                voice_id=voice_id,
                error=str(e),
            )
            return None

    def get_or_load(
        self,
        voice_id: str,
        prompt_wav_path: str,
        prompt_text: str,
    ) -> Optional[CachedSpeakerInfo]:
        """캐시에서 조회하거나 로드.

        Args:
            voice_id: 음성 ID
            prompt_wav_path: 프롬프트 WAV 경로
            prompt_text: 프롬프트 텍스트

        Returns:
            CachedSpeakerInfo 또는 None
        """
        info = self.get(voice_id)
        if info is not None:
            return info

        return self.preload(voice_id, prompt_wav_path, prompt_text)

    def clear(self, voice_id: Optional[str] = None) -> int:
        """캐시 클리어.

        Args:
            voice_id: 특정 음성만 삭제 (None이면 전체)

        Returns:
            삭제된 항목 수
        """
        if voice_id:
            if voice_id in self._cache:
                del self._cache[voice_id]
                return 1
            return 0

        count = len(self._cache)
        self._cache.clear()
        logger.info("Speaker cache cleared", count=count)
        return count

    def get_stats(self) -> dict:
        """캐시 통계 반환.

        Returns:
            통계 딕셔너리
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
            "ttl_seconds": self._ttl_seconds,
            "cached_voices": list(self._cache.keys()),
        }

    def get_for_batch(
        self,
        voice_ids: List[str],
        device: torch.device,
    ) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor, List[float]]:
        """배치용 화자 정보 조회.

        동일 음성에 대해 캐시된 정보를 배치로 반환합니다.

        Args:
            voice_ids: 음성 ID 목록
            device: 대상 디바이스

        Returns:
            (prompt_tokens_list, prompt_features_batch, prompt_features_lens, original_rms_list)
            prompt_features_batch: (B, T, F) 형태로 패딩됨
        """
        prompt_tokens_list = []
        prompt_features_list = []
        original_rms_list = []

        max_len = 0

        for voice_id in voice_ids:
            info = self.get(voice_id)
            if info is None:
                raise ValueError(f"Voice not cached: {voice_id}")

            prompt_tokens_list.append(info.prompt_tokens)
            prompt_features_list.append(info.prompt_features)
            original_rms_list.append(info.original_rms)
            max_len = max(max_len, info.prompt_features_len)

        # 패딩하여 배치 생성 (T, F) -> (B, T, F)
        batch_features = []
        lens = []
        for feat in prompt_features_list:
            lens.append(feat.size(0))
            if feat.size(0) < max_len:
                pad_size = max_len - feat.size(0)
                feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_size))
            batch_features.append(feat.unsqueeze(0))

        prompt_features_batch = torch.cat(batch_features, dim=0).to(device)
        prompt_features_lens = torch.tensor(lens, device=device)

        return prompt_tokens_list, prompt_features_batch, prompt_features_lens, original_rms_list

    def contains(self, voice_id: str) -> bool:
        """캐시에 음성이 있는지 확인."""
        return voice_id in self._cache

    def __len__(self) -> int:
        """캐시된 항목 수."""
        return len(self._cache)

    def __contains__(self, voice_id: str) -> bool:
        """in 연산자 지원."""
        return voice_id in self._cache
