# -*- coding: utf-8 -*-
"""음성 매니저 - 음성 설정 및 매핑 관리.

음성 목록 관리, 음성-모델 매핑, 음성 검증을 담당합니다.
"""

from pathlib import Path
from typing import Dict, List, Optional

from tts_engine.core.config import VoiceConfig, VoicesConfig
from tts_engine.core.exceptions import (
    VoiceError,
    VoiceLoadError,
    VoiceNotFoundError,
    VoiceNotReadyError,
)
from tts_engine.core.logging import get_logger
from tts_engine.domain.voice import VoiceData, VoiceListResponse

logger = get_logger(__name__)


class VoiceManager:
    """음성 매니저 - 음성 설정 및 상태 관리.

    음성 설정을 로드하고, 활성화된 음성 목록을 관리하며,
    음성별 프롬프트 데이터를 제공합니다.

    Example:
        >>> manager = VoiceManager(voices_config)
        >>> voice = manager.get_voice("kor_female_01")
        >>> voices = manager.list_voices(language="ko-KR")
    """

    def __init__(self, voices_config: VoicesConfig):
        """매니저 초기화.

        Args:
            voices_config: 음성 설정
        """
        self._config = voices_config
        self._voices: Dict[str, VoiceData] = {}
        self._ready_voices: set[str] = set()
        self._default_voice_id = voices_config.default_voice_id
        self._fallback_voice_id = voices_config.fallback_voice_id

        # 설정에서 음성 데이터 로드
        self._load_voices_from_config()

        logger.info(
            "VoiceManager initialized",
            total_voices=len(self._voices),
            enabled_voices=self._count_enabled_voices(),
            default_voice=self._default_voice_id,
            fallback_voice=self._fallback_voice_id,
        )

    def _load_voices_from_config(self) -> None:
        """설정에서 음성 데이터를 로드합니다."""
        for voice_id, voice_config in self._config.voices.items():
            try:
                voice_data = self._config_to_voice_data(voice_id, voice_config)
                self._voices[voice_id] = voice_data

                logger.debug(
                    "Voice loaded from config",
                    voice_id=voice_id,
                    name=voice_config.name,
                    enabled=voice_config.enabled,
                )

            except Exception as e:
                logger.error(
                    "Failed to load voice from config",
                    voice_id=voice_id,
                    error=str(e),
                )

    def _config_to_voice_data(
        self,
        voice_id: str,
        config: VoiceConfig,
    ) -> VoiceData:
        """VoiceConfig를 VoiceData로 변환합니다.

        Args:
            voice_id: 음성 ID
            config: 음성 설정

        Returns:
            VoiceData 객체
        """
        return VoiceData(
            voice_id=voice_id,
            name=config.name,
            language=config.language,
            gender=config.gender,
            model_instance=config.model_instance,
            description=config.description,
            sample_rate=config.sample_rate,
            enabled=config.enabled,
            prompt_wav=config.prompt_wav,
            prompt_text=config.prompt_text,
            options=config.options,
        )

    def _count_enabled_voices(self) -> int:
        """활성화된 음성 수를 반환합니다."""
        return sum(1 for v in self._voices.values() if v.enabled)

    def get_voice(self, voice_id: str) -> VoiceData:
        """음성 데이터를 반환합니다.

        Args:
            voice_id: 음성 ID

        Returns:
            VoiceData

        Raises:
            VoiceNotFoundError: 음성이 없는 경우
        """
        if voice_id not in self._voices:
            raise VoiceNotFoundError(
                f"Voice '{voice_id}' not found. "
                f"Available voices: {list(self._voices.keys())}"
            )

        return self._voices[voice_id]

    def get_voice_or_fallback(self, voice_id: str) -> VoiceData:
        """음성을 조회하거나 폴백 음성을 반환합니다.

        Args:
            voice_id: 음성 ID

        Returns:
            VoiceData (요청 음성 또는 폴백)

        Raises:
            VoiceNotFoundError: 음성과 폴백 모두 없는 경우
        """
        # 요청한 음성 시도
        if voice_id in self._voices and self._voices[voice_id].enabled:
            return self._voices[voice_id]

        # 폴백 음성 시도
        if self._fallback_voice_id and self._fallback_voice_id in self._voices:
            logger.warning(
                "Using fallback voice",
                requested=voice_id,
                fallback=self._fallback_voice_id,
            )
            return self._voices[self._fallback_voice_id]

        # 기본 음성 시도
        if self._default_voice_id and self._default_voice_id in self._voices:
            logger.warning(
                "Using default voice",
                requested=voice_id,
                default=self._default_voice_id,
            )
            return self._voices[self._default_voice_id]

        raise VoiceNotFoundError(
            f"Voice '{voice_id}' not found and no fallback available"
        )

    def get_default_voice(self) -> VoiceData:
        """기본 음성을 반환합니다.

        Returns:
            기본 VoiceData

        Raises:
            VoiceNotFoundError: 기본 음성이 설정되지 않은 경우
        """
        if not self._default_voice_id:
            raise VoiceNotFoundError("No default voice configured")

        return self.get_voice(self._default_voice_id)

    def list_voices(
        self,
        language: Optional[str] = None,
        gender: Optional[str] = None,
        enabled_only: bool = True,
    ) -> List[VoiceData]:
        """음성 목록을 반환합니다.

        Args:
            language: 언어 필터 (예: "ko-KR")
            gender: 성별 필터 (예: "female", "male")
            enabled_only: 활성화된 음성만 포함

        Returns:
            VoiceData 목록
        """
        voices = list(self._voices.values())

        # 필터 적용
        if enabled_only:
            voices = [v for v in voices if v.enabled]

        if language:
            voices = [v for v in voices if v.language == language]

        if gender:
            voices = [v for v in voices if v.gender == gender]

        return voices

    def list_voice_ids(
        self,
        enabled_only: bool = True,
    ) -> List[str]:
        """음성 ID 목록을 반환합니다.

        Args:
            enabled_only: 활성화된 음성만 포함

        Returns:
            음성 ID 목록
        """
        if enabled_only:
            return [
                voice_id
                for voice_id, voice in self._voices.items()
                if voice.enabled
            ]
        return list(self._voices.keys())

    def get_voices_response(
        self,
        language: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> VoiceListResponse:
        """API 응답용 음성 목록을 반환합니다.

        Args:
            language: 언어 필터
            gender: 성별 필터

        Returns:
            VoiceListResponse
        """
        voices = self.list_voices(
            language=language,
            gender=gender,
            enabled_only=True,
        )

        return VoiceListResponse(
            voices=voices,
            total_count=len(voices),
            default_voice_id=self._default_voice_id,
        )

    def is_voice_enabled(self, voice_id: str) -> bool:
        """음성이 활성화됐는지 확인합니다.

        Args:
            voice_id: 음성 ID

        Returns:
            활성화 여부
        """
        if voice_id not in self._voices:
            return False
        return self._voices[voice_id].enabled

    def mark_voice_ready(self, voice_id: str) -> None:
        """음성을 사용 가능 상태로 표시합니다.

        모델에 음성 프롬프트 로드 완료 후 호출합니다.

        Args:
            voice_id: 음성 ID

        Raises:
            VoiceNotFoundError: 음성이 없는 경우
        """
        if voice_id not in self._voices:
            raise VoiceNotFoundError(f"Voice '{voice_id}' not found")

        self._ready_voices.add(voice_id)
        logger.debug("Voice marked as ready", voice_id=voice_id)

    def mark_voice_not_ready(self, voice_id: str) -> None:
        """음성을 사용 불가능 상태로 표시합니다.

        Args:
            voice_id: 음성 ID
        """
        self._ready_voices.discard(voice_id)
        logger.debug("Voice marked as not ready", voice_id=voice_id)

    def is_voice_ready(self, voice_id: str) -> bool:
        """음성이 사용 가능한지 확인합니다.

        Args:
            voice_id: 음성 ID

        Returns:
            사용 가능 여부
        """
        return voice_id in self._ready_voices

    def ensure_voice_ready(self, voice_id: str) -> VoiceData:
        """음성이 사용 가능한지 확인하고 반환합니다.

        Args:
            voice_id: 음성 ID

        Returns:
            VoiceData

        Raises:
            VoiceNotFoundError: 음성이 없는 경우
            VoiceNotReadyError: 음성이 사용 불가한 경우
        """
        voice = self.get_voice(voice_id)

        if not voice.enabled:
            raise VoiceNotReadyError(f"Voice '{voice_id}' is disabled")

        if not self.is_voice_ready(voice_id):
            raise VoiceNotReadyError(
                f"Voice '{voice_id}' is not ready. "
                f"Prompt may not be loaded."
            )

        return voice

    def get_model_instance(self, voice_id: str) -> str:
        """음성에 매핑된 모델 인스턴스를 반환합니다.

        Args:
            voice_id: 음성 ID

        Returns:
            모델 인스턴스 이름

        Raises:
            VoiceNotFoundError: 음성이 없는 경우
        """
        voice = self.get_voice(voice_id)
        return voice.model_instance

    def get_voices_by_instance(self, instance_name: str) -> List[VoiceData]:
        """특정 모델 인스턴스에 매핑된 음성들을 반환합니다.

        Args:
            instance_name: 모델 인스턴스 이름

        Returns:
            VoiceData 목록
        """
        return [
            voice
            for voice in self._voices.values()
            if voice.model_instance == instance_name and voice.enabled
        ]

    def get_languages(self) -> List[str]:
        """현재 사용 가능한 언어 목록을 반환합니다.

        Returns:
            언어 코드 목록
        """
        languages = set(
            voice.language
            for voice in self._voices.values()
            if voice.enabled
        )
        return sorted(languages)

    def validate_voice_config(self, voice_id: str) -> List[str]:
        """음성 설정의 유효성을 검증합니다.

        Args:
            voice_id: 음성 ID

        Returns:
            에러 메시지 목록 (빈 리스트면 유효)
        """
        errors = []

        if voice_id not in self._voices:
            errors.append(f"Voice '{voice_id}' not found")
            return errors

        voice = self._voices[voice_id]

        # 프롬프트 파일 존재 확인
        if voice.prompt_wav:
            path = Path(voice.prompt_wav)
            if not path.exists():
                errors.append(
                    f"Prompt WAV file not found: {voice.prompt_wav}"
                )

        # 프롬프트 텍스트 확인
        if voice.prompt_wav and not voice.prompt_text:
            errors.append("Prompt WAV specified but prompt_text is empty")

        # sample_rate 검증
        valid_rates = [16000, 22050, 24000, 44100, 48000]
        if voice.sample_rate not in valid_rates:
            errors.append(
                f"Invalid sample_rate: {voice.sample_rate}. "
                f"Must be one of {valid_rates}"
            )

        return errors

    def validate_all_voices(self) -> Dict[str, List[str]]:
        """모든 음성의 유효성을 검증합니다.

        Returns:
            {voice_id: [errors]} 딕셔너리
        """
        results = {}

        for voice_id in self._voices:
            errors = self.validate_voice_config(voice_id)
            if errors:
                results[voice_id] = errors

        return results

    def get_status(self) -> dict:
        """매니저 상태를 반환합니다."""
        return {
            "total_voices": len(self._voices),
            "enabled_voices": self._count_enabled_voices(),
            "ready_voices": len(self._ready_voices),
            "default_voice_id": self._default_voice_id,
            "fallback_voice_id": self._fallback_voice_id,
            "languages": self.get_languages(),
        }

    @property
    def default_voice_id(self) -> Optional[str]:
        """기본 음성 ID."""
        return self._default_voice_id

    @property
    def fallback_voice_id(self) -> Optional[str]:
        """폴백 음성 ID."""
        return self._fallback_voice_id

    def __len__(self) -> int:
        """등록된 음성 수."""
        return len(self._voices)

    def __contains__(self, voice_id: str) -> bool:
        """in 연산자 지원."""
        return voice_id in self._voices

    # ==================== Admin Management Methods ====================

    def list_voices_by_model(self, model_instance: str) -> List[VoiceData]:
        """특정 모델 인스턴스에 연결된 음성 목록을 반환합니다.

        Args:
            model_instance: 모델 인스턴스 이름

        Returns:
            VoiceData 목록
        """
        return [
            voice
            for voice in self._voices.values()
            if voice.model_instance == model_instance
        ]

    async def add_voice(
        self,
        voice_id: str,
        model_instance: str,
        prompt_audio_path: str,
        prompt_text: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """새 음성을 동적으로 추가합니다.

        Args:
            voice_id: 새 음성 ID
            model_instance: 연결할 모델 인스턴스
            prompt_audio_path: 프롬프트 오디오 파일 경로
            prompt_text: 프롬프트 텍스트
            metadata: 메타데이터 (name, language, gender, description 등)

        Raises:
            VoiceError: 이미 존재하는 음성 ID인 경우
            VoiceLoadError: 프롬프트 로드 실패
        """
        if voice_id in self._voices:
            raise VoiceError(f"Voice '{voice_id}' already exists")

        metadata = metadata or {}

        # VoiceData 생성
        voice_data = VoiceData(
            voice_id=voice_id,
            name=metadata.get("name", voice_id),
            language=metadata.get("language", "ko"),
            gender=metadata.get("gender", "unknown"),
            model_instance=model_instance,
            description=metadata.get("description", ""),
            sample_rate=metadata.get("sample_rate", 24000),
            enabled=True,
            prompt_wav=prompt_audio_path,
            prompt_text=prompt_text,
            options=metadata.get("options"),
        )

        # 프롬프트 파일 존재 확인
        prompt_path = Path(prompt_audio_path)
        if not prompt_path.exists():
            raise VoiceLoadError(
                voice_id,
                f"Prompt audio file not found: {prompt_audio_path}"
            )

        self._voices[voice_id] = voice_data

        logger.info(
            "Voice added dynamically",
            voice_id=voice_id,
            model_instance=model_instance,
        )

    async def update_voice(
        self,
        voice_id: str,
        metadata: dict,
        prompt_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ) -> bool:
        """음성 메타데이터 및 프롬프트를 수정합니다.

        Args:
            voice_id: 음성 ID
            metadata: 수정할 메타데이터
            prompt_audio_path: 새 프롬프트 오디오 파일 경로 (optional)
            prompt_text: 새 프롬프트 텍스트 (optional)

        Returns:
            bool: 프롬프트가 변경되어 reload가 필요한지 여부

        Raises:
            VoiceNotFoundError: 음성이 없는 경우
            VoiceLoadError: 프롬프트 파일이 없는 경우
        """
        if voice_id not in self._voices:
            raise VoiceNotFoundError(f"Voice '{voice_id}' not found")

        voice = self._voices[voice_id]
        prompt_changed = False

        # 메타데이터 업데이트
        if "name" in metadata:
            voice.name = metadata["name"]
        if "language" in metadata:
            voice.language = metadata["language"]
        if "gender" in metadata:
            voice.gender = metadata["gender"]
        if "description" in metadata:
            voice.description = metadata["description"]
        if "sample_rate" in metadata:
            voice.sample_rate = metadata["sample_rate"]

        # 프롬프트 업데이트
        if prompt_audio_path:
            prompt_path = Path(prompt_audio_path)
            if not prompt_path.exists():
                raise VoiceLoadError(
                    voice_id,
                    f"Prompt audio file not found: {prompt_audio_path}"
                )
            voice.prompt_wav = prompt_audio_path
            prompt_changed = True

        if prompt_text:
            voice.prompt_text = prompt_text
            prompt_changed = True

        # 프롬프트가 변경되면 자동으로 reload 처리
        if prompt_changed:
            self._ready_voices.discard(voice_id)
            voice.prompt_data = None

        updated_fields = list(metadata.keys())
        if prompt_audio_path:
            updated_fields.append("prompt_audio_path")
        if prompt_text:
            updated_fields.append("prompt_text")

        logger.info(
            "Voice updated",
            voice_id=voice_id,
            updated_fields=updated_fields,
            prompt_changed=prompt_changed,
        )

        return prompt_changed

    async def remove_voice(self, voice_id: str) -> None:
        """음성을 제거합니다.

        Args:
            voice_id: 음성 ID

        Raises:
            VoiceNotFoundError: 음성이 없는 경우
        """
        if voice_id not in self._voices:
            raise VoiceNotFoundError(f"Voice '{voice_id}' not found")

        # ready 상태 제거
        self._ready_voices.discard(voice_id)

        # 음성 제거
        del self._voices[voice_id]

        logger.info("Voice removed", voice_id=voice_id)

    async def set_voice_enabled(
        self,
        voice_id: str,
        enabled: bool,
    ) -> None:
        """음성 활성화/비활성화를 설정합니다.

        Args:
            voice_id: 음성 ID
            enabled: 활성화 여부

        Raises:
            VoiceNotFoundError: 음성이 없는 경우
        """
        if voice_id not in self._voices:
            raise VoiceNotFoundError(f"Voice '{voice_id}' not found")

        self._voices[voice_id].enabled = enabled

        if not enabled:
            self._ready_voices.discard(voice_id)

        logger.info(
            "Voice enabled status changed",
            voice_id=voice_id,
            enabled=enabled,
        )

    async def reload_voice(self, voice_id: str) -> None:
        """음성 프롬프트를 리로드합니다.

        프롬프트 데이터를 다시 로드하고 ready 상태를 초기화합니다.

        Args:
            voice_id: 음성 ID

        Raises:
            VoiceNotFoundError: 음성이 없는 경우
            VoiceLoadError: 프롬프트 로드 실패
        """
        if voice_id not in self._voices:
            raise VoiceNotFoundError(f"Voice '{voice_id}' not found")

        voice = self._voices[voice_id]

        # ready 상태 초기화 (프롬프트 재로드 필요)
        self._ready_voices.discard(voice_id)

        # 프롬프트 파일 존재 확인
        if voice.prompt_wav:
            prompt_path = Path(voice.prompt_wav)
            if not prompt_path.exists():
                raise VoiceLoadError(
                    voice_id,
                    f"Prompt audio file not found: {voice.prompt_wav}"
                )

        # 프롬프트 데이터 초기화 (모델에서 다시 로드하도록)
        voice.prompt_data = None

        logger.info(
            "Voice prompt reload requested",
            voice_id=voice_id,
        )
