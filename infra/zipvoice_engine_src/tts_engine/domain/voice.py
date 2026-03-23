"""Voice domain models for TTS Engine."""

from dataclasses import dataclass, field
from typing import Any, Optional

from tts_engine.core.constants import Defaults
from tts_engine.core.exceptions import VoiceError
from tts_engine.domain.types import PromptData


@dataclass
class VoiceData:
    """음성 설정 데이터.

    각 음성은 고유 ID, 모델 인스턴스 매핑, 프롬프트 정보를 가집니다.
    """

    voice_id: str
    name: str
    language: str
    gender: str
    model_instance: str  # model_instances 키 참조
    description: str = ""
    sample_rate: int = Defaults.SAMPLE_RATE
    enabled: bool = True
    prompt_wav: Optional[str] = None
    prompt_text: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)

    # 런타임에 로드되는 프롬프트 데이터
    _prompt_data: Optional[PromptData] = field(default=None, repr=False)

    def validate(self) -> bool:
        """음성 데이터 유효성 검증.

        Returns:
            검증 성공 여부

        Raises:
            VoiceError: 유효성 검증 실패 시
        """
        if not self.voice_id or not self.voice_id.strip():
            raise VoiceError("voice_id는 필수입니다", {"voice_id": self.voice_id})

        if not self.model_instance:
            raise VoiceError(
                "model_instance는 필수입니다",
                {"voice_id": self.voice_id},
            )

        valid_rates = [16000, 22050, 24000, 44100, 48000]
        if self.sample_rate not in valid_rates:
            raise VoiceError(
                f"지원하지 않는 sample_rate: {self.sample_rate}",
                {"voice_id": self.voice_id, "valid_rates": valid_rates},
            )

        return True

    @property
    def prompt_data(self) -> Optional[PromptData]:
        """프롬프트 데이터 반환."""
        return self._prompt_data

    @prompt_data.setter
    def prompt_data(self, value: PromptData) -> None:
        """프롬프트 데이터 설정."""
        self._prompt_data = value

    @property
    def is_prompt_loaded(self) -> bool:
        """프롬프트가 로드되었는지 확인."""
        return self._prompt_data is not None and self._prompt_data.is_loaded

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환 (API 응답용)."""
        return {
            "voice_id": self.voice_id,
            "name": self.name,
            "language": self.language,
            "gender": self.gender,
            "model_instance": self.model_instance,
            "description": self.description,
            "sample_rate": self.sample_rate,
            "enabled": self.enabled,
            "prompt_loaded": self.is_prompt_loaded,
        }


@dataclass
class VoiceListResponse:
    """음성 목록 응답."""

    voices: list[VoiceData]
    total: int
    default_voice_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "voices": [v.to_dict() for v in self.voices],
            "total": self.total,
            "default_voice_id": self.default_voice_id,
        }
