"""Synthesis domain models for TTS Engine."""

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from tts_engine.core.constants import AudioFormat, Defaults, SynthesisStatus
from tts_engine.core.exceptions import EmptyTextError, TextTooLongError
from tts_engine.domain.types import AudioData


def generate_request_id() -> str:
    """고유 요청 ID 생성."""
    return f"req-{uuid4().hex[:12]}"


@dataclass(frozen=True)
class SynthesisRequest:
    """합성 요청 데이터.

    불변 객체로 설계하여 스레드 안전성 보장.
    """

    request_id: str
    text: str
    voice_id: str
    speed: float = Defaults.SPEED
    pitch: float = Defaults.PITCH
    volume: float = Defaults.VOLUME
    format: AudioFormat = AudioFormat.WAV
    sample_rate: Optional[int] = None
    enable_streaming: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """초기화 후 타입 검증."""
        # frozen=True이므로 object.__setattr__ 사용
        if not isinstance(self.speed, (int, float)):
            object.__setattr__(self, "speed", float(self.speed))
        if not isinstance(self.pitch, (int, float)):
            object.__setattr__(self, "pitch", float(self.pitch))
        if not isinstance(self.volume, (int, float)):
            object.__setattr__(self, "volume", float(self.volume))

    def validate(self, max_text_length: int = 2000) -> bool:
        """요청 유효성 검증.

        Args:
            max_text_length: 최대 텍스트 길이

        Returns:
            검증 성공 여부

        Raises:
            EmptyTextError: 텍스트가 비어있을 때
            TextTooLongError: 텍스트가 너무 길 때
        """
        text = self.text.strip()

        if not text:
            raise EmptyTextError()

        if len(text) > max_text_length:
            raise TextTooLongError(len(text), max_text_length)

        return True

    @classmethod
    def create(
        cls,
        text: str,
        voice_id: str,
        request_id: Optional[str] = None,
        **kwargs: Any,
    ) -> "SynthesisRequest":
        """팩토리 메서드로 요청 생성.

        Args:
            text: 합성할 텍스트
            voice_id: 음성 ID
            request_id: 요청 ID (없으면 자동 생성)
            **kwargs: 추가 파라미터

        Returns:
            SynthesisRequest 인스턴스
        """
        return cls(
            request_id=request_id or generate_request_id(),
            text=text,
            voice_id=voice_id,
            **kwargs,
        )


@dataclass
class SynthesisResult:
    """합성 결과 데이터."""

    request_id: str
    status: SynthesisStatus
    audio_data: Optional[AudioData] = None
    file_path: Optional[str] = None
    duration_seconds: float = 0.0
    sample_rate: int = Defaults.SAMPLE_RATE
    format: AudioFormat = AudioFormat.WAV
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """성공 여부."""
        return self.status == SynthesisStatus.COMPLETED

    @property
    def rtf(self) -> float:
        """Real-Time Factor (낮을수록 좋음).

        RTF = 처리시간 / 오디오 길이
        RTF < 1.0 이면 실시간보다 빠름
        """
        if self.duration_seconds <= 0:
            return 0.0
        return (self.processing_time_ms / 1000) / self.duration_seconds

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환 (API 응답용)."""
        result = {
            "request_id": self.request_id,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "format": self.format.value,
            "processing_time_ms": self.processing_time_ms,
            "rtf": self.rtf,
        }

        if self.file_path:
            result["file_path"] = self.file_path

        if self.error_message:
            result["error_message"] = self.error_message

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def success(
        cls,
        request_id: str,
        audio_data: AudioData,
        processing_time_ms: float,
        **kwargs: Any,
    ) -> "SynthesisResult":
        """성공 결과 생성."""
        return cls(
            request_id=request_id,
            status=SynthesisStatus.COMPLETED,
            audio_data=audio_data,
            duration_seconds=audio_data.duration_seconds,
            sample_rate=audio_data.sample_rate,
            processing_time_ms=processing_time_ms,
            **kwargs,
        )

    @classmethod
    def failure(
        cls,
        request_id: str,
        error_message: str,
        processing_time_ms: float = 0.0,
        **kwargs: Any,
    ) -> "SynthesisResult":
        """실패 결과 생성."""
        return cls(
            request_id=request_id,
            status=SynthesisStatus.FAILED,
            error_message=error_message,
            processing_time_ms=processing_time_ms,
            **kwargs,
        )


@dataclass
class BatchSynthesisRequest:
    """배치 합성 요청."""

    requests: list[SynthesisRequest]
    batch_id: str = field(default_factory=lambda: f"batch-{uuid4().hex[:8]}")

    @property
    def size(self) -> int:
        """배치 크기."""
        return len(self.requests)

    def group_by_voice(self) -> dict[str, list[SynthesisRequest]]:
        """음성별로 요청 그룹화."""
        groups: dict[str, list[SynthesisRequest]] = {}
        for req in self.requests:
            if req.voice_id not in groups:
                groups[req.voice_id] = []
            groups[req.voice_id].append(req)
        return groups


@dataclass
class BatchSynthesisResult:
    """배치 합성 결과."""

    batch_id: str
    results: list[SynthesisResult]
    total_processing_time_ms: float = 0.0

    @property
    def size(self) -> int:
        """결과 개수."""
        return len(self.results)

    @property
    def success_count(self) -> int:
        """성공 개수."""
        return sum(1 for r in self.results if r.is_success)

    @property
    def failure_count(self) -> int:
        """실패 개수."""
        return self.size - self.success_count

    @property
    def success_rate(self) -> float:
        """성공률 (0.0 ~ 1.0)."""
        if self.size == 0:
            return 0.0
        return self.success_count / self.size

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            "batch_id": self.batch_id,
            "results": [r.to_dict() for r in self.results],
            "total_processing_time_ms": self.total_processing_time_ms,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
        }


@dataclass
class StreamingChunk:
    """스트리밍 오디오 청크."""

    request_id: str
    chunk_index: int
    audio_data: bytes
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
