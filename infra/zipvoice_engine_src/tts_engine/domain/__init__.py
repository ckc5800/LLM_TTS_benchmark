"""Domain module - Domain models and types."""

from tts_engine.domain.synthesis import (
    BatchSynthesisRequest,
    BatchSynthesisResult,
    StreamingChunk,
    SynthesisRequest,
    SynthesisResult,
    generate_request_id,
)
from tts_engine.domain.types import (
    AudioData,
    GPUInfo,
    HealthCheckResult,
    ModelInfo,
    PromptData,
    ServerStats,
)
from tts_engine.domain.voice import (
    VoiceData,
    VoiceListResponse,
)

__all__ = [
    # Synthesis
    "BatchSynthesisRequest",
    "BatchSynthesisResult",
    "StreamingChunk",
    "SynthesisRequest",
    "SynthesisResult",
    "generate_request_id",
    # Types
    "AudioData",
    "GPUInfo",
    "HealthCheckResult",
    "ModelInfo",
    "PromptData",
    "ServerStats",
    # Voice
    "VoiceData",
    "VoiceListResponse",
]
