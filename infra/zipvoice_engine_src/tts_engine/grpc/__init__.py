"""Generated gRPC code for TTS service."""

from tts_engine.proto.tts_pb2 import (
    # Enums
    AudioFormat,
    ErrorCode,
    BatchStatus,
    HealthStatus,
    # Common
    ErrorDetail,
    # Synthesis
    SynthesizeRequest,
    SynthesizeResponse,
    AudioChunk,
    # Voice
    GetVoicesRequest,
    GetVoicesResponse,
    GetVoiceInfoRequest,
    GetVoiceInfoResponse,
    VoiceInfo,
    # Health
    HealthCheckRequest,
    HealthCheckResponse,
    ModelHealthStatus,
    ModelHealthChecks,
    GPUStatus,
    # Warmup
    WarmupRequest,
    WarmupResponse,
    WarmupAllRequest,
    WarmupAllResponse,
    WarmupResult,
    # Batch
    BatchSynthesizeRequest,
    BatchSynthesizeResponse,
    BatchStatusRequest,
    BatchStatusResponse,
    BatchItemResult,
    # Stats
    GetGpuStatusRequest,
    GetGpuStatusResponse,
    GetServerStatsRequest,
    ServerStatsResponse,
    WindowStats,
    GetPoolStatusRequest,
    PoolStatusResponse,
)

from tts_engine.proto.tts_pb2_grpc import (
    TTSServiceServicer,
    TTSServiceStub,
    add_TTSServiceServicer_to_server,
)

__all__ = [
    # Enums
    "AudioFormat",
    "ErrorCode",
    "BatchStatus",
    "HealthStatus",
    # Common
    "ErrorDetail",
    # Synthesis
    "SynthesizeRequest",
    "SynthesizeResponse",
    "AudioChunk",
    # Voice
    "GetVoicesRequest",
    "GetVoicesResponse",
    "GetVoiceInfoRequest",
    "GetVoiceInfoResponse",
    "VoiceInfo",
    # Health
    "HealthCheckRequest",
    "HealthCheckResponse",
    "ModelHealthStatus",
    "ModelHealthChecks",
    "GPUStatus",
    # Warmup
    "WarmupRequest",
    "WarmupResponse",
    "WarmupAllRequest",
    "WarmupAllResponse",
    "WarmupResult",
    # Batch
    "BatchSynthesizeRequest",
    "BatchSynthesizeResponse",
    "BatchStatusRequest",
    "BatchStatusResponse",
    "BatchItemResult",
    # Stats
    "GetGpuStatusRequest",
    "GetGpuStatusResponse",
    "GetServerStatsRequest",
    "ServerStatsResponse",
    "WindowStats",
    "GetPoolStatusRequest",
    "PoolStatusResponse",
    # gRPC
    "TTSServiceServicer",
    "TTSServiceStub",
    "add_TTSServiceServicer_to_server",
]
