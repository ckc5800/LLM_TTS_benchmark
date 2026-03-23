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
    GetVoiceMemoryRequest,
    GetVoiceMemoryResponse,
    MemoryBreakdown,
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
    # Model Management (Admin)
    GetLoadedModelsRequest,
    GetLoadedModelsResponse,
    LoadedModelInfo,
    ReloadModelRequest,
    ReloadModelResponse,
    ReloadAllModelsRequest,
    ReloadAllModelsResponse,
    ModelReloadResult,
    # Voice Management (Admin)
    AddVoiceRequest,
    AddVoiceResponse,
    VoiceMetadata,
    UpdateVoiceRequest,
    UpdateVoiceResponse,
    RemoveVoiceRequest,
    RemoveVoiceResponse,
    SetVoiceEnabledRequest,
    SetVoiceEnabledResponse,
    ReloadVoiceRequest,
    ReloadVoiceResponse,
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
    "GetVoiceMemoryRequest",
    "GetVoiceMemoryResponse",
    "MemoryBreakdown",
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
    # Model Management (Admin)
    "GetLoadedModelsRequest",
    "GetLoadedModelsResponse",
    "LoadedModelInfo",
    "ReloadModelRequest",
    "ReloadModelResponse",
    "ReloadAllModelsRequest",
    "ReloadAllModelsResponse",
    "ModelReloadResult",
    # Voice Management (Admin)
    "AddVoiceRequest",
    "AddVoiceResponse",
    "VoiceMetadata",
    "UpdateVoiceRequest",
    "UpdateVoiceResponse",
    "RemoveVoiceRequest",
    "RemoveVoiceResponse",
    "SetVoiceEnabledRequest",
    "SetVoiceEnabledResponse",
    "ReloadVoiceRequest",
    "ReloadVoiceResponse",
    # gRPC
    "TTSServiceServicer",
    "TTSServiceStub",
    "add_TTSServiceServicer_to_server",
]
