from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AudioFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    AUDIO_FORMAT_UNKNOWN: _ClassVar[AudioFormat]
    AUDIO_FORMAT_WAV: _ClassVar[AudioFormat]
    AUDIO_FORMAT_MP3: _ClassVar[AudioFormat]
    AUDIO_FORMAT_OGG: _ClassVar[AudioFormat]
    AUDIO_FORMAT_PCM: _ClassVar[AudioFormat]
    AUDIO_FORMAT_FLAC: _ClassVar[AudioFormat]

class ErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ERROR_UNKNOWN: _ClassVar[ErrorCode]
    ERROR_VOICE_NOT_FOUND: _ClassVar[ErrorCode]
    ERROR_INVALID_TEXT: _ClassVar[ErrorCode]
    ERROR_TEXT_TOO_LONG: _ClassVar[ErrorCode]
    ERROR_EMPTY_TEXT: _ClassVar[ErrorCode]
    ERROR_MODEL_NOT_LOADED: _ClassVar[ErrorCode]
    ERROR_MODEL_NOT_FOUND: _ClassVar[ErrorCode]
    ERROR_SYNTHESIS_FAILED: _ClassVar[ErrorCode]
    ERROR_GPU_MEMORY_EXCEEDED: _ClassVar[ErrorCode]
    ERROR_TIMEOUT: _ClassVar[ErrorCode]
    ERROR_INVALID_FORMAT: _ClassVar[ErrorCode]
    ERROR_INVALID_PARAMETERS: _ClassVar[ErrorCode]
    ERROR_BATCH_NOT_FOUND: _ClassVar[ErrorCode]
    ERROR_POOL_EXHAUSTED: _ClassVar[ErrorCode]
    ERROR_VOICE_NOT_READY: _ClassVar[ErrorCode]
    ERROR_INTERNAL: _ClassVar[ErrorCode]

class BatchStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BATCH_STATUS_UNKNOWN: _ClassVar[BatchStatus]
    BATCH_STATUS_QUEUED: _ClassVar[BatchStatus]
    BATCH_STATUS_PROCESSING: _ClassVar[BatchStatus]
    BATCH_STATUS_COMPLETED: _ClassVar[BatchStatus]
    BATCH_STATUS_FAILED: _ClassVar[BatchStatus]
    BATCH_STATUS_CANCELLED: _ClassVar[BatchStatus]
    BATCH_STATUS_PARTIAL: _ClassVar[BatchStatus]

class HealthStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    HEALTH_STATUS_UNKNOWN: _ClassVar[HealthStatus]
    HEALTH_STATUS_HEALTHY: _ClassVar[HealthStatus]
    HEALTH_STATUS_DEGRADED: _ClassVar[HealthStatus]
    HEALTH_STATUS_UNHEALTHY: _ClassVar[HealthStatus]
AUDIO_FORMAT_UNKNOWN: AudioFormat
AUDIO_FORMAT_WAV: AudioFormat
AUDIO_FORMAT_MP3: AudioFormat
AUDIO_FORMAT_OGG: AudioFormat
AUDIO_FORMAT_PCM: AudioFormat
AUDIO_FORMAT_FLAC: AudioFormat
ERROR_UNKNOWN: ErrorCode
ERROR_VOICE_NOT_FOUND: ErrorCode
ERROR_INVALID_TEXT: ErrorCode
ERROR_TEXT_TOO_LONG: ErrorCode
ERROR_EMPTY_TEXT: ErrorCode
ERROR_MODEL_NOT_LOADED: ErrorCode
ERROR_MODEL_NOT_FOUND: ErrorCode
ERROR_SYNTHESIS_FAILED: ErrorCode
ERROR_GPU_MEMORY_EXCEEDED: ErrorCode
ERROR_TIMEOUT: ErrorCode
ERROR_INVALID_FORMAT: ErrorCode
ERROR_INVALID_PARAMETERS: ErrorCode
ERROR_BATCH_NOT_FOUND: ErrorCode
ERROR_POOL_EXHAUSTED: ErrorCode
ERROR_VOICE_NOT_READY: ErrorCode
ERROR_INTERNAL: ErrorCode
BATCH_STATUS_UNKNOWN: BatchStatus
BATCH_STATUS_QUEUED: BatchStatus
BATCH_STATUS_PROCESSING: BatchStatus
BATCH_STATUS_COMPLETED: BatchStatus
BATCH_STATUS_FAILED: BatchStatus
BATCH_STATUS_CANCELLED: BatchStatus
BATCH_STATUS_PARTIAL: BatchStatus
HEALTH_STATUS_UNKNOWN: HealthStatus
HEALTH_STATUS_HEALTHY: HealthStatus
HEALTH_STATUS_DEGRADED: HealthStatus
HEALTH_STATUS_UNHEALTHY: HealthStatus

class ErrorDetail(_message.Message):
    __slots__ = ("code", "message", "request_id", "timestamp_ms", "details")
    class DetailsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    code: ErrorCode
    message: str
    request_id: str
    timestamp_ms: int
    details: _containers.ScalarMap[str, str]
    def __init__(self, code: _Optional[_Union[ErrorCode, str]] = ..., message: _Optional[str] = ..., request_id: _Optional[str] = ..., timestamp_ms: _Optional[int] = ..., details: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SynthesizeRequest(_message.Message):
    __slots__ = ("request_id", "text", "voice_id", "speed", "pitch", "volume", "format", "sample_rate", "timestamp_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    PITCH_FIELD_NUMBER: _ClassVar[int]
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    text: str
    voice_id: str
    speed: float
    pitch: float
    volume: float
    format: AudioFormat
    sample_rate: int
    timestamp_ms: int
    def __init__(self, request_id: _Optional[str] = ..., text: _Optional[str] = ..., voice_id: _Optional[str] = ..., speed: _Optional[float] = ..., pitch: _Optional[float] = ..., volume: _Optional[float] = ..., format: _Optional[_Union[AudioFormat, str]] = ..., sample_rate: _Optional[int] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class SynthesizeResponse(_message.Message):
    __slots__ = ("request_id", "success", "audio_data", "format", "sample_rate", "duration_seconds", "data_size", "processing_time_ms", "timestamp_ms", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    DATA_SIZE_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    success: bool
    audio_data: bytes
    format: AudioFormat
    sample_rate: int
    duration_seconds: float
    data_size: int
    processing_time_ms: int
    timestamp_ms: int
    error: ErrorDetail
    def __init__(self, request_id: _Optional[str] = ..., success: bool = ..., audio_data: _Optional[bytes] = ..., format: _Optional[_Union[AudioFormat, str]] = ..., sample_rate: _Optional[int] = ..., duration_seconds: _Optional[float] = ..., data_size: _Optional[int] = ..., processing_time_ms: _Optional[int] = ..., timestamp_ms: _Optional[int] = ..., error: _Optional[_Union[ErrorDetail, _Mapping]] = ...) -> None: ...

class AudioChunk(_message.Message):
    __slots__ = ("request_id", "chunk_data", "chunk_index", "is_last", "chunk_size", "timestamp_ms", "has_error", "error_code", "error_message")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_DATA_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_FIELD_NUMBER: _ClassVar[int]
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    HAS_ERROR_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    chunk_data: bytes
    chunk_index: int
    is_last: bool
    chunk_size: int
    timestamp_ms: int
    has_error: bool
    error_code: ErrorCode
    error_message: str
    def __init__(self, request_id: _Optional[str] = ..., chunk_data: _Optional[bytes] = ..., chunk_index: _Optional[int] = ..., is_last: bool = ..., chunk_size: _Optional[int] = ..., timestamp_ms: _Optional[int] = ..., has_error: bool = ..., error_code: _Optional[_Union[ErrorCode, str]] = ..., error_message: _Optional[str] = ...) -> None: ...

class GetVoicesRequest(_message.Message):
    __slots__ = ("request_id", "language", "gender", "enabled_only")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    ENABLED_ONLY_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    language: str
    gender: str
    enabled_only: bool
    def __init__(self, request_id: _Optional[str] = ..., language: _Optional[str] = ..., gender: _Optional[str] = ..., enabled_only: bool = ...) -> None: ...

class GetVoicesResponse(_message.Message):
    __slots__ = ("request_id", "voices", "total_count", "default_voice_id", "timestamp_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    VOICES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    DEFAULT_VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    voices: _containers.RepeatedCompositeFieldContainer[VoiceInfo]
    total_count: int
    default_voice_id: str
    timestamp_ms: int
    def __init__(self, request_id: _Optional[str] = ..., voices: _Optional[_Iterable[_Union[VoiceInfo, _Mapping]]] = ..., total_count: _Optional[int] = ..., default_voice_id: _Optional[str] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class GetVoiceInfoRequest(_message.Message):
    __slots__ = ("request_id", "voice_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    voice_id: str
    def __init__(self, request_id: _Optional[str] = ..., voice_id: _Optional[str] = ...) -> None: ...

class GetVoiceInfoResponse(_message.Message):
    __slots__ = ("request_id", "found", "voice", "error", "timestamp_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    FOUND_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    found: bool
    voice: VoiceInfo
    error: ErrorDetail
    timestamp_ms: int
    def __init__(self, request_id: _Optional[str] = ..., found: bool = ..., voice: _Optional[_Union[VoiceInfo, _Mapping]] = ..., error: _Optional[_Union[ErrorDetail, _Mapping]] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class VoiceInfo(_message.Message):
    __slots__ = ("voice_id", "name", "language", "gender", "sample_rate", "description", "enabled", "ready", "model_instance", "supported_formats", "preview_url", "memory_usage_mb", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    READY_FIELD_NUMBER: _ClassVar[int]
    MODEL_INSTANCE_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_FORMATS_FIELD_NUMBER: _ClassVar[int]
    PREVIEW_URL_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USAGE_MB_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    voice_id: str
    name: str
    language: str
    gender: str
    sample_rate: int
    description: str
    enabled: bool
    ready: bool
    model_instance: str
    supported_formats: _containers.RepeatedScalarFieldContainer[AudioFormat]
    preview_url: str
    memory_usage_mb: int
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, voice_id: _Optional[str] = ..., name: _Optional[str] = ..., language: _Optional[str] = ..., gender: _Optional[str] = ..., sample_rate: _Optional[int] = ..., description: _Optional[str] = ..., enabled: bool = ..., ready: bool = ..., model_instance: _Optional[str] = ..., supported_formats: _Optional[_Iterable[_Union[AudioFormat, str]]] = ..., preview_url: _Optional[str] = ..., memory_usage_mb: _Optional[int] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ("request_id", "model_name", "include_details")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_DETAILS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    model_name: str
    include_details: bool
    def __init__(self, request_id: _Optional[str] = ..., model_name: _Optional[str] = ..., include_details: bool = ...) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("request_id", "healthy", "status", "status_message", "models", "timestamp_ms", "uptime_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    UPTIME_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    healthy: bool
    status: HealthStatus
    status_message: str
    models: _containers.RepeatedCompositeFieldContainer[ModelHealthStatus]
    timestamp_ms: int
    uptime_ms: int
    def __init__(self, request_id: _Optional[str] = ..., healthy: bool = ..., status: _Optional[_Union[HealthStatus, str]] = ..., status_message: _Optional[str] = ..., models: _Optional[_Iterable[_Union[ModelHealthStatus, _Mapping]]] = ..., timestamp_ms: _Optional[int] = ..., uptime_ms: _Optional[int] = ...) -> None: ...

class ModelHealthStatus(_message.Message):
    __slots__ = ("model_name", "model_type", "healthy", "status", "status_message", "checks", "loaded_voices", "gpu_status", "error", "pool_total", "pool_available", "pool_in_use")
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CHECKS_FIELD_NUMBER: _ClassVar[int]
    LOADED_VOICES_FIELD_NUMBER: _ClassVar[int]
    GPU_STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    POOL_TOTAL_FIELD_NUMBER: _ClassVar[int]
    POOL_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    POOL_IN_USE_FIELD_NUMBER: _ClassVar[int]
    model_name: str
    model_type: str
    healthy: bool
    status: HealthStatus
    status_message: str
    checks: ModelHealthChecks
    loaded_voices: _containers.RepeatedScalarFieldContainer[str]
    gpu_status: GPUStatus
    error: ErrorDetail
    pool_total: int
    pool_available: int
    pool_in_use: int
    def __init__(self, model_name: _Optional[str] = ..., model_type: _Optional[str] = ..., healthy: bool = ..., status: _Optional[_Union[HealthStatus, str]] = ..., status_message: _Optional[str] = ..., checks: _Optional[_Union[ModelHealthChecks, _Mapping]] = ..., loaded_voices: _Optional[_Iterable[str]] = ..., gpu_status: _Optional[_Union[GPUStatus, _Mapping]] = ..., error: _Optional[_Union[ErrorDetail, _Mapping]] = ..., pool_total: _Optional[int] = ..., pool_available: _Optional[int] = ..., pool_in_use: _Optional[int] = ...) -> None: ...

class GetVoiceMemoryRequest(_message.Message):
    __slots__ = ("voice_id", "request_id")
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    voice_id: str
    request_id: str
    def __init__(self, voice_id: _Optional[str] = ..., request_id: _Optional[str] = ...) -> None: ...

class GetVoiceMemoryResponse(_message.Message):
    __slots__ = ("request_id", "voice_id", "prompt_memory_mb", "model_shared_memory_mb", "total_memory_mb", "memory_breakdown", "timestamp_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    PROMPT_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    MODEL_SHARED_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BREAKDOWN_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    voice_id: str
    prompt_memory_mb: int
    model_shared_memory_mb: int
    total_memory_mb: int
    memory_breakdown: MemoryBreakdown
    timestamp_ms: int
    def __init__(self, request_id: _Optional[str] = ..., voice_id: _Optional[str] = ..., prompt_memory_mb: _Optional[int] = ..., model_shared_memory_mb: _Optional[int] = ..., total_memory_mb: _Optional[int] = ..., memory_breakdown: _Optional[_Union[MemoryBreakdown, _Mapping]] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class MemoryBreakdown(_message.Message):
    __slots__ = ("wav_tensor_mb", "features_mb", "tokens_mb", "text_mb", "other_mb")
    WAV_TENSOR_MB_FIELD_NUMBER: _ClassVar[int]
    FEATURES_MB_FIELD_NUMBER: _ClassVar[int]
    TOKENS_MB_FIELD_NUMBER: _ClassVar[int]
    TEXT_MB_FIELD_NUMBER: _ClassVar[int]
    OTHER_MB_FIELD_NUMBER: _ClassVar[int]
    wav_tensor_mb: int
    features_mb: int
    tokens_mb: int
    text_mb: int
    other_mb: int
    def __init__(self, wav_tensor_mb: _Optional[int] = ..., features_mb: _Optional[int] = ..., tokens_mb: _Optional[int] = ..., text_mb: _Optional[int] = ..., other_mb: _Optional[int] = ...) -> None: ...

class ModelHealthChecks(_message.Message):
    __slots__ = ("model_loaded", "voices_loaded", "config_valid", "gpu_available", "pool_healthy")
    MODEL_LOADED_FIELD_NUMBER: _ClassVar[int]
    VOICES_LOADED_FIELD_NUMBER: _ClassVar[int]
    CONFIG_VALID_FIELD_NUMBER: _ClassVar[int]
    GPU_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    POOL_HEALTHY_FIELD_NUMBER: _ClassVar[int]
    model_loaded: bool
    voices_loaded: bool
    config_valid: bool
    gpu_available: bool
    pool_healthy: bool
    def __init__(self, model_loaded: bool = ..., voices_loaded: bool = ..., config_valid: bool = ..., gpu_available: bool = ..., pool_healthy: bool = ...) -> None: ...

class GPUStatus(_message.Message):
    __slots__ = ("available", "device", "device_id", "device_name", "total_memory_mb", "used_memory_mb", "free_memory_mb", "utilization_percent", "memory_percent", "temperature_c", "driver_version", "cuda_version", "error")
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    TOTAL_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    USED_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    FREE_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    UTILIZATION_PERCENT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_PERCENT_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_C_FIELD_NUMBER: _ClassVar[int]
    DRIVER_VERSION_FIELD_NUMBER: _ClassVar[int]
    CUDA_VERSION_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    available: bool
    device: str
    device_id: int
    device_name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization_percent: int
    memory_percent: int
    temperature_c: int
    driver_version: str
    cuda_version: str
    error: str
    def __init__(self, available: bool = ..., device: _Optional[str] = ..., device_id: _Optional[int] = ..., device_name: _Optional[str] = ..., total_memory_mb: _Optional[int] = ..., used_memory_mb: _Optional[int] = ..., free_memory_mb: _Optional[int] = ..., utilization_percent: _Optional[int] = ..., memory_percent: _Optional[int] = ..., temperature_c: _Optional[int] = ..., driver_version: _Optional[str] = ..., cuda_version: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...

class WarmupRequest(_message.Message):
    __slots__ = ("request_id", "model_instance", "test_text", "repeat_count")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_INSTANCE_FIELD_NUMBER: _ClassVar[int]
    TEST_TEXT_FIELD_NUMBER: _ClassVar[int]
    REPEAT_COUNT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    model_instance: str
    test_text: str
    repeat_count: int
    def __init__(self, request_id: _Optional[str] = ..., model_instance: _Optional[str] = ..., test_text: _Optional[str] = ..., repeat_count: _Optional[int] = ...) -> None: ...

class WarmupResponse(_message.Message):
    __slots__ = ("request_id", "success", "model_instance", "message", "duration_ms", "error", "timestamp_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MODEL_INSTANCE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    success: bool
    model_instance: str
    message: str
    duration_ms: int
    error: ErrorDetail
    timestamp_ms: int
    def __init__(self, request_id: _Optional[str] = ..., success: bool = ..., model_instance: _Optional[str] = ..., message: _Optional[str] = ..., duration_ms: _Optional[int] = ..., error: _Optional[_Union[ErrorDetail, _Mapping]] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class WarmupAllRequest(_message.Message):
    __slots__ = ("request_id", "test_text", "repeat_count")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TEST_TEXT_FIELD_NUMBER: _ClassVar[int]
    REPEAT_COUNT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    test_text: str
    repeat_count: int
    def __init__(self, request_id: _Optional[str] = ..., test_text: _Optional[str] = ..., repeat_count: _Optional[int] = ...) -> None: ...

class WarmupAllResponse(_message.Message):
    __slots__ = ("request_id", "overall_success", "message", "results", "total_duration_ms", "timestamp_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    OVERALL_SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    overall_success: bool
    message: str
    results: _containers.RepeatedCompositeFieldContainer[WarmupResult]
    total_duration_ms: int
    timestamp_ms: int
    def __init__(self, request_id: _Optional[str] = ..., overall_success: bool = ..., message: _Optional[str] = ..., results: _Optional[_Iterable[_Union[WarmupResult, _Mapping]]] = ..., total_duration_ms: _Optional[int] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class WarmupResult(_message.Message):
    __slots__ = ("model_instance", "success", "message", "duration_ms", "error")
    MODEL_INSTANCE_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    model_instance: str
    success: bool
    message: str
    duration_ms: int
    error: ErrorDetail
    def __init__(self, model_instance: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ..., duration_ms: _Optional[int] = ..., error: _Optional[_Union[ErrorDetail, _Mapping]] = ...) -> None: ...

class BatchSynthesizeRequest(_message.Message):
    __slots__ = ("request_id", "requests", "priority", "max_concurrent")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    MAX_CONCURRENT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    requests: _containers.RepeatedCompositeFieldContainer[SynthesizeRequest]
    priority: int
    max_concurrent: int
    def __init__(self, request_id: _Optional[str] = ..., requests: _Optional[_Iterable[_Union[SynthesizeRequest, _Mapping]]] = ..., priority: _Optional[int] = ..., max_concurrent: _Optional[int] = ...) -> None: ...

class BatchSynthesizeResponse(_message.Message):
    __slots__ = ("request_id", "batch_id", "status", "message", "total_count", "timestamp_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    batch_id: str
    status: BatchStatus
    message: str
    total_count: int
    timestamp_ms: int
    def __init__(self, request_id: _Optional[str] = ..., batch_id: _Optional[str] = ..., status: _Optional[_Union[BatchStatus, str]] = ..., message: _Optional[str] = ..., total_count: _Optional[int] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class BatchStatusRequest(_message.Message):
    __slots__ = ("request_id", "batch_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    batch_id: str
    def __init__(self, request_id: _Optional[str] = ..., batch_id: _Optional[str] = ...) -> None: ...

class BatchStatusResponse(_message.Message):
    __slots__ = ("request_id", "batch_id", "status", "progress_percent", "total_count", "completed_count", "failed_count", "processing_time_ms", "estimated_remaining_ms", "message", "results", "timestamp_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_PERCENT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_COUNT_FIELD_NUMBER: _ClassVar[int]
    FAILED_COUNT_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    ESTIMATED_REMAINING_MS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    batch_id: str
    status: BatchStatus
    progress_percent: int
    total_count: int
    completed_count: int
    failed_count: int
    processing_time_ms: int
    estimated_remaining_ms: int
    message: str
    results: _containers.RepeatedCompositeFieldContainer[BatchItemResult]
    timestamp_ms: int
    def __init__(self, request_id: _Optional[str] = ..., batch_id: _Optional[str] = ..., status: _Optional[_Union[BatchStatus, str]] = ..., progress_percent: _Optional[int] = ..., total_count: _Optional[int] = ..., completed_count: _Optional[int] = ..., failed_count: _Optional[int] = ..., processing_time_ms: _Optional[int] = ..., estimated_remaining_ms: _Optional[int] = ..., message: _Optional[str] = ..., results: _Optional[_Iterable[_Union[BatchItemResult, _Mapping]]] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class BatchItemResult(_message.Message):
    __slots__ = ("request_id", "success", "file_path", "data_size", "duration_seconds", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    DATA_SIZE_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    success: bool
    file_path: str
    data_size: int
    duration_seconds: float
    error: ErrorDetail
    def __init__(self, request_id: _Optional[str] = ..., success: bool = ..., file_path: _Optional[str] = ..., data_size: _Optional[int] = ..., duration_seconds: _Optional[float] = ..., error: _Optional[_Union[ErrorDetail, _Mapping]] = ...) -> None: ...

class GetGpuStatusRequest(_message.Message):
    __slots__ = ("request_id", "device_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    device_id: int
    def __init__(self, request_id: _Optional[str] = ..., device_id: _Optional[int] = ...) -> None: ...

class GetGpuStatusResponse(_message.Message):
    __slots__ = ("request_id", "gpus", "gpu_count", "timestamp_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    GPUS_FIELD_NUMBER: _ClassVar[int]
    GPU_COUNT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    gpus: _containers.RepeatedCompositeFieldContainer[GPUStatus]
    gpu_count: int
    timestamp_ms: int
    def __init__(self, request_id: _Optional[str] = ..., gpus: _Optional[_Iterable[_Union[GPUStatus, _Mapping]]] = ..., gpu_count: _Optional[int] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class GetServerStatsRequest(_message.Message):
    __slots__ = ("request_id", "include_history")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_HISTORY_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    include_history: bool
    def __init__(self, request_id: _Optional[str] = ..., include_history: bool = ...) -> None: ...

class ServerStatsResponse(_message.Message):
    __slots__ = ("request_id", "active_requests", "queued_requests", "active_batches", "total_requests", "successful_requests", "failed_requests", "avg_processing_time_ms", "min_processing_time_ms", "max_processing_time_ms", "requests_per_second", "uptime_seconds", "start_time_ms", "version", "model_pool_total", "model_pool_available", "timestamp_ms", "stats_1min", "stats_5min", "stats_1hour")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    QUEUED_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_BATCHES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    SUCCESSFUL_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    FAILED_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    AVG_PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    MIN_PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    REQUESTS_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    START_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    MODEL_POOL_TOTAL_FIELD_NUMBER: _ClassVar[int]
    MODEL_POOL_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    STATS_1MIN_FIELD_NUMBER: _ClassVar[int]
    STATS_5MIN_FIELD_NUMBER: _ClassVar[int]
    STATS_1HOUR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    active_requests: int
    queued_requests: int
    active_batches: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_processing_time_ms: int
    min_processing_time_ms: int
    max_processing_time_ms: int
    requests_per_second: float
    uptime_seconds: int
    start_time_ms: int
    version: str
    model_pool_total: int
    model_pool_available: int
    timestamp_ms: int
    stats_1min: WindowStats
    stats_5min: WindowStats
    stats_1hour: WindowStats
    def __init__(self, request_id: _Optional[str] = ..., active_requests: _Optional[int] = ..., queued_requests: _Optional[int] = ..., active_batches: _Optional[int] = ..., total_requests: _Optional[int] = ..., successful_requests: _Optional[int] = ..., failed_requests: _Optional[int] = ..., avg_processing_time_ms: _Optional[int] = ..., min_processing_time_ms: _Optional[int] = ..., max_processing_time_ms: _Optional[int] = ..., requests_per_second: _Optional[float] = ..., uptime_seconds: _Optional[int] = ..., start_time_ms: _Optional[int] = ..., version: _Optional[str] = ..., model_pool_total: _Optional[int] = ..., model_pool_available: _Optional[int] = ..., timestamp_ms: _Optional[int] = ..., stats_1min: _Optional[_Union[WindowStats, _Mapping]] = ..., stats_5min: _Optional[_Union[WindowStats, _Mapping]] = ..., stats_1hour: _Optional[_Union[WindowStats, _Mapping]] = ...) -> None: ...

class GetPoolStatusRequest(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class WindowStats(_message.Message):
    __slots__ = ("total", "success", "failed", "success_rate", "avg_time_ms", "min_time_ms", "max_time_ms", "requests_per_second")
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    FAILED_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_RATE_FIELD_NUMBER: _ClassVar[int]
    AVG_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    MIN_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    REQUESTS_PER_SECOND_FIELD_NUMBER: _ClassVar[int]
    total: int
    success: int
    failed: int
    success_rate: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    requests_per_second: float
    def __init__(self, total: _Optional[int] = ..., success: _Optional[int] = ..., failed: _Optional[int] = ..., success_rate: _Optional[float] = ..., avg_time_ms: _Optional[float] = ..., min_time_ms: _Optional[float] = ..., max_time_ms: _Optional[float] = ..., requests_per_second: _Optional[float] = ...) -> None: ...

class PoolStatusResponse(_message.Message):
    __slots__ = ("request_id", "runtime", "total_slots", "active_count", "waiting_count", "available_slots", "thread_pool_size", "workers_alive", "submitted_count", "completed_count", "failed_count", "synthesis_active", "synthesis_total", "synthesis_success", "synthesis_failed", "uptime_seconds", "overall_rps", "stats_1min", "stats_5min", "stats_1hour", "batch_enabled", "batch_submitted", "batch_processed", "queue_size", "max_queue_size", "queue_utilization", "queue_rejected", "queue_timeout", "prometheus_enabled", "timestamp_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SLOTS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_COUNT_FIELD_NUMBER: _ClassVar[int]
    WAITING_COUNT_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_SLOTS_FIELD_NUMBER: _ClassVar[int]
    THREAD_POOL_SIZE_FIELD_NUMBER: _ClassVar[int]
    WORKERS_ALIVE_FIELD_NUMBER: _ClassVar[int]
    SUBMITTED_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_COUNT_FIELD_NUMBER: _ClassVar[int]
    FAILED_COUNT_FIELD_NUMBER: _ClassVar[int]
    SYNTHESIS_ACTIVE_FIELD_NUMBER: _ClassVar[int]
    SYNTHESIS_TOTAL_FIELD_NUMBER: _ClassVar[int]
    SYNTHESIS_SUCCESS_FIELD_NUMBER: _ClassVar[int]
    SYNTHESIS_FAILED_FIELD_NUMBER: _ClassVar[int]
    UPTIME_SECONDS_FIELD_NUMBER: _ClassVar[int]
    OVERALL_RPS_FIELD_NUMBER: _ClassVar[int]
    STATS_1MIN_FIELD_NUMBER: _ClassVar[int]
    STATS_5MIN_FIELD_NUMBER: _ClassVar[int]
    STATS_1HOUR_FIELD_NUMBER: _ClassVar[int]
    BATCH_ENABLED_FIELD_NUMBER: _ClassVar[int]
    BATCH_SUBMITTED_FIELD_NUMBER: _ClassVar[int]
    BATCH_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    QUEUE_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAX_QUEUE_SIZE_FIELD_NUMBER: _ClassVar[int]
    QUEUE_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    QUEUE_REJECTED_FIELD_NUMBER: _ClassVar[int]
    QUEUE_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    PROMETHEUS_ENABLED_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    runtime: str
    total_slots: int
    active_count: int
    waiting_count: int
    available_slots: int
    thread_pool_size: int
    workers_alive: int
    submitted_count: int
    completed_count: int
    failed_count: int
    synthesis_active: int
    synthesis_total: int
    synthesis_success: int
    synthesis_failed: int
    uptime_seconds: float
    overall_rps: float
    stats_1min: WindowStats
    stats_5min: WindowStats
    stats_1hour: WindowStats
    batch_enabled: bool
    batch_submitted: int
    batch_processed: int
    queue_size: int
    max_queue_size: int
    queue_utilization: float
    queue_rejected: int
    queue_timeout: int
    prometheus_enabled: bool
    timestamp_ms: int
    def __init__(self, request_id: _Optional[str] = ..., runtime: _Optional[str] = ..., total_slots: _Optional[int] = ..., active_count: _Optional[int] = ..., waiting_count: _Optional[int] = ..., available_slots: _Optional[int] = ..., thread_pool_size: _Optional[int] = ..., workers_alive: _Optional[int] = ..., submitted_count: _Optional[int] = ..., completed_count: _Optional[int] = ..., failed_count: _Optional[int] = ..., synthesis_active: _Optional[int] = ..., synthesis_total: _Optional[int] = ..., synthesis_success: _Optional[int] = ..., synthesis_failed: _Optional[int] = ..., uptime_seconds: _Optional[float] = ..., overall_rps: _Optional[float] = ..., stats_1min: _Optional[_Union[WindowStats, _Mapping]] = ..., stats_5min: _Optional[_Union[WindowStats, _Mapping]] = ..., stats_1hour: _Optional[_Union[WindowStats, _Mapping]] = ..., batch_enabled: bool = ..., batch_submitted: _Optional[int] = ..., batch_processed: _Optional[int] = ..., queue_size: _Optional[int] = ..., max_queue_size: _Optional[int] = ..., queue_utilization: _Optional[float] = ..., queue_rejected: _Optional[int] = ..., queue_timeout: _Optional[int] = ..., prometheus_enabled: bool = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class GetLoadedModelsRequest(_message.Message):
    __slots__ = ("request_id",)
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    def __init__(self, request_id: _Optional[str] = ...) -> None: ...

class GetLoadedModelsResponse(_message.Message):
    __slots__ = ("request_id", "models", "timestamp_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    models: _containers.RepeatedCompositeFieldContainer[LoadedModelInfo]
    timestamp_ms: int
    def __init__(self, request_id: _Optional[str] = ..., models: _Optional[_Iterable[_Union[LoadedModelInfo, _Mapping]]] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class LoadedModelInfo(_message.Message):
    __slots__ = ("model_instance", "model_type", "model_path", "status", "loaded_at_ms", "memory_usage_mb", "voice_count", "loaded_voices", "supports_speed", "supports_pitch", "supports_volume")
    MODEL_INSTANCE_FIELD_NUMBER: _ClassVar[int]
    MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LOADED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USAGE_MB_FIELD_NUMBER: _ClassVar[int]
    VOICE_COUNT_FIELD_NUMBER: _ClassVar[int]
    LOADED_VOICES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_SPEED_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_PITCH_FIELD_NUMBER: _ClassVar[int]
    SUPPORTS_VOLUME_FIELD_NUMBER: _ClassVar[int]
    model_instance: str
    model_type: str
    model_path: str
    status: str
    loaded_at_ms: int
    memory_usage_mb: int
    voice_count: int
    loaded_voices: _containers.RepeatedScalarFieldContainer[str]
    supports_speed: bool
    supports_pitch: bool
    supports_volume: bool
    def __init__(self, model_instance: _Optional[str] = ..., model_type: _Optional[str] = ..., model_path: _Optional[str] = ..., status: _Optional[str] = ..., loaded_at_ms: _Optional[int] = ..., memory_usage_mb: _Optional[int] = ..., voice_count: _Optional[int] = ..., loaded_voices: _Optional[_Iterable[str]] = ..., supports_speed: bool = ..., supports_pitch: bool = ..., supports_volume: bool = ...) -> None: ...

class ReloadModelRequest(_message.Message):
    __slots__ = ("request_id", "model_instance", "reload_voices")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_INSTANCE_FIELD_NUMBER: _ClassVar[int]
    RELOAD_VOICES_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    model_instance: str
    reload_voices: bool
    def __init__(self, request_id: _Optional[str] = ..., model_instance: _Optional[str] = ..., reload_voices: bool = ...) -> None: ...

class ReloadModelResponse(_message.Message):
    __slots__ = ("request_id", "success", "message", "reload_time_ms", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RELOAD_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    success: bool
    message: str
    reload_time_ms: int
    error: ErrorDetail
    def __init__(self, request_id: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ..., reload_time_ms: _Optional[int] = ..., error: _Optional[_Union[ErrorDetail, _Mapping]] = ...) -> None: ...

class ReloadAllModelsRequest(_message.Message):
    __slots__ = ("request_id", "reload_voices")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    RELOAD_VOICES_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    reload_voices: bool
    def __init__(self, request_id: _Optional[str] = ..., reload_voices: bool = ...) -> None: ...

class ReloadAllModelsResponse(_message.Message):
    __slots__ = ("request_id", "overall_success", "message", "results", "total_time_ms")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    OVERALL_SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    overall_success: bool
    message: str
    results: _containers.RepeatedCompositeFieldContainer[ModelReloadResult]
    total_time_ms: int
    def __init__(self, request_id: _Optional[str] = ..., overall_success: bool = ..., message: _Optional[str] = ..., results: _Optional[_Iterable[_Union[ModelReloadResult, _Mapping]]] = ..., total_time_ms: _Optional[int] = ...) -> None: ...

class ModelReloadResult(_message.Message):
    __slots__ = ("model_instance", "success", "message", "reload_time_ms")
    MODEL_INSTANCE_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RELOAD_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    model_instance: str
    success: bool
    message: str
    reload_time_ms: int
    def __init__(self, model_instance: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ..., reload_time_ms: _Optional[int] = ...) -> None: ...

class AddVoiceRequest(_message.Message):
    __slots__ = ("request_id", "voice_id", "model_instance", "prompt_audio_path", "prompt_text", "metadata")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_INSTANCE_FIELD_NUMBER: _ClassVar[int]
    PROMPT_AUDIO_PATH_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TEXT_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    voice_id: str
    model_instance: str
    prompt_audio_path: str
    prompt_text: str
    metadata: VoiceMetadata
    def __init__(self, request_id: _Optional[str] = ..., voice_id: _Optional[str] = ..., model_instance: _Optional[str] = ..., prompt_audio_path: _Optional[str] = ..., prompt_text: _Optional[str] = ..., metadata: _Optional[_Union[VoiceMetadata, _Mapping]] = ...) -> None: ...

class VoiceMetadata(_message.Message):
    __slots__ = ("name", "language", "gender", "description", "extra")
    class ExtraEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    EXTRA_FIELD_NUMBER: _ClassVar[int]
    name: str
    language: str
    gender: str
    description: str
    extra: _containers.ScalarMap[str, str]
    def __init__(self, name: _Optional[str] = ..., language: _Optional[str] = ..., gender: _Optional[str] = ..., description: _Optional[str] = ..., extra: _Optional[_Mapping[str, str]] = ...) -> None: ...

class AddVoiceResponse(_message.Message):
    __slots__ = ("request_id", "success", "message", "voice_id", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    success: bool
    message: str
    voice_id: str
    error: ErrorDetail
    def __init__(self, request_id: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ..., voice_id: _Optional[str] = ..., error: _Optional[_Union[ErrorDetail, _Mapping]] = ...) -> None: ...

class UpdateVoiceRequest(_message.Message):
    __slots__ = ("request_id", "voice_id", "metadata", "prompt_audio_path", "prompt_text")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    PROMPT_AUDIO_PATH_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TEXT_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    voice_id: str
    metadata: VoiceMetadata
    prompt_audio_path: str
    prompt_text: str
    def __init__(self, request_id: _Optional[str] = ..., voice_id: _Optional[str] = ..., metadata: _Optional[_Union[VoiceMetadata, _Mapping]] = ..., prompt_audio_path: _Optional[str] = ..., prompt_text: _Optional[str] = ...) -> None: ...

class UpdateVoiceResponse(_message.Message):
    __slots__ = ("request_id", "success", "message", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    success: bool
    message: str
    error: ErrorDetail
    def __init__(self, request_id: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ..., error: _Optional[_Union[ErrorDetail, _Mapping]] = ...) -> None: ...

class RemoveVoiceRequest(_message.Message):
    __slots__ = ("request_id", "voice_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    voice_id: str
    def __init__(self, request_id: _Optional[str] = ..., voice_id: _Optional[str] = ...) -> None: ...

class RemoveVoiceResponse(_message.Message):
    __slots__ = ("request_id", "success", "message", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    success: bool
    message: str
    error: ErrorDetail
    def __init__(self, request_id: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ..., error: _Optional[_Union[ErrorDetail, _Mapping]] = ...) -> None: ...

class SetVoiceEnabledRequest(_message.Message):
    __slots__ = ("request_id", "voice_id", "enabled")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    voice_id: str
    enabled: bool
    def __init__(self, request_id: _Optional[str] = ..., voice_id: _Optional[str] = ..., enabled: bool = ...) -> None: ...

class SetVoiceEnabledResponse(_message.Message):
    __slots__ = ("request_id", "success", "message", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    success: bool
    message: str
    error: ErrorDetail
    def __init__(self, request_id: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ..., error: _Optional[_Union[ErrorDetail, _Mapping]] = ...) -> None: ...

class ReloadVoiceRequest(_message.Message):
    __slots__ = ("request_id", "voice_id")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    voice_id: str
    def __init__(self, request_id: _Optional[str] = ..., voice_id: _Optional[str] = ...) -> None: ...

class ReloadVoiceResponse(_message.Message):
    __slots__ = ("request_id", "success", "message", "reload_time_ms", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RELOAD_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    success: bool
    message: str
    reload_time_ms: int
    error: ErrorDetail
    def __init__(self, request_id: _Optional[str] = ..., success: bool = ..., message: _Optional[str] = ..., reload_time_ms: _Optional[int] = ..., error: _Optional[_Union[ErrorDetail, _Mapping]] = ...) -> None: ...
