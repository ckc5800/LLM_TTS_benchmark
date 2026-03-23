"""Constants and enumerations for TTS Engine."""

from enum import Enum


class RuntimeType(str, Enum):
    """TTS 모델 런타임 타입.

    - PYTORCH: 개발/디버깅용, 유연성 최고 (~3.2s)
    - ONNX: 범용 배포, CPU/GPU 지원 (GPU: ~2.3s)
    - TENSORRT: 프로덕션 권장, 최고 성능 (~1.3s, 2.5x speedup)
    """

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class ExecutionMode(str, Enum):
    """실행 모드."""

    PROCESS = "process"
    THREAD = "thread"
    AUTO = "auto"


class AudioFormat(str, Enum):
    """오디오 출력 포맷."""

    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    PCM = "pcm"
    FLAC = "flac"


class SynthesisStatus(str, Enum):
    """합성 상태."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class HealthStatus(str, Enum):
    """헬스 상태."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class OnnxProvider(str, Enum):
    """ONNX Runtime Execution Provider."""

    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"


class Defaults:
    """기본값 상수.

    ZipVoice 모델 기준 (model.json, infer_zipvoice.py 참조)
    """

    # Audio - ZipVoice uses 24kHz (model.json: feature.sampling_rate)
    SAMPLE_RATE: int = 24000
    AUDIO_FORMAT: str = "wav"

    # Synthesis parameters
    SPEED: float = 1.0
    PITCH: float = 1.0
    VOLUME: float = 1.0

    # ZipVoice specific (infer_zipvoice.py, pytriton_server.py 참조)
    # ZipVoice 기본 모델
    NUM_STEPS: int = 16  # zipvoice: 16, zipvoice_distill: 4 or 8
    T_SHIFT: float = 0.5
    GUIDANCE_SCALE: float = 1.0  # zipvoice: 1.0, zipvoice_distill: 3.0
    FEAT_DIM: int = 100  # model.json: model.feat_dim

    # ZipVoice-Distill 전용
    DISTILL_NUM_STEPS: int = 4
    DISTILL_GUIDANCE_SCALE: float = 3.0

    # Server
    GRPC_HOST: str = "0.0.0.0"
    GRPC_PORT: int = 8220
    GRPC_MAX_WORKERS: int = 10

    # Pool
    MODEL_POOL_SIZE: int = 4
    BATCH_SIZE: int = 4


class Limits:
    """제한값 상수."""

    # Text
    MAX_TEXT_LENGTH: int = 2000
    MIN_TEXT_LENGTH: int = 1

    # Batch
    MAX_BATCH_SIZE: int = 16
    MIN_BATCH_SIZE: int = 1

    # Concurrency
    MAX_CONCURRENT_REQUESTS: int = 100
    MAX_CONCURRENT_RPCS: int = 100

    # Timeout (seconds)
    TASK_TIMEOUT_SECONDS: int = 120
    GRPC_TIMEOUT_SECONDS: int = 30
    BATCH_TIMEOUT_MS: int = 100

    # Health check
    HEALTH_CHECK_INTERVAL_SECONDS: int = 30
    UNHEALTHY_THRESHOLD: int = 3

    # Model instances
    MAX_MODEL_INSTANCES: int = 50
    MIN_MODEL_INSTANCES: int = 1

    # File
    MAX_AUDIO_DURATION_SECONDS: int = 300  # 5 minutes
    FILE_RETENTION_HOURS: int = 24


class Thresholds:
    """임계값 상수."""

    # GPU Memory
    GPU_MEMORY_WARNING_PERCENT: int = 70
    GPU_MEMORY_CRITICAL_PERCENT: int = 85
    GPU_MEMORY_MAX_PERCENT: int = 90

    # Active requests
    ACTIVE_REQUESTS_WARNING: int = 50
    ACTIVE_REQUESTS_CRITICAL: int = 80

    # Latency (ms)
    LATENCY_WARNING_MS: int = 1000
    LATENCY_CRITICAL_MS: int = 3000

    # RTF (Real-Time Factor) - lower is better
    RTF_WARNING: float = 0.5
    RTF_CRITICAL: float = 1.0


class ModelPaths:
    """모델 경로 상수."""

    # Base paths
    DEFAULT_MODEL_BASE: str = "./models"

    # ZipVoice versions
    ZIPVOICE_PYTORCH: str = "zipvoice/v0.0.4"
    ZIPVOICE_ONNX: str = "zipvoice/v0.0.4-onnx"
    ZIPVOICE_TENSORRT: str = "zipvoice/v0.0.4-tensorrt-official"

    # Model files
    PYTORCH_CHECKPOINT: str = "model.pt"
    MODEL_CONFIG: str = "model.json"
    TOKENS_FILE: str = "tokens.txt"

    # ONNX files
    ONNX_TEXT_ENCODER: str = "text_encoder.onnx"
    ONNX_FM_DECODER: str = "fm_decoder.onnx"

    # TensorRT files (벤치마크 결과 기준)
    TRT_ENGINE_BATCH4: str = "fm_decoder.fp16.max_batch_4.plan"
    TRT_ENGINE_BATCH8: str = "fm_decoder.fp16.max_batch_8.plan"
