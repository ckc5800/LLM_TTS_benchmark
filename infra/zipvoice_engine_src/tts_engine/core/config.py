"""Configuration management using Pydantic.

YAML 설정 파일 로드 및 환경 변수 오버라이드를 지원합니다.
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tts_engine.core.constants import (
    AudioFormat,
    Defaults,
    ExecutionMode,
    Limits,
    OnnxProvider,
    RuntimeType,
)
from tts_engine.core.exceptions import ConfigFileNotFoundError, ConfigValidationError


# ============================================================
# 설정 모델 정의
# ============================================================


class TlsConfig(BaseModel):
    """TLS/SSL 설정."""

    enabled: bool = Field(default=False, description="TLS 활성화")
    cert_file: Optional[str] = Field(default=None, description="서버 인증서 파일 경로 (PEM)")
    key_file: Optional[str] = Field(default=None, description="서버 개인키 파일 경로 (PEM)")
    ca_file: Optional[str] = Field(default=None, description="CA 인증서 파일 경로 (mTLS용, 선택)")
    require_client_cert: bool = Field(default=False, description="클라이언트 인증서 요구 (mTLS)")


class GrpcConfig(BaseModel):
    """gRPC 서버 설정."""

    host: str = Field(default=Defaults.GRPC_HOST, description="바인딩 호스트")
    port: int = Field(default=Defaults.GRPC_PORT, ge=1024, le=65535, description="포트")
    max_workers: int = Field(default=Defaults.GRPC_MAX_WORKERS, ge=1, le=100)
    max_concurrent_rpcs: int = Field(default=Limits.MAX_CONCURRENT_RPCS, ge=1)
    keepalive_time_ms: int = Field(default=30000, ge=1000)
    keepalive_timeout_ms: int = Field(default=10000, ge=1000)
    tls: TlsConfig = Field(default_factory=TlsConfig)


class QueueConfig(BaseModel):
    """요청 큐 설정."""

    max_queue_size: int = Field(default=500, ge=1, le=10000, description="최대 대기열 크기")
    queue_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0, description="대기 타임아웃 (초)")


class MetricsConfig(BaseModel):
    """Prometheus 메트릭 서버 설정."""

    enabled: bool = Field(default=True, description="메트릭 서버 활성화")
    host: str = Field(default="0.0.0.0", description="메트릭 서버 호스트")
    port: int = Field(default=9090, ge=1024, le=65535, description="메트릭 서버 포트")


class ServerConfig(BaseModel):
    """서버 설정."""

    grpc: GrpcConfig = Field(default_factory=GrpcConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


class LoggingConfig(BaseModel):
    """로깅 설정."""

    level: str = Field(default="INFO", description="로그 레벨")
    log_dir: str = Field(default="./logs", description="로그 디렉토리")
    app_name: str = Field(default="tts_engine", description="앱 이름")
    retention_days: int = Field(default=30, ge=1, le=365, description="보관 일수")
    rotation_time: str = Field(default="00:00", description="로테이션 시간")
    console_enabled: bool = Field(default=True)
    file_enabled: bool = Field(default=True)
    json_enabled: bool = Field(default=True)
    show_source_location: bool = Field(default=False, description="콘솔에 소스 경로 표시")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class FileStorageConfig(BaseModel):
    """파일 저장소 설정."""

    enabled: bool = Field(default=False, description="파일 저장 활성화")
    path: str = Field(default="./audio_output", description="저장 기본 경로")
    use_date_subdirs: bool = Field(default=True, description="날짜별 하위 디렉토리 생성 (YYYYMMDD)")
    cleanup_enabled: bool = Field(default=True, description="자동 정리 활성화")
    retention_hours: int = Field(default=Limits.FILE_RETENTION_HOURS, ge=1, description="파일 보관 시간")


class AudioProcessingConfig(BaseModel):
    """오디오 처리 설정."""

    # 프롬프트 오디오 전처리
    prompt_trim_silence: bool = Field(default=True, description="프롬프트 앞뒤 침묵 제거")
    prompt_trim_top_db: int = Field(default=20, ge=10, le=60, description="침묵 감지 임계값 (dB)")
    prompt_normalize: bool = Field(default=True, description="프롬프트 RMS 정규화")
    prompt_target_rms: float = Field(default=0.1, ge=0.01, le=1.0, description="목표 RMS 값")

    # 청크 연결
    crossfade_duration: float = Field(default=0.1, ge=0.0, le=1.0, description="교차페이드 길이 (초)")

    # 출력 정규화
    output_normalize: bool = Field(default=False, description="출력 오디오 정규화")
    output_target_rms: float = Field(default=0.1, ge=0.01, le=1.0, description="출력 목표 RMS")


class TextPreprocessConfig(BaseModel):
    """텍스트 전처리 설정."""

    # 한국어 정규화
    normalize_korean: bool = Field(default=True, description="한국어 텍스트 정규화")
    convert_numbers: bool = Field(default=True, description="숫자를 한국어로 변환")
    convert_units: bool = Field(default=True, description="단위를 한국어로 변환")
    convert_emails: bool = Field(default=True, description="이메일을 발음으로 변환")
    convert_phones: bool = Field(default=True, description="전화번호를 발음으로 변환")

    # 괄호 처리
    process_parentheses: bool = Field(default=True, description="괄호 내용 처리")
    parentheses_max_length: int = Field(default=0, ge=0, le=100, description="괄호 내용 읽기 최대 길이")

    # 이모지
    remove_emojis: bool = Field(default=True, description="이모지 제거")


class SynthesisConfig(BaseModel):
    """합성 설정."""

    max_text_length: int = Field(default=Limits.MAX_TEXT_LENGTH, ge=1, le=10000)
    default_sample_rate: int = Field(default=Defaults.SAMPLE_RATE)
    default_format: AudioFormat = Field(default=AudioFormat.WAV)
    file_storage: FileStorageConfig = Field(default_factory=FileStorageConfig)
    audio_processing: AudioProcessingConfig = Field(default_factory=AudioProcessingConfig)
    text_preprocess: TextPreprocessConfig = Field(default_factory=TextPreprocessConfig)

    @field_validator("default_sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        valid_rates = [16000, 22050, 24000, 44100, 48000]
        if v not in valid_rates:
            raise ValueError(f"Invalid sample rate: {v}. Must be one of {valid_rates}")
        return v


class ExecutionConfig(BaseModel):
    """실행 설정."""

    mode: ExecutionMode = Field(default=ExecutionMode.THREAD)
    process_pool_size: int = Field(default=4, ge=1, le=32)
    thread_pool_size: int = Field(default=8, ge=1, le=128)
    task_timeout_seconds: int = Field(default=Limits.TASK_TIMEOUT_SECONDS, ge=10, le=600)


class AutoBatchConfig(BaseModel):
    """자동 배치 설정."""

    enabled: bool = Field(default=True)
    max_batch_size: int = Field(default=Defaults.BATCH_SIZE, ge=1, le=Limits.MAX_BATCH_SIZE)
    min_batch_size: int = Field(default=1, ge=1)
    timeout_ms: int = Field(default=Limits.BATCH_TIMEOUT_MS, ge=10, le=1000)
    group_by_voice: bool = Field(default=True, description="음성별 배치 그룹화")


class SpeakerCacheConfig(BaseModel):
    """화자 캐시 설정."""

    enabled: bool = Field(default=True, description="캐시 활성화")
    max_size: int = Field(default=100, ge=10, le=1000, description="최대 캐시 크기")
    ttl_seconds: float = Field(default=3600.0, ge=60.0, le=86400.0, description="TTL (초)")
    preload_on_startup: bool = Field(default=True, description="시작 시 음성 사전 로드")


class WarmupConfig(BaseModel):
    """워밍업 설정."""

    enabled: bool = Field(default=True)
    on_startup: bool = Field(default=True)
    test_text: str = Field(default="안녕하세요. 테스트입니다.")
    repeat_count: int = Field(default=3, ge=1, le=10)


class PerformanceConfig(BaseModel):
    """성능 설정."""

    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    auto_batch: AutoBatchConfig = Field(default_factory=AutoBatchConfig)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    speaker_cache: SpeakerCacheConfig = Field(default_factory=SpeakerCacheConfig)


class ModelTypeConfig(BaseModel):
    """모델 타입 정의."""

    enabled: bool = Field(default=True)
    module: str = Field(description="모델 모듈 경로")
    class_name: str = Field(alias="class", description="모델 클래스명")
    config_class: Optional[str] = Field(default=None)


class RuntimeOptions(BaseModel):
    """런타임별 옵션."""

    runtime: RuntimeType = Field(default=RuntimeType.PYTORCH)

    # 모델 이름 (zipvoice / zipvoice_distill)
    model_name: str = Field(default="zipvoice", description="모델 이름 (zipvoice, zipvoice_distill)")

    # PyTorch 옵션
    fp16: bool = Field(default=True)
    vocoder_path: Optional[str] = Field(default="./models/vocos/mel-24khz", description="보코더 경로")

    # ONNX 옵션
    provider: OnnxProvider = Field(default=OnnxProvider.CUDA)
    onnx_model_dir: Optional[str] = Field(default=None)
    max_concurrent_gpu: Optional[int] = Field(default=None, ge=1, le=16, description="GPU 동시 실행 제한 (ONNX)")

    # TensorRT 옵션
    engine_file: Optional[str] = Field(default=None)
    tensorrt_model_dir: Optional[str] = Field(default=None)
    trt_concurrent: int = Field(default=2, ge=1, le=32, description="TensorRT 동시 CUDA 스트림 수")

    # ZipVoice 공통 옵션
    num_steps: int = Field(default=Defaults.NUM_STEPS, ge=1, le=32)
    t_shift: float = Field(default=Defaults.T_SHIFT, ge=0.0, le=1.0)
    guidance_scale: float = Field(default=Defaults.GUIDANCE_SCALE, ge=0.0, le=10.0)

    # 스트리밍
    enable_streaming: bool = Field(default=True)
    chunk_size: int = Field(default=4096, ge=1024)


class ModelInstanceConfig(BaseModel):
    """모델 인스턴스 정의."""

    model_type: str = Field(description="model_types 참조 키")
    device: str = Field(default="cuda:0", description="디바이스 (auto, cpu, cuda:N)")
    pool_size: int = Field(default=Defaults.MODEL_POOL_SIZE, ge=1, le=50)
    model_path: Optional[str] = Field(default=None)
    options: RuntimeOptions = Field(default_factory=RuntimeOptions)
    enabled: bool = Field(default=True, description="인스턴스 활성화 여부")

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        import re

        if v not in ["auto", "cpu"] and not re.match(r"^cuda:\d+$", v):
            raise ValueError(f"Invalid device: {v}. Must be 'auto', 'cpu', or 'cuda:N'")
        return v


class HealthConfig(BaseModel):
    """헬스체크 설정."""

    check_interval_seconds: int = Field(default=Limits.HEALTH_CHECK_INTERVAL_SECONDS, ge=5)
    unhealthy_threshold: int = Field(default=Limits.UNHEALTHY_THRESHOLD, ge=1)
    gpu_memory_threshold_percent: int = Field(default=90, ge=50, le=100)
    active_requests_threshold: int = Field(default=Limits.MAX_CONCURRENT_REQUESTS, ge=10)


# ============================================================
# 메인 설정 클래스
# ============================================================


class Settings(BaseSettings):
    """TTS 엔진 전체 설정.

    환경 변수 오버라이드:
    - TTS_ENGINE__SERVER__GRPC__PORT=8221
    - TTS_ENGINE__LOGGING__LEVEL=DEBUG
    """

    model_config = SettingsConfigDict(
        env_prefix="TTS_ENGINE__",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    model_types: dict[str, ModelTypeConfig] = Field(default_factory=dict)
    model_instances: dict[str, ModelInstanceConfig] = Field(default_factory=dict)
    health: HealthConfig = Field(default_factory=HealthConfig)


class VoiceConfig(BaseModel):
    """음성 설정."""

    name: str = Field(description="음성 이름")
    language: str = Field(default="ko-KR")
    gender: str = Field(default="female")
    model_instance: str = Field(description="model_instances 참조 키")
    description: str = Field(default="")
    sample_rate: int = Field(default=Defaults.SAMPLE_RATE)
    enabled: bool = Field(default=True)
    prompt_wav: Optional[str] = Field(default=None, description="프롬프트 WAV 파일 경로")
    prompt_text: Optional[str] = Field(default=None, description="프롬프트 텍스트")
    options: dict[str, Any] = Field(default_factory=dict)


class VoicesConfig(BaseModel):
    """음성 목록 설정."""

    voices: dict[str, VoiceConfig] = Field(default_factory=dict)
    default_voice_id: str = Field(default="")
    fallback_voice_id: Optional[str] = Field(default=None)


# ============================================================
# 설정 로더 함수
# ============================================================


def load_yaml(file_path: Path) -> dict[str, Any]:
    """YAML 파일을 로드합니다."""
    if not file_path.exists():
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def deep_merge(base: dict, override: dict) -> dict:
    """딕셔너리를 깊은 병합합니다.

    Args:
        base: 기본 딕셔너리
        override: 오버라이드할 딕셔너리

    Returns:
        병합된 딕셔너리
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_base_configs(config_dir: Path) -> dict[str, Any]:
    """base/ 디렉토리의 역할별 설정 파일들을 로드하여 병합합니다.

    Args:
        config_dir: config 디렉토리 경로

    Returns:
        병합된 설정 딕셔너리
    """
    base_dir = config_dir / "base"
    merged = {}

    # 역할별 설정 파일 목록
    base_files = [
        "server.yaml",
        "logging.yaml",
        "synthesis.yaml",
        "performance.yaml",
        "models.yaml",
        "health.yaml",
    ]

    for filename in base_files:
        file_path = base_dir / filename
        if file_path.exists():
            data = load_yaml(file_path)
            merged = deep_merge(merged, data)

    return merged


def load_env_config(config_dir: Path, env: str) -> dict[str, Any]:
    """환경별 설정 파일을 로드합니다.

    Args:
        config_dir: config 디렉토리 경로
        env: 환경 이름 (development, staging, production)

    Returns:
        환경별 설정 딕셔너리
    """
    env_file = config_dir / "env" / f"{env}.yaml"
    return load_yaml(env_file)


def load_local_config(config_dir: Path) -> dict[str, Any]:
    """로컬 오버라이드 설정을 로드합니다.

    Args:
        config_dir: config 디렉토리 경로

    Returns:
        로컬 설정 딕셔너리
    """
    local_file = config_dir / "local.yaml"
    return load_yaml(local_file)


def get_environment() -> str:
    """현재 환경을 반환합니다.

    환경 변수 TTS_ENGINE_ENV로 설정 가능.
    기본값: development
    """
    import os
    return os.environ.get("TTS_ENGINE_ENV", "development")


def load_settings(
    config_dir: Optional[str] = None,
    env: Optional[str] = None,
    voices_path: Optional[str] = None,
) -> tuple[Settings, VoicesConfig]:
    """설정 파일들을 로드합니다.

    로드 순서:
    1. base/*.yaml (역할별 기본 설정)
    2. env/{환경}.yaml (환경별 오버라이드)
    3. local.yaml (로컬 오버라이드, 선택적)
    4. 환경 변수 (최종 오버라이드)

    Args:
        config_dir: config 디렉토리 경로 (기본: ./config)
        env: 환경 이름 (기본: TTS_ENGINE_ENV 또는 development)
        voices_path: voices.yaml 경로 (기본: config/voices.yaml)

    Returns:
        (Settings, VoicesConfig) 튜플

    Raises:
        ConfigValidationError: 유효성 검증 실패 시
    """
    # 기본 경로 설정
    config_path = Path(config_dir) if config_dir else Path("config")
    environment = env or get_environment()

    # 1. base/*.yaml 로드 (역할별)
    merged_config = load_base_configs(config_path)

    # 2. env/{환경}.yaml 로드
    env_config = load_env_config(config_path, environment)
    merged_config = deep_merge(merged_config, env_config)

    # 3. local.yaml 로드 (선택적)
    local_config = load_local_config(config_path)
    merged_config = deep_merge(merged_config, local_config)

    # Settings 생성
    try:
        settings = Settings(**merged_config)
    except Exception as e:
        raise ConfigValidationError(f"설정 파싱 오류: {e}")

    # Voices 로드
    voices_file = Path(voices_path) if voices_path else config_path / "voices.yaml"
    try:
        voices_data = load_yaml(voices_file)
        voices = VoicesConfig(**voices_data) if voices_data else VoicesConfig()
    except Exception as e:
        raise ConfigValidationError(f"voices.yaml 파싱 오류: {e}")

    return settings, voices


def load_settings_from_env(env: str = "development") -> tuple[Settings, VoicesConfig]:
    """특정 환경의 설정을 로드합니다.

    Args:
        env: 환경 이름 (development, staging, production)

    Returns:
        (Settings, VoicesConfig) 튜플
    """
    return load_settings(env=env)


def get_default_settings() -> Settings:
    """기본 설정을 반환합니다."""
    return Settings(
        model_types={
            "zipvoice": ModelTypeConfig(
                enabled=True,
                module="tts_engine.models.zipvoice.model",
                class_name="ZipVoiceModel",
                config_class="tts_engine.models.zipvoice.config.ZipVoiceConfig",
            ),
            "dummy": ModelTypeConfig(
                enabled=True,
                module="tts_engine.models.dummy.model",
                class_name="DummyModel",
            ),
        },
        model_instances={
            "zipvoice_pytorch": ModelInstanceConfig(
                model_type="zipvoice",
                device="cuda:0",
                pool_size=4,
                options=RuntimeOptions(
                    runtime=RuntimeType.PYTORCH,
                    num_steps=Defaults.NUM_STEPS,
                ),
            ),
            "zipvoice_tensorrt": ModelInstanceConfig(
                model_type="zipvoice",
                device="cuda:0",
                pool_size=4,
                options=RuntimeOptions(
                    runtime=RuntimeType.TENSORRT,
                    num_steps=Defaults.DISTILL_NUM_STEPS,
                    guidance_scale=Defaults.DISTILL_GUIDANCE_SCALE,
                ),
            ),
            "default": ModelInstanceConfig(
                model_type="dummy",
                device="cpu",
                pool_size=1,
            ),
        },
    )
