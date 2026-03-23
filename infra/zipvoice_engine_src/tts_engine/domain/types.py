"""Domain types for TTS Engine.

공통으로 사용되는 타입 정의들.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class AudioData:
    """오디오 데이터 컨테이너."""

    data: bytes
    sample_rate: int
    channels: int = 1
    bit_depth: int = 16
    duration_seconds: float = 0.0

    @property
    def size_bytes(self) -> int:
        """오디오 데이터 크기 (bytes)."""
        return len(self.data)

    def to_tensor(self) -> torch.Tensor:
        """bytes를 torch.Tensor로 변환."""
        import numpy as np

        # PCM 16-bit를 float32로 변환
        audio_np = np.frombuffer(self.data, dtype=np.int16).astype(np.float32) / 32768.0
        return torch.from_numpy(audio_np)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        sample_rate: int,
    ) -> "AudioData":
        """torch.Tensor를 AudioData로 변환."""
        import numpy as np

        # float32를 PCM 16-bit로 변환
        audio_np = (tensor.cpu().numpy() * 32768.0).astype(np.int16)
        data = audio_np.tobytes()
        duration = len(audio_np) / sample_rate

        return cls(
            data=data,
            sample_rate=sample_rate,
            channels=1,
            bit_depth=16,
            duration_seconds=duration,
        )


@dataclass
class PromptData:
    """프롬프트 데이터 (Zero-shot TTS용)."""

    wav_path: Optional[str] = None
    wav_tensor: Optional[torch.Tensor] = None
    text: str = ""
    tokens: Optional[list[int]] = None
    features: Optional[torch.Tensor] = None
    rms: Optional[float] = None

    @property
    def is_loaded(self) -> bool:
        """프롬프트가 로드되었는지 확인."""
        return self.features is not None


@dataclass
class ModelInfo:
    """모델 정보."""

    name: str
    model_type: str
    runtime: str
    device: str
    loaded: bool = False
    memory_mb: float = 0.0
    inference_count: int = 0
    last_inference_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUInfo:
    """GPU 정보."""

    device_id: int
    name: str
    total_memory_mb: float
    used_memory_mb: float
    free_memory_mb: float
    temperature_c: Optional[float] = None
    utilization_percent: Optional[float] = None

    @property
    def memory_usage_percent(self) -> float:
        """메모리 사용률 (%)."""
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100


@dataclass
class ServerStats:
    """서버 통계."""

    uptime_seconds: float
    total_requests: int
    active_requests: int
    completed_requests: int
    failed_requests: int
    average_latency_ms: float
    requests_per_second: float
    gpu_info: Optional[GPUInfo] = None
    model_pool_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """헬스체크 결과."""

    healthy: bool
    status: str  # healthy, degraded, unhealthy
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
