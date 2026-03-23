# -*- coding: utf-8 -*-
"""GPU 상태 관리 - gRPC API와 Prometheus 메트릭 공통 데이터 소스.

synthesis_stats.py 패턴을 따라 GPU 상태 정보를 중앙에서 관리합니다.

Usage:
    from tts_engine.services.gpu_stats import GPUStats, gpu_stats

    # 싱글톤 인스턴스 사용
    status = gpu_stats.get_all_gpu_status()

    # 특정 디바이스
    status = gpu_stats.get_gpu_status(device_id=0)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from tts_engine.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUStatusData:
    """GPU 상태 데이터 (공통 데이터 구조)."""

    available: bool
    device: str  # e.g., "cuda:0"
    device_id: int
    device_name: str
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    memory_percent: int  # 메모리 사용률 (%)
    utilization_percent: int  # GPU 연산 사용률 (%)
    temperature_c: Optional[int]  # 온도 (섭씨)
    driver_version: Optional[str]
    cuda_version: Optional[str]
    error: Optional[str] = None


class GPUStats:
    """GPU 상태 관리 - gRPC와 Prometheus 공통 데이터 소스."""

    _instance: Optional["GPUStats"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # pynvml 초기화 상태
        self._nvml_initialized = False
        self._init_nvml()

    def _init_nvml(self) -> None:
        """pynvml 초기화 (utilization, temperature 조회용)."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_initialized = True
            logger.debug("pynvml initialized successfully")
        except ImportError:
            logger.debug("pynvml not available, GPU utilization/temperature unavailable")
        except Exception as e:
            logger.warning(f"Failed to initialize pynvml: {e}")

    def _get_nvml_info(self, device_id: int) -> Dict:
        """pynvml로 추가 GPU 정보 조회."""
        result = {
            "utilization_percent": 0,
            "temperature_c": None,
            "driver_version": None,
        }

        if not self._nvml_initialized:
            return result

        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # GPU 사용률
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                result["utilization_percent"] = util.gpu
            except Exception:
                pass

            # 온도
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                result["temperature_c"] = temp
            except Exception:
                pass

            # 드라이버 버전 (한 번만 조회)
            try:
                result["driver_version"] = pynvml.nvmlSystemGetDriverVersion()
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Failed to get nvml info for device {device_id}: {e}")

        return result

    def _get_cuda_version(self) -> Optional[str]:
        """CUDA 버전 조회."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.version.cuda
        except Exception:
            pass
        return None

    def get_gpu_status(self, device_id: int) -> GPUStatusData:
        """특정 GPU의 상태 조회."""
        try:
            import torch

            if not torch.cuda.is_available():
                return GPUStatusData(
                    available=False,
                    device="none",
                    device_id=device_id,
                    device_name="N/A",
                    total_memory_mb=0,
                    used_memory_mb=0,
                    free_memory_mb=0,
                    memory_percent=0,
                    utilization_percent=0,
                    temperature_c=None,
                    driver_version=None,
                    cuda_version=None,
                    error="CUDA not available",
                )

            if device_id >= torch.cuda.device_count():
                return GPUStatusData(
                    available=False,
                    device=f"cuda:{device_id}",
                    device_id=device_id,
                    device_name="N/A",
                    total_memory_mb=0,
                    used_memory_mb=0,
                    free_memory_mb=0,
                    memory_percent=0,
                    utilization_percent=0,
                    temperature_c=None,
                    driver_version=None,
                    cuda_version=None,
                    error=f"Device {device_id} not found",
                )

            # torch.cuda로 기본 정보 조회
            props = torch.cuda.get_device_properties(device_id)
            mem_info = torch.cuda.mem_get_info(device_id)

            free_memory_mb = mem_info[0] // (1024 * 1024)
            total_memory_mb = mem_info[1] // (1024 * 1024)
            used_memory_mb = total_memory_mb - free_memory_mb
            memory_percent = int((used_memory_mb / total_memory_mb) * 100) if total_memory_mb > 0 else 0

            # pynvml로 추가 정보 조회
            nvml_info = self._get_nvml_info(device_id)

            return GPUStatusData(
                available=True,
                device=f"cuda:{device_id}",
                device_id=device_id,
                device_name=props.name,
                total_memory_mb=total_memory_mb,
                used_memory_mb=used_memory_mb,
                free_memory_mb=free_memory_mb,
                memory_percent=memory_percent,
                utilization_percent=nvml_info["utilization_percent"],
                temperature_c=nvml_info["temperature_c"],
                driver_version=nvml_info["driver_version"],
                cuda_version=self._get_cuda_version(),
            )

        except Exception as e:
            logger.warning(f"Failed to get GPU status for device {device_id}: {e}")
            return GPUStatusData(
                available=False,
                device=f"cuda:{device_id}",
                device_id=device_id,
                device_name="N/A",
                total_memory_mb=0,
                used_memory_mb=0,
                free_memory_mb=0,
                memory_percent=0,
                utilization_percent=0,
                temperature_c=None,
                driver_version=None,
                cuda_version=None,
                error=str(e),
            )

    def get_all_gpu_status(self) -> List[GPUStatusData]:
        """모든 GPU의 상태 조회."""
        try:
            import torch

            if not torch.cuda.is_available():
                return [
                    GPUStatusData(
                        available=False,
                        device="none",
                        device_id=0,
                        device_name="N/A",
                        total_memory_mb=0,
                        used_memory_mb=0,
                        free_memory_mb=0,
                        memory_percent=0,
                        utilization_percent=0,
                        temperature_c=None,
                        driver_version=None,
                        cuda_version=None,
                        error="CUDA not available",
                    )
                ]

            device_count = torch.cuda.device_count()
            return [self.get_gpu_status(i) for i in range(device_count)]

        except Exception as e:
            logger.warning(f"Failed to get GPU status: {e}")
            return [
                GPUStatusData(
                    available=False,
                    device="none",
                    device_id=0,
                    device_name="N/A",
                    total_memory_mb=0,
                    used_memory_mb=0,
                    free_memory_mb=0,
                    memory_percent=0,
                    utilization_percent=0,
                    temperature_c=None,
                    driver_version=None,
                    cuda_version=None,
                    error=str(e),
                )
            ]

    def get_gpu_count(self) -> int:
        """GPU 개수 반환."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            pass
        return 0

    def update_prometheus_metrics(self) -> None:
        """Prometheus 메트릭 업데이트."""
        try:
            from tts_engine.server.metrics_server import metrics_registry

            if not metrics_registry:
                return

            for status in self.get_all_gpu_status():
                if not status.available:
                    continue

                device_label = status.device

                # 메모리 메트릭
                metrics_registry.gpu_memory_used.labels(device=device_label).set(
                    status.used_memory_mb
                )
                metrics_registry.gpu_memory_total.labels(device=device_label).set(
                    status.total_memory_mb
                )
                metrics_registry.gpu_memory_percent.labels(device=device_label).set(
                    status.memory_percent
                )

                # 사용률
                metrics_registry.gpu_utilization.labels(device=device_label).set(
                    status.utilization_percent
                )

                # 온도
                if status.temperature_c is not None:
                    metrics_registry.gpu_temperature.labels(device=device_label).set(
                        status.temperature_c
                    )

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to update Prometheus GPU metrics: {e}")


# 싱글톤 인스턴스
gpu_stats = GPUStats()