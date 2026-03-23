# -*- coding: utf-8 -*-
"""Health Check Service - 헬스체크 서비스.

서비스 상태 모니터링 및 헬스체크 기능을 제공합니다.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from tts_engine.core.config import HealthConfig
from tts_engine.core.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """헬스 상태."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """컴포넌트 헬스 정보."""

    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)


@dataclass
class SystemHealth:
    """시스템 전체 헬스 정보."""

    status: HealthStatus
    components: Dict[str, ComponentHealth]
    uptime_seconds: float
    timestamp: float = field(default_factory=time.time)
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환합니다."""
        return {
            "status": self.status.value,
            "uptime_seconds": self.uptime_seconds,
            "timestamp": self.timestamp,
            "version": self.version,
            "components": {
                name: {
                    "status": comp.status.value,
                    "message": comp.message,
                    "details": comp.details,
                    "last_check": comp.last_check,
                }
                for name, comp in self.components.items()
            },
        }


class HealthChecker:
    """헬스 체크 관리자.

    컴포넌트 헬스 체크를 등록하고 실행합니다.

    Example:
        >>> checker = HealthChecker(config)
        >>> checker.register("database", check_database)
        >>> health = await checker.check_all()
    """

    def __init__(self, config: Optional[HealthConfig] = None):
        """헬스 체커 초기화.

        Args:
            config: 헬스체크 설정
        """
        self._config = config or HealthConfig()
        self._checks: Dict[str, Callable[[], ComponentHealth]] = {}
        self._async_checks: Dict[str, Callable[[], "asyncio.Future[ComponentHealth]"]] = {}
        self._start_time = time.time()
        self._last_status = HealthStatus.UNKNOWN
        self._consecutive_failures: Dict[str, int] = {}

        logger.info(
            "HealthChecker initialized",
            check_interval=self._config.check_interval_seconds,
            unhealthy_threshold=self._config.unhealthy_threshold,
        )

    def register(
        self,
        name: str,
        check_func: Callable[[], ComponentHealth],
    ) -> None:
        """동기 헬스 체크를 등록합니다.

        Args:
            name: 컴포넌트 이름
            check_func: 헬스 체크 함수
        """
        self._checks[name] = check_func
        self._consecutive_failures[name] = 0
        logger.debug(f"Registered health check: {name}")

    def register_async(
        self,
        name: str,
        check_func: Callable[[], "asyncio.Future[ComponentHealth]"],
    ) -> None:
        """비동기 헬스 체크를 등록합니다.

        Args:
            name: 컴포넌트 이름
            check_func: 비동기 헬스 체크 함수
        """
        self._async_checks[name] = check_func
        self._consecutive_failures[name] = 0
        logger.debug(f"Registered async health check: {name}")

    async def check_all(self) -> SystemHealth:
        """모든 컴포넌트의 헬스를 체크합니다.

        Returns:
            시스템 전체 헬스 정보
        """
        components: Dict[str, ComponentHealth] = {}
        overall_status = HealthStatus.HEALTHY

        # 동기 체크 실행
        for name, check_func in self._checks.items():
            try:
                health = check_func()
                components[name] = health
                self._update_failure_count(name, health.status)
            except Exception as e:
                logger.error(f"Health check failed: {name}", error=str(e))
                components[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                )
                self._update_failure_count(name, HealthStatus.UNHEALTHY)

        # 비동기 체크 실행
        if self._async_checks:
            async_results = await asyncio.gather(
                *[check() for check in self._async_checks.values()],
                return_exceptions=True,
            )

            for name, result in zip(self._async_checks.keys(), async_results):
                if isinstance(result, Exception):
                    logger.error(f"Async health check failed: {name}", error=str(result))
                    components[name] = ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {str(result)}",
                    )
                    self._update_failure_count(name, HealthStatus.UNHEALTHY)
                else:
                    components[name] = result
                    self._update_failure_count(name, result.status)

        # 전체 상태 결정
        overall_status = self._calculate_overall_status(components)
        self._last_status = overall_status

        uptime = time.time() - self._start_time

        return SystemHealth(
            status=overall_status,
            components=components,
            uptime_seconds=uptime,
        )

    def _update_failure_count(self, name: str, status: HealthStatus) -> None:
        """연속 실패 횟수를 업데이트합니다."""
        if status == HealthStatus.UNHEALTHY:
            self._consecutive_failures[name] = self._consecutive_failures.get(name, 0) + 1
        else:
            self._consecutive_failures[name] = 0

    def _calculate_overall_status(
        self,
        components: Dict[str, ComponentHealth],
    ) -> HealthStatus:
        """전체 상태를 계산합니다."""
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [comp.status for comp in components.values()]

        # 모두 healthy
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        # 하나라도 unhealthy이고 임계값 초과
        for name, comp in components.items():
            if comp.status == HealthStatus.UNHEALTHY:
                if self._consecutive_failures.get(name, 0) >= self._config.unhealthy_threshold:
                    return HealthStatus.UNHEALTHY

        # 일부 degraded 또는 임계값 미달 unhealthy
        return HealthStatus.DEGRADED

    @property
    def uptime_seconds(self) -> float:
        """서비스 가동 시간 (초)."""
        return time.time() - self._start_time

    @property
    def last_status(self) -> HealthStatus:
        """마지막 체크 상태."""
        return self._last_status

    def is_healthy(self) -> bool:
        """서비스가 정상인지 반환합니다."""
        return self._last_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


# ============================================================
# GPU 헬스 체크
# ============================================================


def check_gpu_health(
    device_id: int = 0,
    memory_threshold_percent: int = 90,
) -> ComponentHealth:
    """GPU 헬스를 체크합니다.

    Args:
        device_id: GPU 디바이스 ID
        memory_threshold_percent: 메모리 사용률 임계값 (%)

    Returns:
        ComponentHealth
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return ComponentHealth(
                name=f"gpu_{device_id}",
                status=HealthStatus.DEGRADED,
                message="CUDA not available",
            )

        if device_id >= torch.cuda.device_count():
            return ComponentHealth(
                name=f"gpu_{device_id}",
                status=HealthStatus.UNHEALTHY,
                message=f"GPU {device_id} not found",
            )

        # 메모리 사용량 확인
        memory_allocated = torch.cuda.memory_allocated(device_id)
        memory_reserved = torch.cuda.memory_reserved(device_id)
        total_memory = torch.cuda.get_device_properties(device_id).total_memory

        usage_percent = (memory_allocated / total_memory) * 100

        details = {
            "device_name": torch.cuda.get_device_name(device_id),
            "memory_allocated_mb": memory_allocated / (1024 * 1024),
            "memory_reserved_mb": memory_reserved / (1024 * 1024),
            "total_memory_mb": total_memory / (1024 * 1024),
            "usage_percent": usage_percent,
        }

        if usage_percent >= memory_threshold_percent:
            return ComponentHealth(
                name=f"gpu_{device_id}",
                status=HealthStatus.DEGRADED,
                message=f"High memory usage: {usage_percent:.1f}%",
                details=details,
            )

        return ComponentHealth(
            name=f"gpu_{device_id}",
            status=HealthStatus.HEALTHY,
            message="GPU healthy",
            details=details,
        )

    except ImportError:
        return ComponentHealth(
            name=f"gpu_{device_id}",
            status=HealthStatus.UNKNOWN,
            message="PyTorch not installed",
        )
    except Exception as e:
        return ComponentHealth(
            name=f"gpu_{device_id}",
            status=HealthStatus.UNHEALTHY,
            message=f"GPU check failed: {str(e)}",
        )


# ============================================================
# 모델 풀 헬스 체크
# ============================================================


def create_model_pool_check(
    model_manager: Any,
    min_available_ratio: float = 0.1,
) -> Callable[[], ComponentHealth]:
    """모델 풀 헬스 체크 함수를 생성합니다.

    Args:
        model_manager: ModelManager 인스턴스
        min_available_ratio: 최소 가용 인스턴스 비율

    Returns:
        헬스 체크 함수
    """

    def check() -> ComponentHealth:
        try:
            status_dict = model_manager.get_status()
            pools_status = status_dict.get("pools", {})

            total_instances = 0
            available_instances = 0
            pool_details = {}

            for instance_name, pool_info in pools_status.items():
                pool_total = pool_info.get("pool_size", 0)
                pool_available = pool_info.get("available", 0)

                total_instances += pool_total
                available_instances += pool_available

                pool_details[instance_name] = {
                    "total": pool_total,
                    "available": pool_available,
                    "in_use": pool_total - pool_available,
                }

            if total_instances == 0:
                return ComponentHealth(
                    name="model_pool",
                    status=HealthStatus.DEGRADED,
                    message="No model instances configured",
                    details=pool_details,
                )

            available_ratio = available_instances / total_instances

            if available_ratio < min_available_ratio:
                return ComponentHealth(
                    name="model_pool",
                    status=HealthStatus.DEGRADED,
                    message=f"Low available instances: {available_ratio:.1%}",
                    details=pool_details,
                )

            return ComponentHealth(
                name="model_pool",
                status=HealthStatus.HEALTHY,
                message=f"Model pool healthy: {available_instances}/{total_instances} available",
                details=pool_details,
            )

        except Exception as e:
            return ComponentHealth(
                name="model_pool",
                status=HealthStatus.UNHEALTHY,
                message=f"Pool check failed: {str(e)}",
            )

    return check


# ============================================================
# gRPC 서버 헬스 체크
# ============================================================


def create_grpc_server_check(server: Any) -> Callable[[], ComponentHealth]:
    """gRPC 서버 헬스 체크 함수를 생성합니다.

    Args:
        server: GRPCServer 인스턴스

    Returns:
        헬스 체크 함수
    """

    def check() -> ComponentHealth:
        try:
            if not hasattr(server, "is_running"):
                return ComponentHealth(
                    name="grpc_server",
                    status=HealthStatus.UNKNOWN,
                    message="Server status unknown",
                )

            if server.is_running:
                return ComponentHealth(
                    name="grpc_server",
                    status=HealthStatus.HEALTHY,
                    message="gRPC server running",
                )
            else:
                return ComponentHealth(
                    name="grpc_server",
                    status=HealthStatus.UNHEALTHY,
                    message="gRPC server not running",
                )

        except Exception as e:
            return ComponentHealth(
                name="grpc_server",
                status=HealthStatus.UNHEALTHY,
                message=f"Server check failed: {str(e)}",
            )

    return check


# ============================================================
# 메모리 헬스 체크
# ============================================================


def check_system_memory(threshold_percent: int = 90) -> ComponentHealth:
    """시스템 메모리 헬스를 체크합니다.

    Args:
        threshold_percent: 메모리 사용률 임계값 (%)

    Returns:
        ComponentHealth
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        usage_percent = memory.percent

        details = {
            "total_mb": memory.total / (1024 * 1024),
            "available_mb": memory.available / (1024 * 1024),
            "used_mb": memory.used / (1024 * 1024),
            "percent": usage_percent,
        }

        if usage_percent >= threshold_percent:
            return ComponentHealth(
                name="system_memory",
                status=HealthStatus.DEGRADED,
                message=f"High memory usage: {usage_percent:.1f}%",
                details=details,
            )

        return ComponentHealth(
            name="system_memory",
            status=HealthStatus.HEALTHY,
            message="Memory healthy",
            details=details,
        )

    except ImportError:
        return ComponentHealth(
            name="system_memory",
            status=HealthStatus.UNKNOWN,
            message="psutil not installed",
        )
    except Exception as e:
        return ComponentHealth(
            name="system_memory",
            status=HealthStatus.UNHEALTHY,
            message=f"Memory check failed: {str(e)}",
        )


# ============================================================
# 디스크 헬스 체크
# ============================================================


def check_disk_space(path: str = "/", threshold_percent: int = 90) -> ComponentHealth:
    """디스크 공간 헬스를 체크합니다.

    Args:
        path: 체크할 경로
        threshold_percent: 사용률 임계값 (%)

    Returns:
        ComponentHealth
    """
    try:
        import psutil

        disk = psutil.disk_usage(path)
        usage_percent = disk.percent

        details = {
            "path": path,
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percent": usage_percent,
        }

        if usage_percent >= threshold_percent:
            return ComponentHealth(
                name="disk_space",
                status=HealthStatus.DEGRADED,
                message=f"Low disk space: {100 - usage_percent:.1f}% free",
                details=details,
            )

        return ComponentHealth(
            name="disk_space",
            status=HealthStatus.HEALTHY,
            message="Disk healthy",
            details=details,
        )

    except ImportError:
        return ComponentHealth(
            name="disk_space",
            status=HealthStatus.UNKNOWN,
            message="psutil not installed",
        )
    except Exception as e:
        return ComponentHealth(
            name="disk_space",
            status=HealthStatus.UNHEALTHY,
            message=f"Disk check failed: {str(e)}",
        )
