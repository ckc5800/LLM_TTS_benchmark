# -*- coding: utf-8 -*-
"""합성 통계 관리 - 시간 윈도우 기반."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


@dataclass
class RequestRecord:
    """요청 기록."""
    timestamp: float
    processing_time_ms: float
    success: bool


class TimeWindowStats:
    """시간 윈도우 기반 통계."""

    # 윈도우 크기 (초)
    WINDOW_1MIN = 60
    WINDOW_5MIN = 300
    WINDOW_1HOUR = 3600

    def __init__(self, max_records: int = 100000):
        self._records: Deque[RequestRecord] = deque(maxlen=max_records)
        self._lock = asyncio.Lock()

    async def add_record(self, processing_time_ms: float, success: bool) -> None:
        """요청 기록 추가."""
        record = RequestRecord(
            timestamp=time.time(),
            processing_time_ms=processing_time_ms,
            success=success,
        )
        async with self._lock:
            self._records.append(record)

    def _get_window_stats(self, window_seconds: int) -> dict:
        """특정 윈도우의 통계 계산."""
        now = time.time()
        cutoff = now - window_seconds

        total = 0
        success = 0
        failed = 0
        total_time = 0.0
        min_time = float("inf")
        max_time = 0.0

        for record in self._records:
            if record.timestamp >= cutoff:
                total += 1
                total_time += record.processing_time_ms
                min_time = min(min_time, record.processing_time_ms)
                max_time = max(max_time, record.processing_time_ms)
                if record.success:
                    success += 1
                else:
                    failed += 1

        avg_time = total_time / total if total > 0 else 0.0
        rps = total / window_seconds if window_seconds > 0 else 0.0

        return {
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0.0,
            "avg_time_ms": avg_time,
            "min_time_ms": min_time if min_time != float("inf") else 0.0,
            "max_time_ms": max_time,
            "requests_per_second": rps,
        }

    def get_stats_1min(self) -> dict:
        """1분 윈도우 통계."""
        return self._get_window_stats(self.WINDOW_1MIN)

    def get_stats_5min(self) -> dict:
        """5분 윈도우 통계."""
        return self._get_window_stats(self.WINDOW_5MIN)

    def get_stats_1hour(self) -> dict:
        """1시간 윈도우 통계."""
        return self._get_window_stats(self.WINDOW_1HOUR)

    def get_all_window_stats(self) -> dict:
        """모든 윈도우 통계."""
        return {
            "1min": self.get_stats_1min(),
            "5min": self.get_stats_5min(),
            "1hour": self.get_stats_1hour(),
        }


class SynthesisStats:
    """합성 요청 통계 관리 - 누적 + 시간 윈도우 + Prometheus."""

    def __init__(self, enable_prometheus: bool = True):
        # 누적 통계
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_processing_time_ms = 0.0
        self._min_processing_time_ms = float("inf")
        self._max_processing_time_ms = 0.0
        self._active_requests: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._start_time = time.time()

        # 시간 윈도우 통계
        self._window_stats = TimeWindowStats()

        # Prometheus 메트릭 (metrics_server.py의 싱글톤 사용)
        self._prometheus_enabled = enable_prometheus

    def _get_metrics_registry(self):
        """Prometheus 메트릭 레지스트리 (지연 로딩)."""
        if not self._prometheus_enabled:
            return None
        try:
            from tts_engine.server.metrics_server import metrics_registry
            return metrics_registry
        except ImportError:
            return None

    async def start_request(self, request_id: str, start_time: float) -> None:
        """요청 시작 기록."""
        async with self._lock:
            self._active_requests[request_id] = start_time
            self._total_requests += 1

        registry = self._get_metrics_registry()
        if registry:
            registry.active_requests.set(len(self._active_requests))

    async def end_request(self, request_id: str) -> None:
        """요청 종료 기록."""
        async with self._lock:
            self._active_requests.pop(request_id, None)

        registry = self._get_metrics_registry()
        if registry:
            registry.active_requests.set(len(self._active_requests))

    async def update(self, processing_time_ms: float, success: bool) -> None:
        """통계 업데이트."""
        async with self._lock:
            if success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1

            self._total_processing_time_ms += processing_time_ms
            self._min_processing_time_ms = min(self._min_processing_time_ms, processing_time_ms)
            self._max_processing_time_ms = max(self._max_processing_time_ms, processing_time_ms)

        # 시간 윈도우에 기록
        await self._window_stats.add_record(processing_time_ms, success)

        # Prometheus 메트릭은 servicer.py에서 직접 기록

    def update_pool_status(self, active: int, waiting: int, available: int) -> None:
        """풀 상태 업데이트 (Prometheus용)."""
        registry = self._get_metrics_registry()
        if registry:
            registry.pool_active_slots.set(active)
            registry.pool_waiting.set(waiting)
            registry.pool_available_slots.set(available)

    def get_stats(self) -> dict:
        """누적 통계 반환."""
        avg_time = 0.0
        if self._total_requests > 0:
            avg_time = self._total_processing_time_ms / self._total_requests

        uptime = time.time() - self._start_time

        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "active_requests": len(self._active_requests),
            "avg_processing_time_ms": avg_time,
            "min_processing_time_ms": (
                self._min_processing_time_ms if self._min_processing_time_ms != float("inf") else 0.0
            ),
            "max_processing_time_ms": self._max_processing_time_ms,
            "success_rate": (
                self._successful_requests / self._total_requests if self._total_requests > 0 else 0.0
            ),
            "uptime_seconds": uptime,
            "requests_per_second": self._total_requests / uptime if uptime > 0 else 0.0,
        }

    def get_window_stats(self) -> dict:
        """시간 윈도우 통계 반환."""
        return self._window_stats.get_all_window_stats()

    def get_full_stats(self) -> dict:
        """전체 통계 반환 (누적 + 윈도우)."""
        return {
            "cumulative": self.get_stats(),
            "windows": self.get_window_stats(),
        }

    def get_active_requests(self) -> Dict[str, float]:
        """활성 요청 반환."""
        return dict(self._active_requests)

    @property
    def active_count(self) -> int:
        """활성 요청 수."""
        return len(self._active_requests)

    @property
    def prometheus_enabled(self) -> bool:
        """Prometheus 활성화 여부."""
        registry = self._get_metrics_registry()
        return registry is not None
