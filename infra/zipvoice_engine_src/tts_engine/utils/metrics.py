# -*- coding: utf-8 -*-
"""Metrics Utilities - 메트릭 수집 유틸리티.

Prometheus 호환 메트릭 수집 및 노출을 제공합니다.
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tts_engine.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricValue:
    """메트릭 값 저장."""

    value: float = 0.0
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """카운터 메트릭.

    단조 증가하는 값을 추적합니다.

    Example:
        >>> counter = Counter("requests_total", "Total requests")
        >>> counter.inc()
        >>> counter.inc(5)
        >>> counter.inc(labels={"method": "POST"})
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        """카운터 초기화.

        Args:
            name: 메트릭 이름
            description: 설명
            labels: 레이블 이름 목록
        """
        self._name = name
        self._description = description
        self._label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """카운터를 증가시킵니다.

        Args:
            value: 증가량 (기본 1)
            labels: 레이블 값
        """
        if value < 0:
            raise ValueError("Counter can only increase")

        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] += value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """현재 값을 반환합니다.

        Args:
            labels: 레이블 값

        Returns:
            현재 카운터 값
        """
        label_key = self._get_label_key(labels)
        with self._lock:
            return self._values[label_key]

    def _get_label_key(self, labels: Optional[Dict[str, str]]) -> tuple:
        """레이블을 키로 변환합니다."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def collect(self) -> List[MetricValue]:
        """모든 값을 수집합니다."""
        result = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = dict(label_key) if label_key else {}
                result.append(MetricValue(value=value, labels=labels))
        return result


class Gauge:
    """게이지 메트릭.

    증가/감소 가능한 값을 추적합니다.

    Example:
        >>> gauge = Gauge("active_connections", "Active connections")
        >>> gauge.set(10)
        >>> gauge.inc()
        >>> gauge.dec()
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ):
        """게이지 초기화.

        Args:
            name: 메트릭 이름
            description: 설명
            labels: 레이블 이름 목록
        """
        self._name = name
        self._description = description
        self._label_names = labels or []
        self._values: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """값을 설정합니다.

        Args:
            value: 설정할 값
            labels: 레이블 값
        """
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] = value

    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """값을 증가시킵니다.

        Args:
            value: 증가량
            labels: 레이블 값
        """
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] += value

    def dec(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """값을 감소시킵니다.

        Args:
            value: 감소량
            labels: 레이블 값
        """
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] -= value

    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """현재 값을 반환합니다.

        Args:
            labels: 레이블 값

        Returns:
            현재 게이지 값
        """
        label_key = self._get_label_key(labels)
        with self._lock:
            return self._values[label_key]

    def _get_label_key(self, labels: Optional[Dict[str, str]]) -> tuple:
        """레이블을 키로 변환합니다."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def collect(self) -> List[MetricValue]:
        """모든 값을 수집합니다."""
        result = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = dict(label_key) if label_key else {}
                result.append(MetricValue(value=value, labels=labels))
        return result


class Histogram:
    """히스토그램 메트릭.

    값의 분포를 추적합니다.

    Example:
        >>> histogram = Histogram("request_duration_seconds", "Request duration")
        >>> histogram.observe(0.5)
        >>> histogram.observe(1.2, labels={"method": "POST"})
    """

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[tuple] = None,
    ):
        """히스토그램 초기화.

        Args:
            name: 메트릭 이름
            description: 설명
            labels: 레이블 이름 목록
            buckets: 버킷 경계값 (기본: DEFAULT_BUCKETS)
        """
        self._name = name
        self._description = description
        self._label_names = labels or []
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._bucket_counts: Dict[tuple, Dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self._buckets}
        )
        self._sums: Dict[tuple, float] = defaultdict(float)
        self._counts: Dict[tuple, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """값을 관측합니다.

        Args:
            value: 관측 값
            labels: 레이블 값
        """
        label_key = self._get_label_key(labels)
        with self._lock:
            self._sums[label_key] += value
            self._counts[label_key] += 1

            for bucket in self._buckets:
                if value <= bucket:
                    self._bucket_counts[label_key][bucket] += 1

    def get_percentile(
        self,
        percentile: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        """백분위수를 반환합니다.

        Args:
            percentile: 백분위수 (0-100)
            labels: 레이블 값

        Returns:
            백분위수 값 (근사값)
        """
        label_key = self._get_label_key(labels)
        with self._lock:
            total = self._counts[label_key]
            if total == 0:
                return None

            target = total * (percentile / 100.0)
            cumulative = 0

            for bucket in sorted(self._buckets):
                cumulative = self._bucket_counts[label_key][bucket]
                if cumulative >= target:
                    return bucket

            return self._buckets[-1]

    def get_mean(self, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """평균을 반환합니다.

        Args:
            labels: 레이블 값

        Returns:
            평균값 (None if no observations)
        """
        label_key = self._get_label_key(labels)
        with self._lock:
            count = self._counts[label_key]
            if count == 0:
                return None
            return self._sums[label_key] / count

    def get_count(self, labels: Optional[Dict[str, str]] = None) -> int:
        """관측 횟수를 반환합니다.

        Args:
            labels: 레이블 값

        Returns:
            관측 횟수
        """
        label_key = self._get_label_key(labels)
        with self._lock:
            return self._counts[label_key]

    def get_sum(self, labels: Optional[Dict[str, str]] = None) -> float:
        """관측값 합계를 반환합니다.

        Args:
            labels: 레이블 값

        Returns:
            합계
        """
        label_key = self._get_label_key(labels)
        with self._lock:
            return self._sums[label_key]

    def _get_label_key(self, labels: Optional[Dict[str, str]]) -> tuple:
        """레이블을 키로 변환합니다."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def buckets(self) -> tuple:
        return self._buckets

    def collect(self) -> Dict[str, Any]:
        """모든 값을 수집합니다."""
        result = {}
        with self._lock:
            for label_key in set(self._counts.keys()):
                labels = dict(label_key) if label_key else {}
                label_str = str(labels)
                result[label_str] = {
                    "count": self._counts[label_key],
                    "sum": self._sums[label_key],
                    "buckets": dict(self._bucket_counts[label_key]),
                    "labels": labels,
                }
        return result


class Timer:
    """타이머 컨텍스트 매니저.

    코드 블록의 실행 시간을 측정하여 히스토그램에 기록합니다.

    Example:
        >>> histogram = Histogram("request_duration")
        >>> with Timer(histogram):
        ...     do_something()
    """

    def __init__(
        self,
        histogram: Histogram,
        labels: Optional[Dict[str, str]] = None,
    ):
        """타이머 초기화.

        Args:
            histogram: 대상 히스토그램
            labels: 레이블 값
        """
        self._histogram = histogram
        self._labels = labels
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start is not None:
            elapsed = time.perf_counter() - self._start
            self._histogram.observe(elapsed, self._labels)


class MetricsRegistry:
    """메트릭 레지스트리.

    모든 메트릭을 중앙에서 관리하고 수집합니다.

    Example:
        >>> registry = MetricsRegistry()
        >>> counter = registry.counter("requests_total", "Total requests")
        >>> counter.inc()
        >>> metrics = registry.collect_all()
    """

    _instance: Optional["MetricsRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsRegistry":
        """싱글톤 인스턴스를 반환합니다."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._metrics: Dict[str, Any] = {}
        self._collectors: List[Callable[[], Dict[str, Any]]] = []
        self._initialized = True
        logger.debug("MetricsRegistry initialized")

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """카운터를 생성하거나 반환합니다.

        Args:
            name: 메트릭 이름
            description: 설명
            labels: 레이블 이름 목록

        Returns:
            Counter 인스턴스
        """
        if name not in self._metrics:
            self._metrics[name] = Counter(name, description, labels)
        return self._metrics[name]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """게이지를 생성하거나 반환합니다.

        Args:
            name: 메트릭 이름
            description: 설명
            labels: 레이블 이름 목록

        Returns:
            Gauge 인스턴스
        """
        if name not in self._metrics:
            self._metrics[name] = Gauge(name, description, labels)
        return self._metrics[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[tuple] = None,
    ) -> Histogram:
        """히스토그램을 생성하거나 반환합니다.

        Args:
            name: 메트릭 이름
            description: 설명
            labels: 레이블 이름 목록
            buckets: 버킷 경계값

        Returns:
            Histogram 인스턴스
        """
        if name not in self._metrics:
            self._metrics[name] = Histogram(name, description, labels, buckets)
        return self._metrics[name]

    def register_collector(
        self,
        collector: Callable[[], Dict[str, Any]],
    ) -> None:
        """외부 컬렉터를 등록합니다.

        Args:
            collector: 메트릭을 반환하는 콜백 함수
        """
        self._collectors.append(collector)

    def collect_all(self) -> Dict[str, Any]:
        """모든 메트릭을 수집합니다.

        Returns:
            메트릭 딕셔너리
        """
        result = {}

        # 등록된 메트릭 수집
        for name, metric in self._metrics.items():
            if hasattr(metric, "collect"):
                result[name] = {
                    "type": type(metric).__name__.lower(),
                    "description": metric.description,
                    "values": metric.collect(),
                }

        # 외부 컬렉터 수집
        for collector in self._collectors:
            try:
                external_metrics = collector()
                result.update(external_metrics)
            except Exception as e:
                logger.warning(f"Collector failed: {e}")

        return result

    def reset(self) -> None:
        """모든 메트릭을 초기화합니다 (테스트용)."""
        self._metrics.clear()
        self._collectors.clear()

    @classmethod
    def reset_instance(cls) -> None:
        """싱글톤 인스턴스를 초기화합니다 (테스트용)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._metrics.clear()
                cls._instance._collectors.clear()


# ============================================================
# TTS 전용 메트릭
# ============================================================

# 글로벌 레지스트리
_registry = MetricsRegistry()

# TTS 메트릭 정의
tts_requests_total = _registry.counter(
    "tts_requests_total",
    "Total TTS synthesis requests",
    labels=["voice_id", "status"],
)

tts_request_duration_seconds = _registry.histogram(
    "tts_request_duration_seconds",
    "TTS request processing duration in seconds",
    labels=["voice_id"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

tts_audio_duration_seconds = _registry.histogram(
    "tts_audio_duration_seconds",
    "Generated audio duration in seconds",
    labels=["voice_id"],
    buckets=(1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

tts_active_requests = _registry.gauge(
    "tts_active_requests",
    "Currently active TTS requests",
    labels=["voice_id"],
)

tts_rtf = _registry.histogram(
    "tts_rtf",
    "Real-Time Factor (processing time / audio duration)",
    labels=["voice_id", "runtime"],
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
)

model_pool_available = _registry.gauge(
    "tts_model_pool_available",
    "Available model instances in pool",
    labels=["instance_name"],
)

model_pool_total = _registry.gauge(
    "tts_model_pool_total",
    "Total model instances in pool",
    labels=["instance_name"],
)

gpu_memory_used_bytes = _registry.gauge(
    "tts_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    labels=["device"],
)

gpu_memory_total_bytes = _registry.gauge(
    "tts_gpu_memory_total_bytes",
    "Total GPU memory in bytes",
    labels=["device"],
)


def get_registry() -> MetricsRegistry:
    """글로벌 메트릭 레지스트리를 반환합니다."""
    return _registry


def record_synthesis_request(
    voice_id: str,
    status: str,
    duration_seconds: float,
    audio_duration_seconds: float,
    runtime: str = "pytorch",
) -> None:
    """합성 요청 메트릭을 기록합니다.

    Args:
        voice_id: 음성 ID
        status: 상태 (success, error)
        duration_seconds: 처리 시간 (초)
        audio_duration_seconds: 생성된 오디오 길이 (초)
        runtime: 런타임 타입
    """
    labels_status = {"voice_id": voice_id, "status": status}
    labels_voice = {"voice_id": voice_id}
    labels_rtf = {"voice_id": voice_id, "runtime": runtime}

    tts_requests_total.inc(labels=labels_status)
    tts_request_duration_seconds.observe(duration_seconds, labels=labels_voice)

    if audio_duration_seconds > 0:
        tts_audio_duration_seconds.observe(audio_duration_seconds, labels=labels_voice)
        rtf = duration_seconds / audio_duration_seconds
        tts_rtf.observe(rtf, labels=labels_rtf)


def update_gpu_metrics(device: str, used_bytes: int, total_bytes: int) -> None:
    """GPU 메트릭을 업데이트합니다.

    Args:
        device: GPU 디바이스 이름
        used_bytes: 사용 중인 메모리 (바이트)
        total_bytes: 전체 메모리 (바이트)
    """
    labels = {"device": device}
    gpu_memory_used_bytes.set(used_bytes, labels=labels)
    gpu_memory_total_bytes.set(total_bytes, labels=labels)


def update_model_pool_metrics(instance_name: str, available: int, total: int) -> None:
    """모델 풀 메트릭을 업데이트합니다.

    Args:
        instance_name: 모델 인스턴스 이름
        available: 사용 가능한 인스턴스 수
        total: 전체 인스턴스 수
    """
    labels = {"instance_name": instance_name}
    model_pool_available.set(available, labels=labels)
    model_pool_total.set(total, labels=labels)
