# -*- coding: utf-8 -*-
"""Prometheus 메트릭 HTTP 서버.

Prometheus가 스크래핑할 수 있는 /metrics 엔드포인트를 제공합니다.

Usage:
    from tts_engine.server.metrics_server import MetricsServer

    # 서버 시작
    metrics_server = MetricsServer(port=9090)
    await metrics_server.start()

    # 서버 중지
    await metrics_server.stop()

Prometheus 설정 예시 (prometheus.yml):
    scrape_configs:
      - job_name: 'tts-engine'
        static_configs:
          - targets: ['localhost:9090']
        scrape_interval: 15s
"""

import asyncio
from typing import Optional
from aiohttp import web

from tts_engine.core.logging import get_logger

logger = get_logger(__name__)

# Prometheus 클라이언트 (선택적)
try:
    from prometheus_client import (
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        Summary,
        Info,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics endpoint disabled.")


class PrometheusMetricsRegistry:
    """TTS 엔진용 Prometheus 메트릭 레지스트리.

    모든 메트릭을 중앙에서 관리합니다.
    """

    _instance: Optional["PrometheusMetricsRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        if not PROMETHEUS_AVAILABLE:
            return

        # ============================================================
        # 서버 정보
        # ============================================================
        self.info = Info(
            "tts_engine",
            "TTS Engine information",
        )

        # ============================================================
        # 요청 카운터
        # ============================================================
        self.requests_total = Counter(
            "tts_engine_requests_total",
            "Total number of synthesis requests",
            ["method", "status"],  # method: sync/streaming, status: success/failed
        )

        self.requests_rejected = Counter(
            "tts_engine_requests_rejected_total",
            "Total number of rejected requests (queue full)",
        )

        # ============================================================
        # 게이지 (현재 상태)
        # ============================================================
        self.active_requests = Gauge(
            "tts_engine_active_requests",
            "Number of currently active requests",
        )

        self.queue_size = Gauge(
            "tts_engine_queue_size",
            "Current request queue size",
        )

        self.queue_max_size = Gauge(
            "tts_engine_queue_max_size",
            "Maximum request queue size",
        )

        self.pool_total_slots = Gauge(
            "tts_engine_pool_total_slots",
            "Total executor pool slots",
        )

        self.pool_active_slots = Gauge(
            "tts_engine_pool_active_slots",
            "Active executor pool slots (GPU in use)",
        )

        self.pool_waiting = Gauge(
            "tts_engine_pool_waiting",
            "Requests waiting for GPU slot",
        )

        self.pool_available_slots = Gauge(
            "tts_engine_pool_available_slots",
            "Available executor pool slots",
        )

        # ============================================================
        # GPU 메트릭
        # ============================================================
        self.gpu_memory_used = Gauge(
            "tts_engine_gpu_memory_used_mb",
            "GPU memory used in MB",
            ["device"],  # device: cuda:0, cuda:1, etc.
        )

        self.gpu_memory_total = Gauge(
            "tts_engine_gpu_memory_total_mb",
            "GPU total memory in MB",
            ["device"],
        )

        self.gpu_memory_percent = Gauge(
            "tts_engine_gpu_memory_percent",
            "GPU memory usage percentage",
            ["device"],
        )

        self.gpu_utilization = Gauge(
            "tts_engine_gpu_utilization_percent",
            "GPU compute utilization percentage",
            ["device"],
        )

        self.gpu_temperature = Gauge(
            "tts_engine_gpu_temperature_celsius",
            "GPU temperature in Celsius",
            ["device"],
        )

        # ============================================================
        # 히스토그램 (처리 시간 분포)
        # ============================================================
        self.processing_time = Histogram(
            "tts_engine_processing_time_seconds",
            "Request processing time in seconds",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        self.audio_duration = Histogram(
            "tts_engine_audio_duration_seconds",
            "Generated audio duration in seconds",
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
        )

        # ============================================================
        # Summary (백분위수)
        # ============================================================
        self.processing_latency = Summary(
            "tts_engine_processing_latency_seconds",
            "Processing latency summary with quantiles",
        )

        # ============================================================
        # 윈도우별 RPS (커스텀 게이지)
        # ============================================================
        self.rps_1min = Gauge(
            "tts_engine_rps_1min",
            "Requests per second (last 1 minute)",
        )

        self.rps_5min = Gauge(
            "tts_engine_rps_5min",
            "Requests per second (last 5 minutes)",
        )

        self.rps_1hour = Gauge(
            "tts_engine_rps_1hour",
            "Requests per second (last 1 hour)",
        )

        logger.info("Prometheus metrics registry initialized")

    def set_info(self, version: str, runtime: str):
        """서버 정보 설정."""
        if PROMETHEUS_AVAILABLE:
            self.info.info({
                "version": version,
                "runtime": runtime,
            })

    def record_request(self, method: str, status: str, processing_time_seconds: float):
        """요청 완료 기록."""
        if not PROMETHEUS_AVAILABLE:
            return
        self.requests_total.labels(method=method, status=status).inc()
        self.processing_time.observe(processing_time_seconds)
        self.processing_latency.observe(processing_time_seconds)

    def record_audio_duration(self, duration_seconds: float):
        """생성된 오디오 길이 기록."""
        if PROMETHEUS_AVAILABLE:
            self.audio_duration.observe(duration_seconds)

    def record_rejection(self):
        """요청 거부 기록."""
        if PROMETHEUS_AVAILABLE:
            self.requests_rejected.inc()

    def update_gauges(
        self,
        active_requests: int = 0,
        queue_size: int = 0,
        queue_max_size: int = 0,
        pool_total: int = 0,
        pool_active: int = 0,
        pool_waiting: int = 0,
        pool_available: int = 0,
        rps_1min: float = 0.0,
        rps_5min: float = 0.0,
        rps_1hour: float = 0.0,
    ):
        """게이지 값 업데이트."""
        if not PROMETHEUS_AVAILABLE:
            return
        self.active_requests.set(active_requests)
        self.queue_size.set(queue_size)
        self.queue_max_size.set(queue_max_size)
        self.pool_total_slots.set(pool_total)
        self.pool_active_slots.set(pool_active)
        self.pool_waiting.set(pool_waiting)
        self.pool_available_slots.set(pool_available)
        self.rps_1min.set(rps_1min)
        self.rps_5min.set(rps_5min)
        self.rps_1hour.set(rps_1hour)


# 싱글톤 인스턴스
metrics_registry = PrometheusMetricsRegistry() if PROMETHEUS_AVAILABLE else None


class MetricsServer:
    """Prometheus 메트릭 HTTP 서버.

    /metrics 엔드포인트를 제공하여 Prometheus가 스크래핑할 수 있게 합니다.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9090):
        self._host = host
        self._port = port
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._is_running = False

    async def start(self) -> bool:
        """메트릭 서버 시작."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Metrics server not started.")
            return False

        if self._is_running:
            logger.warning("Metrics server is already running")
            return True

        try:
            self._app = web.Application()
            self._app.router.add_get("/metrics", self._handle_metrics)
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/", self._handle_root)

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()

            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()

            self._is_running = True
            logger.info(
                "Prometheus metrics server started",
                host=self._host,
                port=self._port,
                endpoint=f"http://{self._host}:{self._port}/metrics",
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False

    async def stop(self):
        """메트릭 서버 중지."""
        if not self._is_running:
            return

        if self._runner:
            await self._runner.cleanup()

        self._is_running = False
        logger.info("Prometheus metrics server stopped")

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """GET /metrics - Prometheus 메트릭 반환."""
        try:
            metrics_output = generate_latest(REGISTRY)
            # CONTENT_TYPE_LATEST는 "text/plain; charset=utf-8" 형식
            # aiohttp는 content_type에 charset을 포함하면 에러 발생
            return web.Response(
                body=metrics_output,
                content_type="text/plain",
                charset="utf-8",
            )
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return web.Response(status=500, text=str(e))

    async def _handle_health(self, request: web.Request) -> web.Response:
        """GET /health - 헬스체크."""
        return web.json_response({"status": "ok"})

    async def _handle_root(self, request: web.Request) -> web.Response:
        """GET / - 안내 페이지."""
        html = """
        <html>
        <head><title>TTS Engine Metrics</title></head>
        <body>
            <h1>TTS Engine Metrics Server</h1>
            <p>Prometheus metrics endpoint: <a href="/metrics">/metrics</a></p>
            <p>Health check: <a href="/health">/health</a></p>
        </body>
        </html>
        """
        return web.Response(text=html, content_type="text/html")

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def endpoint(self) -> str:
        return f"http://{self._host}:{self._port}/metrics"
