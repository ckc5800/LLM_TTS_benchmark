# -*- coding: utf-8 -*-
"""gRPC Server - TTS 엔진 gRPC 서버 설정 및 실행.

gRPC 서버 생성, 설정, 시작, 종료를 관리합니다.
"""

import asyncio
import signal
from typing import Optional

import grpc

from tts_engine.core.config import Settings, VoicesConfig, TlsConfig
from tts_engine.core.logging import get_logger
from tts_engine.server.servicer import TTSServicer
from tts_engine.services.model_manager import ModelManager
from tts_engine.services.synthesis_service import SynthesisService
from tts_engine.services.voice_manager import VoiceManager

# Proto 생성 코드 임포트
from tts_engine.proto import add_TTSServiceServicer_to_server

# Prometheus 메트릭 서버
from tts_engine.server.metrics_server import MetricsServer, metrics_registry

logger = get_logger(__name__)


class GRPCServer:
    """gRPC 서버 관리 클래스.

    서버 라이프사이클(시작, 종료, 그레이스풀 종료)을 관리합니다.

    Attributes:
        _settings: 애플리케이션 설정
        _server: gRPC 서버 인스턴스
        _servicer: TTS 서비스 핸들러
        _is_running: 실행 상태
        _metrics_server: Prometheus 메트릭 서버
    """

    def __init__(
        self,
        settings: Settings,
        synthesis_service: SynthesisService,
        voice_manager: VoiceManager,
        model_manager: ModelManager,
    ):
        """gRPC 서버 초기화.

        Args:
            settings: 애플리케이션 설정
            synthesis_service: 합성 서비스 인스턴스
            voice_manager: 음성 관리 인스턴스
            model_manager: 모델 관리 인스턴스
        """
        self._settings = settings
        self._synthesis_service = synthesis_service
        self._voice_manager = voice_manager
        self._model_manager = model_manager

        self._server: Optional[grpc.aio.Server] = None
        self._servicer: Optional[TTSServicer] = None
        self._metrics_server: Optional[MetricsServer] = None
        self._is_running = False
        self._shutdown_event = asyncio.Event()

        logger.info(
            "GRPCServer initialized",
            host=settings.server.grpc.host,
            port=settings.server.grpc.port,
        )

    async def start(self) -> None:
        """gRPC 서버를 시작합니다.

        서버를 생성하고 지정된 포트에서 서비스를 제공합니다.
        """
        if self._is_running:
            logger.warning("Server is already running")
            return

        grpc_config = self._settings.server.grpc

        # gRPC 서버 옵션 설정
        options = [
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.keepalive_time_ms", grpc_config.keepalive_time_ms),
            ("grpc.keepalive_timeout_ms", grpc_config.keepalive_timeout_ms),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 5000),
        ]

        # 서버 생성
        self._server = grpc.aio.server(
            options=options,
            maximum_concurrent_rpcs=grpc_config.max_concurrent_rpcs,
        )

        # Servicer 생성 및 등록
        self._servicer = TTSServicer(
            settings=self._settings,
            synthesis_service=self._synthesis_service,
            voice_manager=self._voice_manager,
            model_manager=self._model_manager,
        )

        add_TTSServiceServicer_to_server(self._servicer, self._server)

        # 포트 바인딩 (TLS 여부에 따라)
        address = f"{grpc_config.host}:{grpc_config.port}"
        tls_config = grpc_config.tls

        if tls_config.enabled:
            # TLS 활성화 - 보안 포트 바인딩
            server_credentials = self._create_server_credentials(tls_config)
            if server_credentials is None:
                raise ValueError("TLS enabled but failed to create credentials")
            self._server.add_secure_port(address, server_credentials)
            logger.info("TLS enabled for gRPC server", address=address)
        else:
            # TLS 비활성화 - 비보안 포트 바인딩
            self._server.add_insecure_port(address)

        # 서버 시작
        await self._server.start()
        self._is_running = True

        # 배치 모드 시작 (활성화된 경우)
        if self._synthesis_service.batch_mode_enabled:
            await self._synthesis_service.start_batch_mode()
            logger.info("Batch mode started with gRPC server")

        # Prometheus 메트릭 서버 시작 (설정된 경우)
        metrics_config = self._settings.server.metrics
        if metrics_config and metrics_config.enabled:
            self._metrics_server = MetricsServer(
                host=metrics_config.host,
                port=metrics_config.port,
            )
            metrics_started = await self._metrics_server.start()
            if metrics_started and metrics_registry:
                # 서버 정보 설정
                runtime = "unknown"
                for instance in self._settings.model_instances.values():
                    if instance.options and hasattr(instance.options, "runtime"):
                        runtime = instance.options.runtime or "pytorch"
                        break
                metrics_registry.set_info(version="1.0.0", runtime=runtime)

        logger.info(
            "gRPC server started",
            address=address,
            max_concurrent_rpcs=grpc_config.max_concurrent_rpcs,
            batch_mode=self._synthesis_service.batch_mode_enabled,
            metrics_enabled=metrics_config.enabled if metrics_config else False,
        )

    async def stop(self, grace_period: float = 5.0) -> None:
        """gRPC 서버를 종료합니다.

        Args:
            grace_period: 그레이스풀 종료 대기 시간 (초)
        """
        if not self._is_running or not self._server:
            logger.warning("Server is not running")
            return

        logger.info("Stopping gRPC server", grace_period=grace_period)

        # 배치 모드 중지
        if self._synthesis_service.batch_mode_enabled:
            await self._synthesis_service.stop_batch_mode()
            logger.info("Batch mode stopped")

        # Prometheus 메트릭 서버 중지
        if self._metrics_server and self._metrics_server.is_running:
            await self._metrics_server.stop()
            logger.info("Metrics server stopped")

        # 그레이스풀 종료
        await self._server.stop(grace_period)
        self._is_running = False
        self._shutdown_event.set()

        logger.info("gRPC server stopped")

    async def wait_for_termination(self) -> None:
        """서버 종료까지 대기합니다."""
        if self._server:
            await self._server.wait_for_termination()

    def _create_server_credentials(self, tls_config: TlsConfig) -> Optional[grpc.ServerCredentials]:
        """TLS 서버 자격 증명을 생성합니다.

        Args:
            tls_config: TLS 설정

        Returns:
            gRPC 서버 자격 증명 또는 None (실패 시)
        """
        if not tls_config.cert_file or not tls_config.key_file:
            logger.error("TLS enabled but cert_file or key_file not specified")
            return None

        try:
            # 인증서 파일 읽기
            with open(tls_config.cert_file, "rb") as f:
                server_cert = f.read()

            with open(tls_config.key_file, "rb") as f:
                server_key = f.read()

            # CA 인증서 (mTLS용, 선택)
            root_cert = None
            if tls_config.ca_file:
                with open(tls_config.ca_file, "rb") as f:
                    root_cert = f.read()

            # 클라이언트 인증 요구 여부
            if tls_config.require_client_cert:
                # mTLS: 클라이언트 인증서 필수
                credentials = grpc.ssl_server_credentials(
                    [(server_key, server_cert)],
                    root_certificates=root_cert,
                    require_client_auth=True,
                )
                logger.info("mTLS enabled - client certificate required")
            else:
                # 단방향 TLS: 서버 인증서만
                credentials = grpc.ssl_server_credentials(
                    [(server_key, server_cert)],
                    root_certificates=root_cert,
                    require_client_auth=False,
                )

            logger.info(
                "TLS credentials created",
                cert_file=tls_config.cert_file,
                require_client_cert=tls_config.require_client_cert,
            )
            return credentials

        except FileNotFoundError as e:
            logger.error(f"TLS certificate file not found: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create TLS credentials: {e}")
            return None

    def setup_signal_handlers(self) -> None:
        """시그널 핸들러를 설정합니다.

        SIGINT, SIGTERM 시그널을 받아 그레이스풀 종료합니다.
        """
        import platform

        # Windows에서는 add_signal_handler가 지원되지 않음
        if platform.system() == 'Windows':
            logger.debug("Signal handlers not available on Windows - use Ctrl+C to stop")
            return

        loop = asyncio.get_event_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s)),
            )

        logger.debug("Signal handlers registered")

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """시그널을 처리합니다.

        Args:
            sig: 수신된 시그널
        """
        logger.info("Received signal", signal=sig.name)
        await self.stop()

    @property
    def is_running(self) -> bool:
        """서버 실행 상태."""
        return self._is_running

    @property
    def servicer(self) -> Optional[TTSServicer]:
        """TTS Servicer 인스턴스."""
        return self._servicer


async def create_server(settings: Settings, voices_config: VoicesConfig) -> GRPCServer:
    """gRPC 서버를 생성합니다.

    모든 의존성을 초기화하고 서버를 준비합니다.

    Args:
        settings: 애플리케이션 설정
        voices_config: 음성 설정

    Returns:
        초기화된 GRPCServer 인스턴스
    """
    logger.info("Creating gRPC server with dependencies")

    # 음성 관리자 초기화
    voice_manager = VoiceManager(voices_config)

    # ONNX 런타임 사용 시 모델 풀 생성 스킵 (ONNXExecutor 세션 풀 사용)
    # 설정에서 런타임 확인
    skip_pool_runtimes = []
    for instance_name, instance_config in settings.model_instances.items():
        if instance_config.options and hasattr(instance_config.options, "runtime"):
            runtime = instance_config.options.runtime
            if runtime and runtime.lower() == "onnx":
                skip_pool_runtimes.append("onnx")
                logger.info(
                    "ONNX runtime detected - will use ONNXExecutor session pool",
                    instance_name=instance_name,
                )
                break

    # 모델 관리자 초기화 (voice_manager 주입)
    model_manager = ModelManager(
        model_types=settings.model_types,
        model_instances=settings.model_instances,
        voice_manager=voice_manager,
        skip_pool_runtimes=skip_pool_runtimes,
    )

    # 모델 풀 초기화
    await model_manager.initialize()

    # 합성 서비스 초기화
    synthesis_service = SynthesisService(
        settings=settings,
        model_manager=model_manager,
        voice_manager=voice_manager,
    )

    # ONNX 런타임일 때 세션 풀 초기화
    await synthesis_service.initialize()

    # 서버 생성
    server = GRPCServer(
        settings=settings,
        synthesis_service=synthesis_service,
        voice_manager=voice_manager,
        model_manager=model_manager,
    )

    return server


async def run_server(settings: Settings, voices_config: VoicesConfig) -> None:
    """gRPC 서버를 실행합니다.

    서버를 생성하고 시작한 후 종료까지 대기합니다.

    Args:
        settings: 애플리케이션 설정
        voices_config: 음성 설정
    """
    server = await create_server(settings, voices_config)

    try:
        # 시그널 핸들러 설정
        server.setup_signal_handlers()

        # 서버 시작
        await server.start()

        # 워밍업 실행 (설정된 경우)
        logger.debug(
            "Warmup settings check",
            warmup_enabled=settings.performance.warmup.enabled,
            warmup_on_startup=settings.performance.warmup.on_startup,
        )
        if settings.performance.warmup.enabled and settings.performance.warmup.on_startup:
            logger.info("Running startup warmup")
            warmup_config = settings.performance.warmup
            warmup_results = await server._model_manager.warmup(
                test_text=warmup_config.test_text,
                repeat_count=warmup_config.repeat_count,
            )
            success_count = sum(1 for success in warmup_results.values() if success)
            total_count = len(warmup_results)
            logger.info(
                "Startup warmup completed",
                success=success_count,
                total=total_count,
                results=warmup_results,
            )

        logger.info(
            "TTS Engine gRPC server is ready",
            address=f"{settings.server.grpc.host}:{settings.server.grpc.port}",
        )

        # 종료까지 대기
        await server.wait_for_termination()

    except Exception as e:
        logger.error("Server error", error=str(e))
        raise

    finally:
        if server.is_running:
            await server.stop()
        logger.info("Server shutdown complete")
