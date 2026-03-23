# -*- coding: utf-8 -*-
"""헬스체크 및 통계 RPC 핸들러."""

import time
from typing import TYPE_CHECKING

import grpc

from tts_engine.core.logging import get_logger

from tts_engine.proto import (
    ErrorCode,
    ErrorDetail,
    GetGpuStatusRequest,
    GetGpuStatusResponse,
    GetServerStatsRequest,
    GPUStatus,
    HealthCheckRequest,
    HealthCheckResponse,
    HealthStatus,
    ModelHealthChecks,
    ModelHealthStatus,
    ServerStatsResponse,
    WarmupAllRequest,
    WarmupAllResponse,
    WarmupRequest,
    WarmupResponse,
    WarmupResult,
    WindowStats,
)

if TYPE_CHECKING:
    from tts_engine.core.config import Settings
    from tts_engine.services.model_manager import ModelManager
    from tts_engine.services.synthesis_service import SynthesisService

logger = get_logger(__name__)


class HealthHandlersMixin:
    """헬스체크 및 통계 핸들러 믹스인."""

    _settings: "Settings"
    _synthesis_service: "SynthesisService"
    _model_manager: "ModelManager"
    _start_time: float

    def _create_window_stats(self, window_data: dict) -> WindowStats:
        """딕셔너리에서 WindowStats 객체 생성.

        Args:
            window_data: 시간 윈도우 통계 딕셔너리

        Returns:
            WindowStats protobuf 객체
        """
        return WindowStats(
            total=window_data.get("total", 0),
            success=window_data.get("success", 0),
            failed=window_data.get("failed", 0),
            success_rate=window_data.get("success_rate", 0.0),
            avg_time_ms=window_data.get("avg_time_ms", 0.0),
            min_time_ms=window_data.get("min_time_ms", 0.0),
            max_time_ms=window_data.get("max_time_ms", 0.0),
            requests_per_second=window_data.get("requests_per_second", 0.0),
        )

    async def HealthCheck(
        self,
        request: HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> HealthCheckResponse:
        """서버 헬스체크.

        Args:
            request: HealthCheckRequest
                - model_name: 특정 모델만 체크 (선택적)
                - include_details: 상세 정보 포함 여부
        """
        request_id = request.request_id or ""
        model_name_filter = request.model_name if request.HasField("model_name") else None
        include_details = request.include_details

        logger.debug(
            "HealthCheck request",
            request_id=request_id,
            model_name=model_name_filter,
            include_details=include_details,
        )

        try:
            model_statuses = []
            overall_healthy = True

            # model_name 파라미터로 특정 모델만 체크하거나 전체 체크
            if model_name_filter:
                # 특정 모델만 체크
                instances_to_check = [model_name_filter]
            else:
                # 전체 모델 체크
                instances_to_check = self._model_manager.list_instances()

            for instance_name in instances_to_check:
                pool = self._model_manager.get_pool(instance_name)
                if not pool:
                    # 모델 인스턴스를 찾을 수 없는 경우
                    logger.warning(
                        "Model instance not found",
                        instance_name=instance_name,
                        request_id=request_id,
                        mode="specific" if model_name_filter else "all",
                    )

                    if model_name_filter:
                        # 특정 모델 요청인 경우 에러 응답
                        return HealthCheckResponse(
                            request_id=request_id,
                            healthy=False,
                            status=HealthStatus.HEALTH_STATUS_UNHEALTHY,
                            status_message=f"Model instance '{instance_name}' not found",
                            timestamp_ms=int(time.time() * 1000),
                            uptime_ms=int((time.time() - self._start_time) * 1000),
                        )
                    # 전체 체크 모드: 스킵하고 다음 모델 확인
                    continue

                pool_status = pool.get_status()
                is_healthy = pool_status.get("available", 0) > 0

                if not is_healthy:
                    overall_healthy = False

                # include_details에 따라 상세 정보 포함 여부 결정
                if include_details:
                    # 상세 정보 포함
                    model_status = ModelHealthStatus(
                        model_name=instance_name,
                        model_type=pool_status.get("model_type", "unknown"),
                        healthy=is_healthy,
                        status=(
                            HealthStatus.HEALTH_STATUS_HEALTHY
                            if is_healthy
                            else HealthStatus.HEALTH_STATUS_UNHEALTHY
                        ),
                        status_message="OK" if is_healthy else "No available instances",
                        checks=ModelHealthChecks(
                            model_loaded=pool_status.get("initialized", False),
                            voices_loaded=True,
                            config_valid=True,
                            gpu_available=True,
                            pool_healthy=is_healthy,
                        ),
                        pool_total=pool_status.get("total", 0),
                        pool_available=pool_status.get("available", 0),
                        pool_in_use=pool_status.get("in_use", 0),
                    )
                else:
                    # 간략한 정보만 포함
                    model_status = ModelHealthStatus(
                        model_name=instance_name,
                        model_type=pool_status.get("model_type", "unknown"),
                        healthy=is_healthy,
                        status=(
                            HealthStatus.HEALTH_STATUS_HEALTHY
                            if is_healthy
                            else HealthStatus.HEALTH_STATUS_UNHEALTHY
                        ),
                        status_message="OK" if is_healthy else "Unhealthy",
                        # checks, pool 정보는 생략
                    )
                model_statuses.append(model_status)

            uptime_ms = int((time.time() - self._start_time) * 1000)

            return HealthCheckResponse(
                request_id=request_id,
                healthy=overall_healthy,
                status=(
                    HealthStatus.HEALTH_STATUS_HEALTHY
                    if overall_healthy
                    else HealthStatus.HEALTH_STATUS_DEGRADED
                ),
                status_message="All systems operational" if overall_healthy else "Degraded",
                models=model_statuses,
                timestamp_ms=int(time.time() * 1000),
                uptime_ms=uptime_ms,
            )

        except Exception as e:
            logger.error("HealthCheck failed", request_id=request_id, error=str(e))
            return HealthCheckResponse(
                request_id=request_id,
                healthy=False,
                status=HealthStatus.HEALTH_STATUS_UNHEALTHY,
                status_message=f"Health check failed: {e}",
                timestamp_ms=int(time.time() * 1000),
                uptime_ms=int((time.time() - self._start_time) * 1000),
            )

    async def GetGpuStatus(
        self,
        request: GetGpuStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> GetGpuStatusResponse:
        """GPU 상태 조회."""
        from tts_engine.services.gpu_stats import gpu_stats

        request_id = request.request_id or ""
        logger.debug("GetGpuStatus request", request_id=request_id)

        gpu_statuses = []

        # 특정 디바이스 요청
        if request.HasField("device_id"):
            status = gpu_stats.get_gpu_status(request.device_id)
            gpu_statuses.append(self._convert_to_proto_gpu_status(status))
        else:
            # 모든 디바이스
            for status in gpu_stats.get_all_gpu_status():
                gpu_statuses.append(self._convert_to_proto_gpu_status(status))

        return GetGpuStatusResponse(
            request_id=request_id,
            gpus=gpu_statuses,
            gpu_count=len(gpu_statuses),
            timestamp_ms=int(time.time() * 1000),
        )

    def _convert_to_proto_gpu_status(self, status) -> GPUStatus:
        """GPUStatusData를 Proto GPUStatus로 변환."""
        return GPUStatus(
            available=status.available,
            device=status.device,
            device_id=status.device_id,
            device_name=status.device_name,
            total_memory_mb=status.total_memory_mb,
            used_memory_mb=status.used_memory_mb,
            free_memory_mb=status.free_memory_mb,
            memory_percent=status.memory_percent,
            utilization_percent=status.utilization_percent,
            temperature_c=status.temperature_c,
            driver_version=status.driver_version,
            cuda_version=status.cuda_version,
            error=status.error,
        )

    async def GetServerStats(
        self,
        request: GetServerStatsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ServerStatsResponse:
        """서버 통계 조회.

        Args:
            request: GetServerStatsRequest
                - include_history: 시간 윈도우 통계 포함 여부
        """
        request_id = request.request_id or ""
        include_history = request.include_history

        logger.debug(
            "GetServerStats request",
            request_id=request_id,
            include_history=include_history,
        )

        try:
            synthesis_stats = self._synthesis_service.get_stats()

            # 시간 윈도우 통계 가져오기 (requests_per_second 계산용)
            window_stats = self._synthesis_service.get_window_stats()
            window_1min = window_stats.get("1min", {})

            # 1분 윈도우에서 실제 RPS 가져오기
            requests_per_second = window_1min.get("requests_per_second", 0.0)

            total_pool = 0
            available_pool = 0
            for instance_name in self._model_manager.list_instances():
                pool = self._model_manager.get_pool(instance_name)
                if pool:
                    status = pool.get_status()
                    total_pool += status.get("total", 0)
                    available_pool += status.get("available", 0)

            uptime_seconds = int(time.time() - self._start_time)

            # 기본 응답 구성
            response = ServerStatsResponse(
                request_id=request_id,
                active_requests=synthesis_stats.get("active_requests", 0),
                queued_requests=0,
                active_batches=0,
                total_requests=synthesis_stats.get("total_requests", 0),
                successful_requests=synthesis_stats.get("successful_requests", 0),
                failed_requests=synthesis_stats.get("failed_requests", 0),
                avg_processing_time_ms=int(synthesis_stats.get("avg_processing_time_ms", 0)),
                min_processing_time_ms=int(synthesis_stats.get("min_processing_time_ms", 0)),
                max_processing_time_ms=int(synthesis_stats.get("max_processing_time_ms", 0)),
                requests_per_second=requests_per_second,
                uptime_seconds=uptime_seconds,
                start_time_ms=int(self._start_time * 1000),
                version=getattr(self._settings.server, 'version', "1.0.0"),
                model_pool_total=total_pool,
                model_pool_available=available_pool,
                timestamp_ms=int(time.time() * 1000),
            )

            # include_history가 true인 경우 시간 윈도우 통계 추가
            if include_history:
                window_5min = window_stats.get("5min", {})
                window_1hour = window_stats.get("1hour", {})

                response.stats_1min.CopyFrom(self._create_window_stats(window_1min))
                response.stats_5min.CopyFrom(self._create_window_stats(window_5min))
                response.stats_1hour.CopyFrom(self._create_window_stats(window_1hour))

            return response

        except Exception as e:
            logger.error("GetServerStats failed", request_id=request_id, error=str(e))
            return ServerStatsResponse(
                request_id=request_id,
                timestamp_ms=int(time.time() * 1000),
            )

    async def WarmupModel(
        self,
        request: WarmupRequest,
        context: grpc.aio.ServicerContext,
    ) -> WarmupResponse:
        """모델 워밍업."""
        request_id = request.request_id or ""
        start_time = time.time()

        logger.info(
            f"[{request_id}] WarmupModel started",
            model_instance=request.model_instance,
        )

        try:
            test_text = (
                request.test_text
                if request.HasField("test_text")
                else "워밍업 테스트 문장입니다."
            )
            repeat_count = (
                request.repeat_count if request.HasField("repeat_count") else 3
            )

            warmup_results = await self._model_manager.warmup(
                test_text=test_text,
                repeat_count=repeat_count,
                instance_name=request.model_instance,
            )
            success = warmup_results.get(request.model_instance, False)

            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"[{request_id}] WarmupModel completed",
                model_instance=request.model_instance,
                success=success,
                duration_ms=duration_ms,
            )

            return WarmupResponse(
                request_id=request_id,
                success=success,
                model_instance=request.model_instance,
                message="Warmup completed" if success else "Warmup failed",
                duration_ms=duration_ms,
                timestamp_ms=int(time.time() * 1000),
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] WarmupModel failed",
                model_instance=request.model_instance,
                duration_ms=duration_ms,
                error=str(e),
            )

            return WarmupResponse(
                request_id=request_id,
                success=False,
                model_instance=request.model_instance,
                message=f"Warmup failed: {e}",
                duration_ms=duration_ms,
                timestamp_ms=int(time.time() * 1000),
                error=ErrorDetail(
                    code=ErrorCode.ERROR_INTERNAL,
                    message=str(e),
                    request_id=request_id,
                    timestamp_ms=int(time.time() * 1000),
                ),
            )

    async def WarmupAllModels(
        self,
        request: WarmupAllRequest,
        context: grpc.aio.ServicerContext,
    ) -> WarmupAllResponse:
        """모든 모델 워밍업."""
        request_id = request.request_id or ""
        start_time = time.time()

        logger.info(f"[{request_id}] WarmupAllModels started")

        test_text = (
            request.test_text
            if request.HasField("test_text")
            else "워밍업 테스트 문장입니다."
        )
        repeat_count = request.repeat_count if request.HasField("repeat_count") else 3

        # 모든 모델 워밍업 (instance_name=None)
        warmup_results = await self._model_manager.warmup(
            test_text=test_text,
            repeat_count=repeat_count,
        )

        total_duration_ms = int((time.time() - start_time) * 1000)

        # 결과 변환
        results = [
            WarmupResult(
                model_instance=instance_name,
                success=success,
                message="Warmup completed" if success else "Warmup failed",
                duration_ms=total_duration_ms // len(warmup_results) if warmup_results else 0,
            )
            for instance_name, success in warmup_results.items()
        ]
        overall_success = all(warmup_results.values()) if warmup_results else False

        logger.info(
            f"[{request_id}] WarmupAllModels completed",
            overall_success=overall_success,
            model_count=len(results),
            total_duration_ms=total_duration_ms,
        )

        return WarmupAllResponse(
            request_id=request_id,
            overall_success=overall_success,
            message=(
                "All models warmed up successfully"
                if overall_success
                else "Some models failed to warm up"
            ),
            results=results,
            total_duration_ms=total_duration_ms,
            timestamp_ms=int(time.time() * 1000),
        )
