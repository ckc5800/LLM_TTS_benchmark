# -*- coding: utf-8 -*-
"""gRPC Servicer - TTS 서비스 gRPC 핸들러."""

import time
import uuid
from typing import AsyncIterator

import grpc

from tts_engine.core.config import Settings
from tts_engine.core.exceptions import VoiceNotFoundError
from tts_engine.core.logging import get_logger
from tts_engine.services.model_manager import ModelManager
from tts_engine.services.synthesis_service import SynthesisService
from tts_engine.services.voice_manager import VoiceManager

from tts_engine.proto import (
    AudioChunk,
    BatchStatus,
    BatchSynthesizeRequest,
    BatchSynthesizeResponse,
    BatchStatusRequest,
    BatchStatusResponse,
    ErrorCode,
    ErrorDetail,
    GetPoolStatusRequest,
    GetVoiceInfoRequest,
    GetVoiceInfoResponse,
    GetVoiceMemoryRequest,
    GetVoiceMemoryResponse,
    GetVoicesRequest,
    GetVoicesResponse,
    PoolStatusResponse,
    SynthesizeRequest as ProtoSynthesizeRequest,
    SynthesizeResponse,
    TTSServiceServicer,
    WindowStats,
    # Model Management (Admin)
    GetLoadedModelsRequest,
    GetLoadedModelsResponse,
    LoadedModelInfo,
    ReloadModelRequest,
    ReloadModelResponse,
    ReloadAllModelsRequest,
    ReloadAllModelsResponse,
    ModelReloadResult,
    # Voice Management (Admin)
    AddVoiceRequest,
    AddVoiceResponse,
    UpdateVoiceRequest,
    UpdateVoiceResponse,
    RemoveVoiceRequest,
    RemoveVoiceResponse,
    SetVoiceEnabledRequest,
    SetVoiceEnabledResponse,
    ReloadVoiceRequest,
    ReloadVoiceResponse,
)

from tts_engine.server.proto_converters import (
    convert_to_domain_request,
    convert_to_proto_format,
    convert_to_proto_voice_info,
    map_exception_to_error,
)
from tts_engine.server.health_handlers import HealthHandlersMixin
from tts_engine.server.metrics_server import metrics_registry

logger = get_logger(__name__)


class TTSServicer(HealthHandlersMixin, TTSServiceServicer):
    """TTS gRPC 서비스 핸들러."""

    def __init__(
        self,
        settings: Settings,
        synthesis_service: SynthesisService,
        voice_manager: VoiceManager,
        model_manager: ModelManager,
    ):
        """gRPC Servicer 초기화."""
        self._settings = settings
        self._synthesis_service = synthesis_service
        self._voice_manager = voice_manager
        self._model_manager = model_manager
        self._start_time = time.time()
        logger.info("TTSServicer initialized")

    async def SynthesizeText(
        self,
        request: ProtoSynthesizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> SynthesizeResponse:
        """단일 텍스트 합성."""
        request_id = request.request_id or str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            f"[{request_id}] SynthesizeText started",
            voice_id=request.voice_id,
            text_length=len(request.text),
        )

        try:
            # 1. 요청 변환
            synthesis_request = convert_to_domain_request(request, request_id)

            # 2. 합성 실행
            synth_start = time.time()
            result = await self._synthesis_service.synthesize(synthesis_request)
            synth_time_ms = int((time.time() - synth_start) * 1000)

            logger.debug(
                f"[{request_id}] Synthesis completed",
                synth_time_ms=synth_time_ms,
            )

            # 3. 결과 변환
            processing_time_ms = int((time.time() - start_time) * 1000)

            if result.audio_data is None:
                audio_bytes = b""
                audio_size = 0
            elif isinstance(result.audio_data, bytes):
                audio_bytes = result.audio_data
                audio_size = len(result.audio_data)
            else:
                audio_bytes = result.audio_data.data if result.audio_data else b""
                audio_size = result.audio_data.size_bytes if result.audio_data else 0

            logger.info(
                f"[{request_id}] SynthesizeText completed",
                processing_time_ms=processing_time_ms,
                audio_size=audio_size,
                duration_s=f"{result.duration_seconds:.2f}",
            )

            # Prometheus 메트릭 기록
            if metrics_registry:
                metrics_registry.record_request(
                    method="sync",
                    status="success",
                    processing_time_seconds=processing_time_ms / 1000.0,
                )
                if result.duration_seconds > 0:
                    metrics_registry.record_audio_duration(result.duration_seconds)

            return SynthesizeResponse(
                request_id=request_id,
                success=True,
                audio_data=audio_bytes,
                format=convert_to_proto_format(result.format),
                sample_rate=result.sample_rate,
                duration_seconds=result.duration_seconds,
                data_size=audio_size,
                processing_time_ms=processing_time_ms,
                timestamp_ms=int(time.time() * 1000),
            )

        except Exception as e:
            error_code, error_message = map_exception_to_error(e)
            processing_time_ms = int((time.time() - start_time) * 1000)

            logger.error(
                f"[{request_id}] SynthesizeText failed",
                processing_time_ms=processing_time_ms,
                error=str(e),
                error_code=error_code,
            )

            # Prometheus 메트릭 기록 (실패)
            if metrics_registry:
                metrics_registry.record_request(
                    method="sync",
                    status="failed",
                    processing_time_seconds=processing_time_ms / 1000.0,
                )

            return SynthesizeResponse(
                request_id=request_id,
                success=False,
                processing_time_ms=processing_time_ms,
                timestamp_ms=int(time.time() * 1000),
                error=ErrorDetail(
                    code=error_code,
                    message=error_message,
                    request_id=request_id,
                    timestamp_ms=int(time.time() * 1000),
                ),
            )

    async def SynthesizeTextStreaming(
        self,
        request: ProtoSynthesizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[AudioChunk]:
        """스트리밍 텍스트 합성."""
        request_id = request.request_id or str(uuid.uuid4())
        start_time = time.time()
        chunk_index = 0

        logger.info(
            f"[{request_id}] SynthesizeTextStreaming started",
            voice_id=request.voice_id,
            text_length=len(request.text),
        )

        try:
            synthesis_request = convert_to_domain_request(request, request_id)

            async for chunk_data in self._synthesis_service.synthesize_streaming(
                synthesis_request
            ):
                yield AudioChunk(
                    request_id=request_id,
                    chunk_data=chunk_data,
                    chunk_index=chunk_index,
                    is_last=False,
                    chunk_size=len(chunk_data),
                    timestamp_ms=int(time.time() * 1000),
                )
                chunk_index += 1

            yield AudioChunk(
                request_id=request_id,
                chunk_data=b"",
                chunk_index=chunk_index,
                is_last=True,
                timestamp_ms=int(time.time() * 1000),
            )

            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[{request_id}] SynthesizeTextStreaming completed",
                processing_time_ms=processing_time_ms,
                total_chunks=chunk_index + 1,
            )

            # Prometheus 메트릭 기록 (스트리밍 성공)
            if metrics_registry:
                metrics_registry.record_request(
                    method="streaming",
                    status="success",
                    processing_time_seconds=processing_time_ms / 1000.0,
                )

        except Exception as e:
            error_code, error_message = map_exception_to_error(e)
            processing_time_ms = int((time.time() - start_time) * 1000)

            logger.error(
                f"[{request_id}] SynthesizeTextStreaming failed",
                processing_time_ms=processing_time_ms,
                error=str(e),
                error_code=error_code,
            )

            # Prometheus 메트릭 기록 (스트리밍 실패)
            if metrics_registry:
                metrics_registry.record_request(
                    method="streaming",
                    status="failed",
                    processing_time_seconds=processing_time_ms / 1000.0,
                )

            # 스트리밍에서는 에러 청크를 전송 후 종료
            yield AudioChunk(
                request_id=request_id,
                chunk_data=b"",
                chunk_index=chunk_index,
                is_last=True,
                has_error=True,
                error_code=error_code,
                error_message=error_message,
                timestamp_ms=int(time.time() * 1000),
            )

    async def GetVoices(
        self,
        request: GetVoicesRequest,
        context: grpc.aio.ServicerContext,
    ) -> GetVoicesResponse:
        """음성 목록 조회."""
        request_id = request.request_id or str(uuid.uuid4())
        start_time = time.time()

        logger.debug(
            f"[{request_id}] GetVoices started",
            language=request.language if request.HasField("language") else None,
            gender=request.gender if request.HasField("gender") else None,
        )

        try:
            language = request.language if request.HasField("language") else None
            gender = request.gender if request.HasField("gender") else None
            enabled_only = (
                request.enabled_only if request.HasField("enabled_only") else True
            )

            voices = self._voice_manager.list_voices(
                language=language,
                gender=gender,
                enabled_only=enabled_only,
            )

            voice_infos = [convert_to_proto_voice_info(v) for v in voices]
            default_voice_id = self._voice_manager.default_voice_id or ""

            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.debug(
                f"[{request_id}] GetVoices completed",
                processing_time_ms=processing_time_ms,
                count=len(voice_infos),
            )

            return GetVoicesResponse(
                request_id=request_id,
                voices=voice_infos,
                total_count=len(voice_infos),
                default_voice_id=default_voice_id,
                timestamp_ms=int(time.time() * 1000),
            )

        except Exception as e:
            error_code, error_message = map_exception_to_error(e)
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] GetVoices failed",
                processing_time_ms=processing_time_ms,
                error=str(e),
            )
            return GetVoicesResponse(
                request_id=request_id,
                total_count=0,
                timestamp_ms=int(time.time() * 1000),
                error=ErrorDetail(
                    code=error_code,
                    message=error_message,
                    request_id=request_id,
                    timestamp_ms=int(time.time() * 1000),
                ),
            )

    async def GetVoiceInfo(
        self,
        request: GetVoiceInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> GetVoiceInfoResponse:
        """특정 음성 정보 조회."""
        request_id = request.request_id or str(uuid.uuid4())
        start_time = time.time()

        logger.debug(
            f"[{request_id}] GetVoiceInfo started",
            voice_id=request.voice_id,
        )

        try:
            voice = self._voice_manager.get_voice(request.voice_id)
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.debug(
                f"[{request_id}] GetVoiceInfo completed",
                processing_time_ms=processing_time_ms,
                found=True,
            )
            return GetVoiceInfoResponse(
                request_id=request_id,
                found=True,
                voice=convert_to_proto_voice_info(voice),
                timestamp_ms=int(time.time() * 1000),
            )
        except VoiceNotFoundError:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.debug(
                f"[{request_id}] GetVoiceInfo completed",
                processing_time_ms=processing_time_ms,
                found=False,
            )
            return GetVoiceInfoResponse(
                request_id=request_id,
                found=False,
                timestamp_ms=int(time.time() * 1000),
                error=ErrorDetail(
                    code=ErrorCode.ERROR_VOICE_NOT_FOUND,
                    message=f"Voice '{request.voice_id}' not found",
                    request_id=request_id,
                    timestamp_ms=int(time.time() * 1000),
                ),
            )
        except Exception as e:
            error_code, error_message = map_exception_to_error(e)
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] GetVoiceInfo failed",
                processing_time_ms=processing_time_ms,
                error=str(e),
            )
            return GetVoiceInfoResponse(
                request_id=request_id,
                found=False,
                timestamp_ms=int(time.time() * 1000),
                error=ErrorDetail(
                    code=error_code,
                    message=error_message,
                    request_id=request_id,
                    timestamp_ms=int(time.time() * 1000),
                ),
            )

    async def GetVoiceMemory(
        self,
        request: GetVoiceMemoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> GetVoiceMemoryResponse:
        """음성별 메모리 사용량 조회 (관리/모니터링용)."""
        from tts_engine.proto import MemoryBreakdown

        request_id = request.request_id or str(uuid.uuid4())

        logger.debug(
            "GetVoiceMemory started",
            request_id=request_id,
            voice_id=request.voice_id,
        )

        try:
            voice = self._voice_manager.get_voice(request.voice_id)

            # 프롬프트 로드되지 않은 경우
            if not voice.is_prompt_loaded or not voice.prompt_data:
                return GetVoiceMemoryResponse(
                    request_id=request_id,
                    voice_id=request.voice_id,
                    prompt_memory_mb=0,
                    model_shared_memory_mb=0,
                    total_memory_mb=0,
                    timestamp_ms=int(time.time() * 1000),
                )

            # 메모리 계산
            prompt_data = voice.prompt_data
            wav_tensor_mb = 0
            features_mb = 0
            tokens_mb = 0
            text_mb = 0

            try:
                if prompt_data.wav_tensor is not None:
                    wav_bytes = prompt_data.wav_tensor.element_size() * prompt_data.wav_tensor.numel()
                    wav_tensor_mb = int(wav_bytes / (1024 * 1024))

                if prompt_data.features is not None:
                    features_bytes = prompt_data.features.element_size() * prompt_data.features.numel()
                    features_mb = int(features_bytes / (1024 * 1024))

                if prompt_data.tokens is not None:
                    tokens_bytes = len(prompt_data.tokens) * 28
                    tokens_mb = int(tokens_bytes / (1024 * 1024))

                if prompt_data.text:
                    text_bytes = len(prompt_data.text.encode('utf-8'))
                    text_mb = int(text_bytes / (1024 * 1024))

            except Exception as calc_error:
                logger.warning(
                    "Failed to calculate memory breakdown",
                    request_id=request_id,
                    voice_id=request.voice_id,
                    error=str(calc_error),
                )

            prompt_memory_mb = max(1, wav_tensor_mb + features_mb + tokens_mb + text_mb)

            # 모델 공유 메모리 추정 (전체 모델 GPU 메모리 / 음성 수)
            model_shared_mb = 0
            try:
                import torch
                if torch.cuda.is_available():
                    # 모델 인스턴스의 GPU 메모리 조회
                    model_instance = voice.model_instance
                    pool = self._model_manager.get_pool(model_instance)
                    if pool and hasattr(pool, '_model'):
                        device = getattr(pool._model, '_device_torch', None)
                        if device and device.type == 'cuda':
                            allocated_mb = int(torch.cuda.memory_allocated(device) / (1024 * 1024))
                            # 전체 음성 수로 나누기
                            voice_count = len(self._voice_manager.list_voices())
                            model_shared_mb = allocated_mb // max(1, voice_count)
            except Exception as model_error:
                logger.debug(
                    "Failed to calculate shared model memory",
                    request_id=request_id,
                    error=str(model_error),
                )

            total_memory_mb = prompt_memory_mb + model_shared_mb

            logger.debug(
                "GetVoiceMemory completed",
                request_id=request_id,
                voice_id=request.voice_id,
                prompt_memory_mb=prompt_memory_mb,
                model_shared_mb=model_shared_mb,
                total_memory_mb=total_memory_mb,
            )

            return GetVoiceMemoryResponse(
                request_id=request_id,
                voice_id=request.voice_id,
                prompt_memory_mb=prompt_memory_mb,
                model_shared_memory_mb=model_shared_mb,
                total_memory_mb=total_memory_mb,
                memory_breakdown=MemoryBreakdown(
                    wav_tensor_mb=wav_tensor_mb,
                    features_mb=features_mb,
                    tokens_mb=tokens_mb,
                    text_mb=text_mb,
                    other_mb=0,
                ),
                timestamp_ms=int(time.time() * 1000),
            )

        except VoiceNotFoundError:
            logger.warning(
                "Voice not found for memory query",
                request_id=request_id,
                voice_id=request.voice_id,
            )
            return GetVoiceMemoryResponse(
                request_id=request_id,
                voice_id=request.voice_id,
                prompt_memory_mb=0,
                model_shared_memory_mb=0,
                total_memory_mb=0,
                timestamp_ms=int(time.time() * 1000),
            )
        except Exception as e:
            logger.error(
                "GetVoiceMemory failed",
                request_id=request_id,
                voice_id=request.voice_id,
                error=str(e),
            )
            return GetVoiceMemoryResponse(
                request_id=request_id,
                voice_id=request.voice_id,
                prompt_memory_mb=0,
                model_shared_memory_mb=0,
                total_memory_mb=0,
                timestamp_ms=int(time.time() * 1000),
            )

    async def SynthesizeBatch(
        self,
        request: BatchSynthesizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> BatchSynthesizeResponse:
        """배치 합성 요청."""
        request_id = request.request_id or str(uuid.uuid4())
        batch_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            f"[{request_id}] SynthesizeBatch started",
            batch_id=batch_id,
            total_count=len(request.requests),
        )

        try:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[{request_id}] SynthesizeBatch queued",
                batch_id=batch_id,
                processing_time_ms=processing_time_ms,
            )

            return BatchSynthesizeResponse(
                request_id=request_id,
                batch_id=batch_id,
                status=BatchStatus.BATCH_STATUS_QUEUED,
                message=f"Batch queued with {len(request.requests)} requests",
                total_count=len(request.requests),
                timestamp_ms=int(time.time() * 1000),
            )

        except Exception as e:
            error_code, error_message = map_exception_to_error(e)
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] SynthesizeBatch failed",
                batch_id=batch_id,
                processing_time_ms=processing_time_ms,
                error=str(e),
            )
            return BatchSynthesizeResponse(
                request_id=request_id,
                batch_id=batch_id,
                status=BatchStatus.BATCH_STATUS_FAILED,
                message=error_message,
                total_count=0,
                timestamp_ms=int(time.time() * 1000),
            )

    async def GetBatchStatus(
        self,
        request: BatchStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> BatchStatusResponse:
        """배치 상태 조회."""
        request_id = request.request_id or str(uuid.uuid4())
        start_time = time.time()

        logger.debug(
            f"[{request_id}] GetBatchStatus started",
            batch_id=request.batch_id,
        )

        try:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.debug(
                f"[{request_id}] GetBatchStatus completed",
                batch_id=request.batch_id,
                processing_time_ms=processing_time_ms,
            )

            return BatchStatusResponse(
                request_id=request_id,
                batch_id=request.batch_id,
                status=BatchStatus.BATCH_STATUS_UNKNOWN,
                message="Batch not found",
                timestamp_ms=int(time.time() * 1000),
            )

        except Exception as e:
            error_code, error_message = map_exception_to_error(e)
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] GetBatchStatus failed",
                batch_id=request.batch_id,
                processing_time_ms=processing_time_ms,
                error=str(e),
            )
            return BatchStatusResponse(
                request_id=request_id,
                batch_id=request.batch_id,
                status=BatchStatus.BATCH_STATUS_FAILED,
                message=error_message,
                timestamp_ms=int(time.time() * 1000),
            )

    async def GetPoolStatus(
        self,
        request: GetPoolStatusRequest,
        context: grpc.aio.ServicerContext,
    ) -> PoolStatusResponse:
        """런타임 풀 상태 조회."""
        request_id = request.request_id or str(uuid.uuid4())

        logger.debug(f"[{request_id}] GetPoolStatus started")

        try:
            pool_status = self._synthesis_service.get_pool_status()
            synthesis_stats = pool_status.get("synthesis_stats", {})
            batch_stats = pool_status.get("batch_stats", {})
            window_stats = pool_status.get("window_stats", {})

            # 실행기 통계 가져오기
            executor_stats = {}
            if self._synthesis_service._runtime_executor:
                executor_stats = self._synthesis_service._runtime_executor.get_stats()

            # 시간 윈도우 통계 변환
            def to_window_stats(stats: dict) -> WindowStats:
                return WindowStats(
                    total=stats.get("total", 0),
                    success=stats.get("success", 0),
                    failed=stats.get("failed", 0),
                    success_rate=stats.get("success_rate", 0.0),
                    avg_time_ms=stats.get("avg_time_ms", 0.0),
                    min_time_ms=stats.get("min_time_ms", 0.0),
                    max_time_ms=stats.get("max_time_ms", 0.0),
                    requests_per_second=stats.get("requests_per_second", 0.0),
                )

            return PoolStatusResponse(
                request_id=request_id,
                runtime=pool_status.get("runtime", "unknown"),
                total_slots=pool_status.get("total_slots", 0),
                active_count=pool_status.get("active", 0),
                waiting_count=pool_status.get("waiting", 0),
                available_slots=pool_status.get("available", 0),
                thread_pool_size=pool_status.get("thread_pool_size", 0),
                workers_alive=pool_status.get("workers_alive", 0),
                submitted_count=executor_stats.get("submitted_count", 0),
                completed_count=executor_stats.get("completed_count", 0),
                failed_count=executor_stats.get("failed_count", 0),
                synthesis_active=synthesis_stats.get("active_requests", 0),
                synthesis_total=synthesis_stats.get("total_requests", 0),
                synthesis_success=synthesis_stats.get("successful_requests", 0),
                synthesis_failed=synthesis_stats.get("failed_requests", 0),
                uptime_seconds=synthesis_stats.get("uptime_seconds", 0.0),
                overall_rps=synthesis_stats.get("requests_per_second", 0.0),
                stats_1min=to_window_stats(window_stats.get("1min", {})),
                stats_5min=to_window_stats(window_stats.get("5min", {})),
                stats_1hour=to_window_stats(window_stats.get("1hour", {})),
                batch_enabled=batch_stats.get("enabled", False),
                batch_submitted=batch_stats.get("total_submitted", 0),
                batch_processed=batch_stats.get("total_processed", 0),
                # 큐 상태
                queue_size=pool_status.get("queue_size", 0),
                max_queue_size=pool_status.get("max_queue_size", 0),
                queue_utilization=pool_status.get("queue_utilization", 0.0),
                queue_rejected=pool_status.get("queue_rejected", 0),
                queue_timeout=pool_status.get("queue_timeout", 0),
                prometheus_enabled=pool_status.get("prometheus_enabled", False),
                timestamp_ms=int(time.time() * 1000),
            )

        except Exception as e:
            logger.error(f"[{request_id}] GetPoolStatus failed: {e}")
            return PoolStatusResponse(
                request_id=request_id,
                runtime="error",
                timestamp_ms=int(time.time() * 1000),
            )

    # ==================== Model Management (Admin) ====================

    async def GetLoadedModels(
        self,
        request: GetLoadedModelsRequest,
        context: grpc.aio.ServicerContext,
    ) -> GetLoadedModelsResponse:
        """로드된 모델 목록 조회."""
        request_id = request.request_id or str(uuid.uuid4())

        logger.debug(f"[{request_id}] GetLoadedModels started")

        try:
            models = []
            model_instances = self._model_manager.list_model_instances()

            for instance_name in model_instances:
                try:
                    pool = self._model_manager.get_pool(instance_name)
                    if pool is None:
                        continue

                    # 모델 정보 수집
                    model_info = self._model_manager.get_model_info(instance_name)
                    voices = self._voice_manager.list_voices_by_model(instance_name)

                    models.append(LoadedModelInfo(
                        model_instance=instance_name,
                        model_type=model_info.get("model_type", "unknown"),
                        model_path=model_info.get("model_path", ""),
                        status="ready" if pool else "error",
                        loaded_at_ms=int(model_info.get("loaded_at", 0) * 1000),
                        memory_usage_mb=model_info.get("memory_usage_mb", 0),
                        voice_count=len(voices),
                        loaded_voices=[v.voice_id for v in voices],
                        supports_speed=model_info.get("supports_speed", False),
                        supports_pitch=model_info.get("supports_pitch", False),
                        supports_volume=model_info.get("supports_volume", True),
                    ))
                except Exception as model_error:
                    logger.warning(
                        f"Failed to get model info for {instance_name}",
                        error=str(model_error),
                    )

            logger.debug(
                f"[{request_id}] GetLoadedModels completed",
                model_count=len(models),
            )

            return GetLoadedModelsResponse(
                request_id=request_id,
                models=models,
                timestamp_ms=int(time.time() * 1000),
            )

        except Exception as e:
            logger.error(f"[{request_id}] GetLoadedModels failed: {e}")
            return GetLoadedModelsResponse(
                request_id=request_id,
                models=[],
                timestamp_ms=int(time.time() * 1000),
            )

    async def ReloadModel(
        self,
        request: ReloadModelRequest,
        context: grpc.aio.ServicerContext,
    ) -> ReloadModelResponse:
        """특정 모델 리로드."""
        request_id = request.request_id or str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            f"[{request_id}] ReloadModel started",
            model_instance=request.model_instance,
            reload_voices=request.reload_voices,
        )

        try:
            await self._model_manager.reload_model(
                request.model_instance,
                reload_voices=request.reload_voices,
            )

            reload_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"[{request_id}] ReloadModel completed",
                model_instance=request.model_instance,
                reload_time_ms=reload_time_ms,
            )

            return ReloadModelResponse(
                request_id=request_id,
                success=True,
                message=f"Model '{request.model_instance}' reloaded successfully",
                reload_time_ms=reload_time_ms,
            )

        except Exception as e:
            reload_time_ms = int((time.time() - start_time) * 1000)
            error_code, error_message = map_exception_to_error(e)

            logger.error(
                f"[{request_id}] ReloadModel failed",
                model_instance=request.model_instance,
                error=str(e),
            )

            return ReloadModelResponse(
                request_id=request_id,
                success=False,
                message=str(e),
                reload_time_ms=reload_time_ms,
                error=ErrorDetail(
                    code=error_code,
                    message=error_message,
                    request_id=request_id,
                    timestamp_ms=int(time.time() * 1000),
                ),
            )

    async def ReloadAllModels(
        self,
        request: ReloadAllModelsRequest,
        context: grpc.aio.ServicerContext,
    ) -> ReloadAllModelsResponse:
        """전체 모델 리로드."""
        request_id = request.request_id or str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            f"[{request_id}] ReloadAllModels started",
            reload_voices=request.reload_voices,
        )

        results = []
        overall_success = True

        try:
            model_instances = self._model_manager.list_model_instances()

            for instance_name in model_instances:
                model_start = time.time()
                try:
                    await self._model_manager.reload_model(
                        instance_name,
                        reload_voices=request.reload_voices,
                    )
                    model_time_ms = int((time.time() - model_start) * 1000)

                    results.append(ModelReloadResult(
                        model_instance=instance_name,
                        success=True,
                        message="Reloaded successfully",
                        reload_time_ms=model_time_ms,
                    ))

                except Exception as model_error:
                    model_time_ms = int((time.time() - model_start) * 1000)
                    overall_success = False

                    results.append(ModelReloadResult(
                        model_instance=instance_name,
                        success=False,
                        message=str(model_error),
                        reload_time_ms=model_time_ms,
                    ))

            total_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"[{request_id}] ReloadAllModels completed",
                overall_success=overall_success,
                total_time_ms=total_time_ms,
            )

            return ReloadAllModelsResponse(
                request_id=request_id,
                overall_success=overall_success,
                message="All models reloaded" if overall_success else "Some models failed to reload",
                results=results,
                total_time_ms=total_time_ms,
            )

        except Exception as e:
            total_time_ms = int((time.time() - start_time) * 1000)

            logger.error(f"[{request_id}] ReloadAllModels failed: {e}")

            return ReloadAllModelsResponse(
                request_id=request_id,
                overall_success=False,
                message=str(e),
                results=results,
                total_time_ms=total_time_ms,
            )

    # ==================== Voice Management (Admin) ====================

    async def AddVoice(
        self,
        request: AddVoiceRequest,
        context: grpc.aio.ServicerContext,
    ) -> AddVoiceResponse:
        """새 음성 추가."""
        request_id = request.request_id or str(uuid.uuid4())

        logger.info(
            f"[{request_id}] AddVoice started",
            voice_id=request.voice_id,
            model_instance=request.model_instance,
        )

        try:
            # 메타데이터 변환
            metadata = {}
            if request.metadata:
                metadata = {
                    "name": request.metadata.name,
                    "language": request.metadata.language,
                    "gender": request.metadata.gender,
                    "description": request.metadata.description,
                }
                if request.metadata.extra:
                    metadata.update(request.metadata.extra)

            await self._voice_manager.add_voice(
                voice_id=request.voice_id,
                model_instance=request.model_instance,
                prompt_audio_path=request.prompt_audio_path,
                prompt_text=request.prompt_text,
                metadata=metadata,
            )

            logger.info(
                f"[{request_id}] AddVoice completed",
                voice_id=request.voice_id,
            )

            return AddVoiceResponse(
                request_id=request_id,
                success=True,
                message=f"Voice '{request.voice_id}' added successfully",
                voice_id=request.voice_id,
            )

        except Exception as e:
            error_code, error_message = map_exception_to_error(e)

            logger.error(
                f"[{request_id}] AddVoice failed",
                voice_id=request.voice_id,
                error=str(e),
            )

            return AddVoiceResponse(
                request_id=request_id,
                success=False,
                message=str(e),
                voice_id=request.voice_id,
                error=ErrorDetail(
                    code=error_code,
                    message=error_message,
                    request_id=request_id,
                    timestamp_ms=int(time.time() * 1000),
                ),
            )

    async def UpdateVoice(
        self,
        request: UpdateVoiceRequest,
        context: grpc.aio.ServicerContext,
    ) -> UpdateVoiceResponse:
        """음성 메타데이터 및 프롬프트 수정."""
        request_id = request.request_id or str(uuid.uuid4())

        logger.info(
            f"[{request_id}] UpdateVoice started",
            voice_id=request.voice_id,
        )

        try:
            # 메타데이터 변환
            metadata = {}
            if request.metadata:
                if request.metadata.name:
                    metadata["name"] = request.metadata.name
                if request.metadata.language:
                    metadata["language"] = request.metadata.language
                if request.metadata.gender:
                    metadata["gender"] = request.metadata.gender
                if request.metadata.description:
                    metadata["description"] = request.metadata.description
                if request.metadata.extra:
                    metadata.update(request.metadata.extra)

            # 프롬프트 파라미터
            prompt_audio_path = request.prompt_audio_path if request.prompt_audio_path else None
            prompt_text = request.prompt_text if request.prompt_text else None

            prompt_changed = await self._voice_manager.update_voice(
                voice_id=request.voice_id,
                metadata=metadata,
                prompt_audio_path=prompt_audio_path,
                prompt_text=prompt_text,
            )

            message = f"Voice '{request.voice_id}' updated successfully"
            if prompt_changed:
                message += " (prompt will be reloaded on next synthesis)"

            logger.info(
                f"[{request_id}] UpdateVoice completed",
                voice_id=request.voice_id,
                prompt_changed=prompt_changed,
            )

            return UpdateVoiceResponse(
                request_id=request_id,
                success=True,
                message=message,
            )

        except Exception as e:
            error_code, error_message = map_exception_to_error(e)

            logger.error(
                f"[{request_id}] UpdateVoice failed",
                voice_id=request.voice_id,
                error=str(e),
            )

            return UpdateVoiceResponse(
                request_id=request_id,
                success=False,
                message=str(e),
                error=ErrorDetail(
                    code=error_code,
                    message=error_message,
                    request_id=request_id,
                    timestamp_ms=int(time.time() * 1000),
                ),
            )

    async def RemoveVoice(
        self,
        request: RemoveVoiceRequest,
        context: grpc.aio.ServicerContext,
    ) -> RemoveVoiceResponse:
        """음성 제거."""
        request_id = request.request_id or str(uuid.uuid4())

        logger.info(
            f"[{request_id}] RemoveVoice started",
            voice_id=request.voice_id,
        )

        try:
            await self._voice_manager.remove_voice(request.voice_id)

            logger.info(
                f"[{request_id}] RemoveVoice completed",
                voice_id=request.voice_id,
            )

            return RemoveVoiceResponse(
                request_id=request_id,
                success=True,
                message=f"Voice '{request.voice_id}' removed successfully",
            )

        except Exception as e:
            error_code, error_message = map_exception_to_error(e)

            logger.error(
                f"[{request_id}] RemoveVoice failed",
                voice_id=request.voice_id,
                error=str(e),
            )

            return RemoveVoiceResponse(
                request_id=request_id,
                success=False,
                message=str(e),
                error=ErrorDetail(
                    code=error_code,
                    message=error_message,
                    request_id=request_id,
                    timestamp_ms=int(time.time() * 1000),
                ),
            )

    async def SetVoiceEnabled(
        self,
        request: SetVoiceEnabledRequest,
        context: grpc.aio.ServicerContext,
    ) -> SetVoiceEnabledResponse:
        """음성 활성화/비활성화."""
        request_id = request.request_id or str(uuid.uuid4())

        logger.info(
            f"[{request_id}] SetVoiceEnabled started",
            voice_id=request.voice_id,
            enabled=request.enabled,
        )

        try:
            await self._voice_manager.set_voice_enabled(
                request.voice_id,
                request.enabled,
            )

            status = "enabled" if request.enabled else "disabled"
            logger.info(
                f"[{request_id}] SetVoiceEnabled completed",
                voice_id=request.voice_id,
                enabled=request.enabled,
            )

            return SetVoiceEnabledResponse(
                request_id=request_id,
                success=True,
                message=f"Voice '{request.voice_id}' {status} successfully",
            )

        except Exception as e:
            error_code, error_message = map_exception_to_error(e)

            logger.error(
                f"[{request_id}] SetVoiceEnabled failed",
                voice_id=request.voice_id,
                error=str(e),
            )

            return SetVoiceEnabledResponse(
                request_id=request_id,
                success=False,
                message=str(e),
                error=ErrorDetail(
                    code=error_code,
                    message=error_message,
                    request_id=request_id,
                    timestamp_ms=int(time.time() * 1000),
                ),
            )

    async def ReloadVoice(
        self,
        request: ReloadVoiceRequest,
        context: grpc.aio.ServicerContext,
    ) -> ReloadVoiceResponse:
        """음성 리로드 (프롬프트 재로드)."""
        request_id = request.request_id or str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            f"[{request_id}] ReloadVoice started",
            voice_id=request.voice_id,
        )

        try:
            await self._voice_manager.reload_voice(request.voice_id)

            reload_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"[{request_id}] ReloadVoice completed",
                voice_id=request.voice_id,
                reload_time_ms=reload_time_ms,
            )

            return ReloadVoiceResponse(
                request_id=request_id,
                success=True,
                message=f"Voice '{request.voice_id}' reloaded successfully",
                reload_time_ms=reload_time_ms,
            )

        except Exception as e:
            reload_time_ms = int((time.time() - start_time) * 1000)
            error_code, error_message = map_exception_to_error(e)

            logger.error(
                f"[{request_id}] ReloadVoice failed",
                voice_id=request.voice_id,
                error=str(e),
            )

            return ReloadVoiceResponse(
                request_id=request_id,
                success=False,
                message=str(e),
                reload_time_ms=reload_time_ms,
                error=ErrorDetail(
                    code=error_code,
                    message=error_message,
                    request_id=request_id,
                    timestamp_ms=int(time.time() * 1000),
                ),
            )
