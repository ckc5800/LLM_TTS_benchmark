# -*- coding: utf-8 -*-
"""합성 서비스 - TTS 합성 오케스트레이션."""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Dict, List, Optional

from tts_engine.core.config import Settings
from tts_engine.core.constants import AudioFormat, RuntimeType
from tts_engine.core.exceptions import (
    EmptyTextError,
    PoolExhaustedError,
    SynthesisError,
    SynthesisTimeoutError,
    TextTooLongError,
    VoiceNotFoundError,
    VoiceNotReadyError,
)
from tts_engine.core.logging import get_logger, log_synthesis
from tts_engine.domain.synthesis import (
    SynthesisRequest,
    SynthesisResult,
    SynthesisStatus,
    BatchSynthesisRequest,
    BatchSynthesisResult,
)
from tts_engine.services.model_manager import ModelManager
from tts_engine.services.voice_manager import VoiceManager
from tts_engine.services.dynamic_batcher import DynamicBatcher
from tts_engine.services.execution import RuntimeExecutor, RuntimeExecutorFactory
from tts_engine.services.batch_synthesis_executor import BatchSynthesisExecutor
from tts_engine.services.synthesis_executor import SynthesisExecutor
from tts_engine.services.synthesis_stats import SynthesisStats
from tts_engine.services.file_storage import FileStorageService
from tts_engine.utils.audio import convert_format, wav_bytes_to_numpy

logger = get_logger(__name__)


class SynthesisService:
    """TTS 합성 서비스 - 워크플로우 오케스트레이션."""

    def __init__(
        self,
        settings: Settings,
        model_manager: ModelManager,
        voice_manager: VoiceManager,
    ):
        """합성 서비스 초기화."""
        self._settings = settings
        self._model_manager = model_manager
        self._voice_manager = voice_manager
        self._synthesis_config = settings.synthesis
        self._stats = SynthesisStats()

        # 파일 저장 서비스
        self._file_storage = FileStorageService(settings.synthesis.file_storage)

        # DynamicBatcher 설정 (고가용성)
        self._dynamic_batcher: Optional[DynamicBatcher] = None
        self._batch_mode_enabled = settings.performance.auto_batch.enabled

        # 런타임별 실행기
        # PyTorch: 멀티프로세스 (GPU 메모리 독립)
        # ONNX/TensorRT: 스레드 풀 (기존 방식)
        self._runtime_executor: Optional[RuntimeExecutor] = None
        self._executor: Optional[ThreadPoolExecutor] = None  # 레거시 호환

        # 런타임 타입 결정 (첫 번째 enabled 인스턴스 기준)
        self._runtime_type: Optional[RuntimeType] = None
        self._model_config: Optional[dict] = None

        # 단일 합성 실행기
        self._synthesis_executor = SynthesisExecutor(
            settings=settings,
            model_manager=model_manager,
            voice_manager=voice_manager,
        )

        # 배치 합성 실행기
        self._batch_executor = BatchSynthesisExecutor(
            settings=settings,
            model_manager=model_manager,
            voice_manager=voice_manager,
        )

        if self._batch_mode_enabled:
            self._dynamic_batcher = DynamicBatcher(
                max_batch_size=settings.performance.auto_batch.max_batch_size,
                max_wait_ms=settings.performance.auto_batch.timeout_ms,
                group_by_voice=settings.performance.auto_batch.group_by_voice,
            )
            self._dynamic_batcher.set_batch_handler(self._batch_executor.handle_batch)

            # 스레드 풀 (ONNX, TensorRT용 또는 레거시)
            max_workers = settings.performance.execution.thread_pool_size
            self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tts_batch_")

        logger.info(
            "SynthesisService initialized",
            max_text_length=self._synthesis_config.max_text_length,
            default_sample_rate=self._synthesis_config.default_sample_rate,
            batch_mode=self._batch_mode_enabled,
        )

    async def initialize(self) -> None:
        """비동기 초기화 - 런타임별 실행기를 미리 초기화합니다."""
        # 런타임 타입 결정
        self._determine_runtime_type()

        # ONNX/TensorRT 런타임인 경우 실행기 미리 초기화
        if self._runtime_type in (RuntimeType.ONNX, RuntimeType.TENSORRT):
            logger.info(
                "Initializing runtime executor",
                runtime_type=self._runtime_type.value,
            )
            await self._initialize_runtime_executor()

        # 파일 저장 서비스 시작
        await self._file_storage.start()

        logger.info(
            "SynthesisService async initialization complete",
            runtime_type=self._runtime_type.value if self._runtime_type else "unknown",
            file_storage_enabled=self._file_storage.enabled,
        )

    def _determine_runtime_type(self) -> None:
        """설정에서 런타임 타입을 결정합니다."""
        for instance_name, config in self._settings.model_instances.items():
            if not config.enabled:
                continue

            if config.options and hasattr(config.options, "runtime"):
                runtime_str = config.options.runtime
                if runtime_str:
                    try:
                        self._runtime_type = RuntimeType(runtime_str.lower())
                        # 모델 설정 저장 (나중에 합성에서 사용)
                        self._model_config = {
                            "pool_size": config.pool_size,
                            "device": config.device,
                            "model_path": config.model_path,
                            # ONNX 설정
                            "onnx_model_dir": getattr(config.options, "onnx_model_dir", None),
                            "max_concurrent_gpu": getattr(config.options, "max_concurrent_gpu", None) or config.pool_size,
                            # TensorRT 설정
                            "tensorrt_model_dir": getattr(config.options, "tensorrt_model_dir", None),
                            "engine_file": getattr(config.options, "engine_file", None),
                            "trt_concurrent": getattr(config.options, "trt_concurrent", 2),
                            # 공통 설정
                            "vocoder_path": getattr(config.options, "vocoder_path", None),
                            "num_steps": getattr(config.options, "num_steps", 16),
                            "t_shift": getattr(config.options, "t_shift", 0.5),
                            "guidance_scale": getattr(config.options, "guidance_scale", 1.0),
                        }
                        logger.info(
                            "Runtime type determined",
                            instance_name=instance_name,
                            runtime_type=self._runtime_type.value,
                            trt_concurrent=self._model_config.get("trt_concurrent"),
                        )
                        return
                    except ValueError:
                        logger.warning(f"Unknown runtime: {runtime_str}")

        # 기본값
        self._runtime_type = RuntimeType.PYTORCH
        logger.info("Using default runtime type: pytorch")

    async def synthesize(
        self,
        request: SynthesisRequest,
        timeout: Optional[float] = None,
    ) -> SynthesisResult:
        """텍스트를 음성으로 합성합니다."""
        start_time = time.time()
        request_id = request.request_id or str(uuid.uuid4())
        timeout = timeout or self._settings.performance.execution.task_timeout_seconds

        logger.debug(
            f"[{request_id}] Synthesis request received",
            voice_id=request.voice_id,
            text_length=len(request.text),
        )

        await self._stats.start_request(request_id, start_time)

        try:
            # 1. 요청 검증
            validate_start = time.time()
            self._validate_request(request)
            logger.debug(
                f"[{request_id}] Request validated",
                elapsed_ms=int((time.time() - validate_start) * 1000),
            )

            # 2. 음성 데이터 조회
            voice_start = time.time()
            voice = self._voice_manager.get_voice_or_fallback(request.voice_id)
            logger.debug(
                f"[{request_id}] Voice resolved",
                voice_id=voice.voice_id,
                model_instance=voice.model_instance,
                elapsed_ms=int((time.time() - voice_start) * 1000),
            )

            if not voice.enabled:
                raise VoiceNotReadyError(f"Voice '{request.voice_id}' is disabled")

            # 음성에 연결된 모델 인스턴스 이름 조회
            instance_name = voice.model_instance

            # 3. 내부 요청 생성
            internal_request = SynthesisRequest.create(
                text=request.text,
                voice_id=request.voice_id,
                request_id=request_id,
                speed=request.speed,
                pitch=request.pitch,
                volume=request.volume,
                format=request.format or self._synthesis_config.default_format,
                sample_rate=request.sample_rate or voice.sample_rate,
            )

            # 4. 합성 실행
            synth_start = time.time()
            if self._batch_mode_enabled and self._dynamic_batcher and self._dynamic_batcher._is_running:
                try:
                    result = await asyncio.wait_for(
                        self._dynamic_batcher.submit(internal_request),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    raise SynthesisTimeoutError(
                        f"Synthesis timed out after {timeout}s",
                        request_id=request_id,
                    )
            else:
                try:
                    result = await asyncio.wait_for(
                        self._execute_synthesis(internal_request, instance_name),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    raise SynthesisTimeoutError(
                        f"Synthesis timed out after {timeout}s",
                        request_id=request_id,
                    )

            synth_elapsed_ms = int((time.time() - synth_start) * 1000)
            logger.debug(
                f"[{request_id}] Model synthesis completed",
                elapsed_ms=synth_elapsed_ms,
            )

            # 5. 포맷 변환 (요청 포맷이 WAV가 아닌 경우)
            requested_format = internal_request.format
            if requested_format and requested_format != AudioFormat.WAV and result.audio_data:
                convert_start = time.time()
                audio_bytes = (
                    result.audio_data.data
                    if hasattr(result.audio_data, 'data')
                    else result.audio_data
                )
                try:
                    # WAV bytes -> numpy -> target format
                    audio_np, wav_sr = wav_bytes_to_numpy(audio_bytes)
                    converted_bytes = convert_format(
                        audio_data=audio_np,
                        sample_rate=wav_sr,
                        target_format=requested_format,
                    )
                    # 결과 업데이트
                    result = SynthesisResult(
                        request_id=result.request_id,
                        status=result.status,
                        audio_data=converted_bytes,
                        file_path=result.file_path,
                        duration_seconds=result.duration_seconds,
                        sample_rate=wav_sr,
                        format=requested_format,
                        processing_time_ms=result.processing_time_ms,
                        error_message=result.error_message,
                        metadata=result.metadata,
                    )
                    logger.debug(
                        f"[{request_id}] Audio format converted",
                        target_format=requested_format.value,
                        elapsed_ms=int((time.time() - convert_start) * 1000),
                    )
                except Exception as e:
                    logger.warning(
                        f"[{request_id}] Format conversion failed, keeping WAV",
                        target_format=requested_format.value,
                        error=str(e),
                    )

            # 6. 파일 저장 (활성화된 경우)
            file_path = None
            if result.audio_data:
                storage_start = time.time()
                audio_bytes = (
                    result.audio_data.data
                    if hasattr(result.audio_data, 'data')
                    else result.audio_data
                )
                file_path = await self._file_storage.save(
                    request_id=request_id,
                    audio_data=audio_bytes,
                    audio_format=result.format,
                    sample_rate=result.sample_rate,
                )
                if file_path:
                    logger.debug(
                        f"[{request_id}] Audio file saved",
                        file_path=file_path,
                        elapsed_ms=int((time.time() - storage_start) * 1000),
                    )
                    result = SynthesisResult(
                        request_id=result.request_id,
                        status=result.status,
                        audio_data=result.audio_data,
                        file_path=file_path,
                        duration_seconds=result.duration_seconds,
                        sample_rate=result.sample_rate,
                        format=result.format,
                        processing_time_ms=result.processing_time_ms,
                        error_message=result.error_message,
                        metadata=result.metadata,
                    )

            # 통계 업데이트
            processing_time_ms = (time.time() - start_time) * 1000
            await self._stats.update(processing_time_ms, success=True)

            # 합성 완료 로그 (INFO 레벨 - 핵심 작업)
            log_synthesis(
                request_id=request_id,
                voice_id=request.voice_id,
                text_length=len(request.text),
                duration_ms=processing_time_ms,
                success=True,
                file_path=file_path,
            )

            return result

        except Exception as e:
            # 실패 통계 업데이트
            processing_time_ms = (time.time() - start_time) * 1000
            await self._stats.update(processing_time_ms, success=False)

            # 실패 로그
            log_synthesis(
                request_id=request_id,
                voice_id=request.voice_id,
                text_length=len(request.text),
                duration_ms=processing_time_ms,
                success=False,
                error=str(e),
            )

            # 알려진 예외는 그대로 전파
            if isinstance(e, (
                EmptyTextError,
                TextTooLongError,
                VoiceNotFoundError,
                VoiceNotReadyError,
                SynthesisTimeoutError,
                PoolExhaustedError,
            )):
                raise

            # 알 수 없는 예외는 래핑
            raise SynthesisError(
                f"Synthesis failed: {e}",
                request_id=request_id,
            )

        finally:
            # 활성 요청 추적에서 제거
            await self._stats.end_request(request_id)

    async def synthesize_streaming(
        self,
        request: SynthesisRequest,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[bytes, None]:
        """스트리밍 방식으로 텍스트를 음성으로 합성합니다."""
        start_time = time.time()
        request_id = request.request_id or str(uuid.uuid4())
        timeout = timeout or self._settings.performance.execution.task_timeout_seconds

        logger.debug(
            f"[{request_id}] Streaming synthesis request received",
            voice_id=request.voice_id,
            text_length=len(request.text),
        )

        await self._stats.start_request(request_id, start_time)

        # 파일 저장용 청크 버퍼 (활성화된 경우에만 사용)
        audio_chunks: List[bytes] = [] if self._file_storage.enabled else []
        audio_format = request.format or self._synthesis_config.default_format

        try:
            # 1. 요청 검증
            self._validate_request(request)

            # 2. 음성 및 모델 조회
            voice = self._voice_manager.get_voice_or_fallback(request.voice_id)
            instance_name = voice.model_instance

            logger.debug(
                f"[{request_id}] Streaming: Voice resolved",
                voice_id=voice.voice_id,
                model_instance=instance_name,
            )

            # 3. 내부 요청 생성
            internal_request = SynthesisRequest.create(
                text=request.text,
                voice_id=request.voice_id,
                request_id=request_id,
                speed=request.speed,
                pitch=request.pitch,
                volume=request.volume,
                format=audio_format,
                sample_rate=request.sample_rate or voice.sample_rate,
            )

            # 4. 모델 획득 및 스트리밍
            chunk_count = 0
            first_chunk_time = None
            async with self._model_manager.get_model(instance_name, timeout=timeout) as model:
                async for chunk in model.synthesize_streaming(internal_request):
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        logger.debug(
                            f"[{request_id}] Streaming: First chunk",
                            time_to_first_chunk_ms=int((first_chunk_time - start_time) * 1000),
                        )
                    yield chunk
                    chunk_count += 1
                    if self._file_storage.enabled:
                        audio_chunks.append(chunk)

            # 5. 파일 저장 (활성화된 경우)
            file_path = None
            if self._file_storage.enabled and audio_chunks:
                storage_start = time.time()
                combined_audio = b"".join(audio_chunks)
                file_path = await self._file_storage.save(
                    request_id=request_id,
                    audio_data=combined_audio,
                    audio_format=audio_format,
                    sample_rate=internal_request.sample_rate or voice.sample_rate,
                )
                if file_path:
                    logger.debug(
                        f"[{request_id}] Streaming: Audio file saved",
                        file_path=file_path,
                        elapsed_ms=int((time.time() - storage_start) * 1000),
                    )

            # 통계 업데이트
            processing_time_ms = (time.time() - start_time) * 1000
            await self._stats.update(processing_time_ms, success=True)

            logger.debug(
                f"[{request_id}] Streaming synthesis completed",
                chunk_count=chunk_count,
                processing_time_ms=int(processing_time_ms),
                file_path=file_path,
            )

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            await self._stats.update(processing_time_ms, success=False)

            logger.error(
                f"[{request_id}] Streaming synthesis failed",
                processing_time_ms=int(processing_time_ms),
                error=str(e),
            )

            if isinstance(e, (
                EmptyTextError,
                TextTooLongError,
                VoiceNotFoundError,
                VoiceNotReadyError,
                PoolExhaustedError,
            )):
                raise

            raise SynthesisError(
                f"Streaming synthesis failed: {e}",
                request_id=request_id,
            )

        finally:
            await self._stats.end_request(request_id)

    async def synthesize_batch(
        self,
        batch_request: BatchSynthesisRequest,
        max_concurrent: Optional[int] = None,
    ) -> BatchSynthesisResult:
        """여러 텍스트를 배치로 합성합니다."""
        start_time = time.time()
        batch_id = batch_request.batch_id or str(uuid.uuid4())
        max_concurrent = max_concurrent or 4

        logger.info(
            "Batch synthesis started",
            batch_id=batch_id,
            total_count=len(batch_request.requests),
            max_concurrent=max_concurrent,
        )

        results: List[SynthesisResult] = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_single(request: SynthesisRequest) -> SynthesisResult:
            async with semaphore:
                try:
                    return await self.synthesize(request)
                except Exception as e:
                    return SynthesisResult(
                        request_id=request.request_id or str(uuid.uuid4()),
                        status=SynthesisStatus.FAILED,
                        error_message=str(e),
                    )

        # 모든 요청을 동시에 처리
        tasks = [process_single(req) for req in batch_request.requests]
        results = await asyncio.gather(*tasks)

        # 통계 계산
        completed = sum(1 for r in results if r.status == SynthesisStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == SynthesisStatus.FAILED)
        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Batch synthesis completed",
            batch_id=batch_id,
            completed=completed,
            failed=failed,
            duration_ms=processing_time_ms,
        )

        return BatchSynthesisResult(
            batch_id=batch_id,
            results=results,
            total_count=len(results),
            completed_count=completed,
            failed_count=failed,
            processing_time_ms=processing_time_ms,
        )

    async def _execute_synthesis(
        self,
        request: SynthesisRequest,
        instance_name: str,
    ) -> SynthesisResult:
        """SynthesisExecutor에 위임하여 합성 실행."""
        return await self._synthesis_executor.execute(request, instance_name)

    def _validate_request(self, request: SynthesisRequest) -> None:
        """합성 요청을 검증합니다."""
        # 빈 텍스트 체크
        text = request.text.strip() if request.text else ""
        if not text:
            raise EmptyTextError("Text cannot be empty")

        # 텍스트 길이 체크
        max_length = self._synthesis_config.max_text_length
        if len(text) > max_length:
            raise TextTooLongError(
                f"Text length {len(text)} exceeds maximum {max_length}",
                text_length=len(text),
                max_length=max_length,
            )

        # 파라미터 범위 검증
        if request.speed is not None and not (0.5 <= request.speed <= 2.0):
            logger.warning(
                "Speed out of range, clamping",
                speed=request.speed,
                request_id=request.request_id,
            )

        if request.pitch is not None and not (0.5 <= request.pitch <= 2.0):
            logger.warning(
                "Pitch out of range, clamping",
                pitch=request.pitch,
                request_id=request.request_id,
            )

        if request.volume is not None and not (0.0 <= request.volume <= 1.0):
            logger.warning(
                "Volume out of range, clamping",
                volume=request.volume,
                request_id=request.request_id,
            )

    def get_stats(self) -> dict:
        """합성 통계를 반환합니다."""
        stats = self._stats.get_stats()
        stats["file_storage"] = self._file_storage.get_stats()
        return stats

    def get_window_stats(self) -> dict:
        """시간 윈도우 통계를 반환합니다."""
        return self._stats.get_window_stats()

    def get_active_requests(self) -> Dict[str, float]:
        """현재 활성 요청을 반환합니다."""
        return self._stats.get_active_requests()

    @property
    def active_count(self) -> int:
        """활성 요청 수."""
        return self._stats.active_count

    # =========================================================================
    # DynamicBatcher 관련 메서드 (고가용성)
    # =========================================================================

    async def start_batch_mode(self) -> None:
        """배치 모드를 시작합니다."""
        # 런타임 타입 확인 (model_manager에서 가져오기)
        await self._initialize_runtime_executor()

        if self._dynamic_batcher and not self._dynamic_batcher._is_running:
            await self._dynamic_batcher.start()
            logger.info(
                "Batch mode started",
                runtime_type=self._runtime_type.value if self._runtime_type else "unknown",
                executor_type=type(self._runtime_executor).__name__ if self._runtime_executor else "ThreadPool",
            )

    async def _initialize_runtime_executor(self) -> None:
        """런타임별 실행기를 초기화합니다."""
        if self._runtime_executor is not None:
            return

        # 첫 번째 enabled 인스턴스에서 런타임 타입 및 설정 추출
        self._model_config = self._extract_model_config()

        if self._runtime_type is None:
            logger.warning("No runtime type configured, skipping executor initialization")
            return

        # RuntimeExecutorFactory를 통해 실행기 생성
        pool_size = self._model_config.get("pool_size", 4) if self._model_config else 4

        try:
            self._runtime_executor = RuntimeExecutorFactory.create(
                runtime_type=self._runtime_type,
                pool_size=pool_size,
                model_config=self._model_config,
            )
            await self._runtime_executor.start()
            logger.info(
                "Runtime executor started",
                runtime_type=self._runtime_type.value,
                pool_size=pool_size,
            )
        except ValueError as e:
            logger.warning(f"Failed to create executor: {e}")

        # SynthesisExecutor 업데이트
        self._synthesis_executor.update_executor(
            runtime_executor=self._runtime_executor,
            runtime_type=self._runtime_type,
            model_config=self._model_config,
        )

        # BatchExecutor에 실행기 업데이트
        self._batch_executor.update_executor(
            runtime_executor=self._runtime_executor,
            thread_executor=self._executor,
            runtime_type=self._runtime_type,
            model_config=self._model_config,
        )

    def _extract_model_config(self) -> Optional[Dict[str, Any]]:
        """설정에서 모델 구성을 추출합니다."""
        try:
            for instance_name, config in self._settings.model_instances.items():
                if config.enabled:
                    self._runtime_type = config.options.runtime
                    options = config.options
                    return {
                        "model_path": config.model_path,
                        "device": config.device,
                        "pool_size": config.pool_size,
                        "fp16": options.fp16,
                        "vocoder_path": options.vocoder_path,
                        "num_steps": options.num_steps,
                        "t_shift": options.t_shift,
                        "guidance_scale": options.guidance_scale,
                        "max_concurrent_gpu": options.max_concurrent_gpu or config.pool_size,
                        "onnx_model_dir": getattr(options, "onnx_model_dir", None),
                        "trt_concurrent": getattr(options, "trt_concurrent", 2),
                        "tensorrt_model_dir": getattr(options, "tensorrt_model_dir", None),
                        "engine_file": getattr(options, "engine_file", None),
                    }
        except Exception as e:
            logger.warning(f"Failed to extract model config: {e}")
            self._runtime_type = RuntimeType.TENSORRT
        return None

    async def stop_batch_mode(self) -> None:
        """배치 모드를 중지합니다."""
        if self._dynamic_batcher and self._dynamic_batcher._is_running:
            await self._dynamic_batcher.stop()
            logger.info("Batch mode stopped")

        # 런타임 실행기 정리
        if self._runtime_executor:
            await self._runtime_executor.stop()
            self._runtime_executor = None
            logger.info("Runtime executor stopped")

        # ThreadPoolExecutor 정리
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
            logger.info("ThreadPoolExecutor shutdown")

        # 파일 저장 서비스 정리
        await self._file_storage.stop()

    # NOTE: Batch synthesis methods are now in BatchSynthesisExecutor
    # - _batch_synthesize_handler
    # - _batch_synthesize_pytorch
    # - _batch_synthesize_onnx
    # - _batch_synthesize_tensorrt
    # - _batch_synthesize_threadpool

    def get_batch_stats(self) -> dict:
        """배치 처리 통계를 반환합니다."""
        if self._dynamic_batcher:
            return self._dynamic_batcher.get_stats()
        return {"enabled": False}

    def get_pool_status(self) -> dict:
        """런타임 풀 상태를 반환합니다.

        Returns:
            {
                "runtime": str,
                "total_slots": int,
                "active": int,
                "waiting": int,
                "available": int,
                "synthesis_stats": {...},
                "window_stats": {...},
                "batch_stats": {...},
                "prometheus_enabled": bool,
            }
        """
        pool_status = {
            "runtime": self._runtime_type.value if self._runtime_type else "unknown",
            "total_slots": 0,
            "active": 0,
            "waiting": 0,
            "available": 0,
        }

        # 런타임 실행기에서 상태 가져오기
        if self._runtime_executor:
            executor_status = self._runtime_executor.get_pool_status()
            pool_status.update(executor_status)

        # 합성 통계 추가
        pool_status["synthesis_stats"] = self.get_stats()

        # 시간 윈도우 통계 추가
        pool_status["window_stats"] = self._stats.get_window_stats()

        # 배치 통계 추가
        pool_status["batch_stats"] = self.get_batch_stats()

        # Prometheus 활성화 여부
        pool_status["prometheus_enabled"] = self._stats.prometheus_enabled

        return pool_status

    @property
    def batch_mode_enabled(self) -> bool:
        """배치 모드 활성화 여부."""
        return self._batch_mode_enabled
