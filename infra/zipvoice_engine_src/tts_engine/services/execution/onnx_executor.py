# -*- coding: utf-8 -*-
"""ONNX 세션 풀 + 스레드 풀 실행기."""

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from tts_engine.core.constants import RuntimeType
from tts_engine.core.exceptions import ExecutionError, ExecutionTimeoutError
from tts_engine.core.logging import get_logger
from tts_engine.services.execution.base import QueuedRuntimeExecutor

logger = get_logger(__name__)


class ONNXExecutor(QueuedRuntimeExecutor):
    """ONNX 세션 풀 + 스레드 풀 실행기.

    내장 큐로 요청을 관리합니다.
    """

    def __init__(
        self,
        pool_size: int = 4,
        model_config: Optional[dict] = None,
        max_queue_size: int = 500,
        queue_timeout: float = 30.0,
    ):
        self._model_config = model_config or {}
        self._max_concurrent_gpu = self._model_config.get("max_concurrent_gpu", pool_size)

        super().__init__(
            runtime_type=RuntimeType.ONNX,
            pool_size=pool_size,
            max_queue_size=max_queue_size,
            queue_timeout=queue_timeout,
        )

        self._executor: Optional[ThreadPoolExecutor] = None
        self._session_pool = None
        self._gpu_semaphore: Optional[threading.Semaphore] = None

        # 통계
        self._submitted_count = 0
        self._completed_count = 0
        self._failed_count = 0

        # 실시간 상태 추적
        self._active_count = 0
        self._waiting_count = 0
        self._status_lock = threading.Lock()

    async def _start_runtime(self) -> None:
        """ONNX 런타임 시작."""
        from tts_engine.services.execution.onnx_session_pool import (
            OnnxSessionPool,
            OnnxSessionConfig,
        )

        onnx_model_dir = self._model_config.get("onnx_model_dir") or self._model_config.get("model_path", "")

        config = OnnxSessionConfig(
            model_dir=onnx_model_dir,
            vocoder_path=self._model_config.get("vocoder_path", "./models/vocos/mel-24khz"),
            device=self._model_config.get("device", "cuda:0"),
            num_threads=self._model_config.get("num_threads", 1),
            text_encoder_path=self._model_config.get("text_encoder_path"),
            fm_decoder_path=self._model_config.get("fm_decoder_path"),
            feat_scale=self._model_config.get("feat_scale", 0.1),
            target_rms=self._model_config.get("target_rms", 0.1),
        )

        self._session_pool = OnnxSessionPool(config=config, pool_size=self._pool_size)
        self._session_pool.start()

        self._executor = ThreadPoolExecutor(
            max_workers=self._pool_size * 2,
            thread_name_prefix="onnx_",
        )

        self._gpu_semaphore = threading.Semaphore(self._max_concurrent_gpu)

        logger.info(
            "ONNX runtime started",
            pool_size=self._pool_size,
            max_concurrent_gpu=self._max_concurrent_gpu,
        )

    async def _stop_runtime(self) -> None:
        """ONNX 런타임 중지."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        if self._session_pool:
            self._session_pool.stop()
            self._session_pool = None

        logger.info("ONNX runtime stopped")

    async def _execute(self, func: Callable, *args, **kwargs) -> Any:
        """GPU 세마포어를 사용하여 실행합니다."""
        if not self._executor or not self._session_pool:
            raise ExecutionError("ONNXExecutor is not running")

        from tts_engine.services.execution.onnx_session_pool import (
            SynthesisTask,
            synthesize_with_session,
        )

        request_data = kwargs.get("request_data", {})

        task = SynthesisTask(
            task_id=request_data.get("task_id", str(uuid.uuid4())),
            text=request_data.get("text", ""),
            voice_id=request_data.get("voice_id", ""),
            prompt_wav_path=request_data.get("prompt_wav_path", ""),
            prompt_text=request_data.get("prompt_text", ""),
            speed=request_data.get("speed", 1.0),
            num_steps=request_data.get("num_steps", 16),
            t_shift=request_data.get("t_shift", 0.5),
            guidance_scale=request_data.get("guidance_scale", 1.0),
        )

        self._submitted_count += 1

        def _execute_task():
            # 대기 시작
            with self._status_lock:
                self._waiting_count += 1

            self._gpu_semaphore.acquire()

            # 대기 종료, 실행 시작
            with self._status_lock:
                self._waiting_count -= 1
                self._active_count += 1

            try:
                with self._session_pool.acquire(timeout=30.0) as session:
                    return synthesize_with_session(session, task)
            finally:
                # 실행 종료
                with self._status_lock:
                    self._active_count -= 1
                self._gpu_semaphore.release()

        loop = asyncio.get_event_loop()

        try:
            future = loop.run_in_executor(self._executor, _execute_task)
            result = await future

            if result.success:
                self._completed_count += 1
                return {
                    "audio_data": result.audio_data,
                    "sample_rate": result.sample_rate,
                    "processing_time_ms": result.processing_time_ms,
                }
            else:
                self._failed_count += 1
                raise ExecutionError(result.error_message)

        except ExecutionError:
            raise
        except Exception as e:
            self._failed_count += 1
            raise ExecutionError(f"Task failed: {e}")

    def get_stats(self) -> dict:
        with self._status_lock:
            active = self._active_count
            waiting = self._waiting_count

        queue_status = self._request_queue.get_status()

        stats = {
            "type": "onnx_session_pool",
            "runtime": self._runtime_type.value,
            "pool_size": self._pool_size,
            "max_concurrent_gpu": self._max_concurrent_gpu,
            "is_running": self._is_running,
            "submitted_count": self._submitted_count,
            "completed_count": self._completed_count,
            "failed_count": self._failed_count,
            # 실시간 상태
            "active_count": active,
            "waiting_count": waiting,
            "available_slots": self._max_concurrent_gpu - active,
            # 큐 상태
            "queue_size": queue_status["queue_size"],
            "queue_rejected": queue_status["total_rejected"],
            "queue_timeout": queue_status["total_timeout"],
        }
        if self._session_pool:
            stats["session_pool"] = self._session_pool.get_stats()
        return stats

    def get_pool_status(self) -> dict:
        """실시간 풀 상태를 반환합니다."""
        with self._status_lock:
            active = self._active_count
            waiting = self._waiting_count

        session_stats = self._session_pool.get_stats() if self._session_pool else {}
        queue_status = self._request_queue.get_status()

        return {
            "runtime": "onnx",
            "total_slots": self._max_concurrent_gpu,
            "active": active,
            "waiting": waiting,
            "available": self._max_concurrent_gpu - active,
            "session_pool": {
                "pool_size": session_stats.get("pool_size", 0),
                "available": session_stats.get("available", 0),
                "in_use": session_stats.get("in_use", 0),
            },
            # 큐 상태
            "queue_size": queue_status["queue_size"],
            "max_queue_size": queue_status["max_queue_size"],
            "queue_utilization": queue_status["queue_utilization"],
            "queue_rejected": queue_status["total_rejected"],
            "queue_timeout": queue_status["total_timeout"],
        }
