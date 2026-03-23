# -*- coding: utf-8 -*-
"""TensorRT 스레드 풀 실행기."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from tts_engine.core.constants import RuntimeType
from tts_engine.core.exceptions import ExecutionError, ExecutionTimeoutError
from tts_engine.core.logging import get_logger
from tts_engine.services.execution.base import QueuedRuntimeExecutor

logger = get_logger(__name__)


class TensorRTExecutor(QueuedRuntimeExecutor):
    """TensorRT 스레드 풀 실행기.

    GPU 세마포어로 동시 실행을 제한하고, 내장 큐로 요청을 관리합니다.
    """

    def __init__(
        self,
        pool_size: int = 4,
        model_config: Optional[dict] = None,
        max_queue_size: int = 500,
        queue_timeout: float = 30.0,
    ):
        self._model_config = model_config or {}
        self._trt_concurrent = self._model_config.get("trt_concurrent", 4)

        super().__init__(
            runtime_type=RuntimeType.TENSORRT,
            pool_size=pool_size,
            max_queue_size=max_queue_size,
            queue_timeout=queue_timeout,
        )

        self._executor: Optional[ThreadPoolExecutor] = None
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
        """TensorRT 런타임 시작."""
        self._executor = ThreadPoolExecutor(
            max_workers=self._pool_size * 2,
            thread_name_prefix="trt_",
        )
        self._gpu_semaphore = threading.Semaphore(self._trt_concurrent)

        logger.info(
            "TensorRT runtime started",
            pool_size=self._pool_size,
            trt_concurrent=self._trt_concurrent,
        )

    async def _stop_runtime(self) -> None:
        """TensorRT 런타임 중지."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        logger.info("TensorRT runtime stopped")

    async def _execute(self, func: Callable, *args, **kwargs) -> Any:
        """GPU 세마포어를 사용하여 실행합니다."""
        if not self._executor:
            raise ExecutionError("TensorRTExecutor is not running")

        self._submitted_count += 1
        task_id = kwargs.get("request_data", {}).get("task_id", "unknown")

        def _execute_with_semaphore():
            # 대기 시작
            with self._status_lock:
                self._waiting_count += 1

            logger.debug("TRT GPU semaphore waiting", task_id=task_id)
            self._gpu_semaphore.acquire()

            # 대기 종료, 실행 시작
            with self._status_lock:
                self._waiting_count -= 1
                self._active_count += 1

            logger.debug("TRT GPU semaphore acquired", task_id=task_id)
            try:
                return func(*args)
            finally:
                # 실행 종료
                with self._status_lock:
                    self._active_count -= 1
                self._gpu_semaphore.release()
                logger.debug("TRT GPU semaphore released", task_id=task_id)

        loop = asyncio.get_event_loop()

        try:
            future = loop.run_in_executor(self._executor, _execute_with_semaphore)
            result = await future
            self._completed_count += 1
            return result

        except Exception as e:
            self._failed_count += 1
            raise ExecutionError(f"Task failed: {e}")

    @property
    def trt_concurrent(self) -> int:
        """TensorRT 동시 GPU 실행 수."""
        return self._trt_concurrent

    def get_stats(self) -> dict:
        with self._status_lock:
            active = self._active_count
            waiting = self._waiting_count

        queue_status = self._request_queue.get_status()

        return {
            "type": "tensorrt_threadpool",
            "runtime": self._runtime_type.value,
            "pool_size": self._pool_size,
            "trt_concurrent": self._trt_concurrent,
            "is_running": self._is_running,
            "submitted_count": self._submitted_count,
            "completed_count": self._completed_count,
            "failed_count": self._failed_count,
            # 실시간 상태
            "active_count": active,
            "waiting_count": waiting,
            "available_slots": self._trt_concurrent - active,
            # 큐 상태
            "queue_size": queue_status["queue_size"],
            "queue_rejected": queue_status["total_rejected"],
            "queue_timeout": queue_status["total_timeout"],
        }

    def get_pool_status(self) -> dict:
        """실시간 풀 상태를 반환합니다."""
        with self._status_lock:
            active = self._active_count
            waiting = self._waiting_count

        queue_status = self._request_queue.get_status()

        return {
            "runtime": "tensorrt",
            "total_slots": self._trt_concurrent,
            "active": active,
            "waiting": waiting,
            "available": self._trt_concurrent - active,
            "thread_pool_size": self._pool_size * 2,
            # 큐 상태
            "queue_size": queue_status["queue_size"],
            "max_queue_size": queue_status["max_queue_size"],
            "queue_utilization": queue_status["queue_utilization"],
            "queue_rejected": queue_status["total_rejected"],
            "queue_timeout": queue_status["total_timeout"],
        }
