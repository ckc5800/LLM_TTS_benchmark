# -*- coding: utf-8 -*-
"""런타임 실행기 베이스 클래스."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from tts_engine.core.constants import RuntimeType
from tts_engine.core.logging import get_logger
from tts_engine.services.execution.request_queue import (
    RequestQueue,
    QueueFullError,
    QueueTimeoutError,
    RequestPriority,
)

logger = get_logger(__name__)


class QueuedRuntimeExecutor(ABC):
    """큐잉을 지원하는 런타임 실행기 추상 베이스 클래스.

    모든 런타임 실행기가 상속해야 하는 베이스 클래스입니다.
    내장 요청 큐를 통해 과부하 시에도 안정적으로 요청을 처리합니다.
    """

    def __init__(
        self,
        runtime_type: RuntimeType,
        pool_size: int = 4,
        max_queue_size: int = 500,
        queue_timeout: float = 30.0,
    ):
        self._runtime_type = runtime_type
        self._pool_size = pool_size
        self._is_running = False

        # 요청 큐
        self._request_queue = RequestQueue(
            max_queue_size=max_queue_size,
            default_timeout=queue_timeout,
        )

        # 큐 처리 워커 태스크들
        self._queue_workers: list[asyncio.Task] = []
        self._num_queue_workers = pool_size  # 풀 크기만큼 워커 생성

    @property
    def runtime_type(self) -> RuntimeType:
        return self._runtime_type

    @property
    def pool_size(self) -> int:
        return self._pool_size

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def start(self) -> None:
        """실행기를 시작합니다."""
        if self._is_running:
            logger.warning(f"{self.__class__.__name__} is already running")
            return

        # 큐 시작
        await self._request_queue.start()

        # 서브클래스 초기화
        await self._start_runtime()

        # 큐 처리 워커 시작
        for i in range(self._num_queue_workers):
            worker = asyncio.create_task(self._queue_worker(i))
            self._queue_workers.append(worker)

        self._is_running = True
        logger.info(
            f"{self.__class__.__name__} started",
            pool_size=self._pool_size,
            queue_workers=self._num_queue_workers,
        )

    async def stop(self) -> None:
        """실행기를 중지합니다."""
        if not self._is_running:
            return

        logger.info(f"Stopping {self.__class__.__name__}")
        self._is_running = False

        # 큐 워커 중지
        for worker in self._queue_workers:
            worker.cancel()
        await asyncio.gather(*self._queue_workers, return_exceptions=True)
        self._queue_workers.clear()

        # 큐 중지
        await self._request_queue.stop()

        # 서브클래스 정리
        await self._stop_runtime()

        logger.info(f"{self.__class__.__name__} stopped")

    async def submit(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        **kwargs,
    ) -> Any:
        """작업을 제출합니다.

        큐가 가득 차면 QueueFullError를 발생시킵니다.
        타임아웃이 발생하면 QueueTimeoutError를 발생시킵니다.
        """
        if not self._is_running:
            raise RuntimeError(f"{self.__class__.__name__} is not running")

        request_id = kwargs.get("request_data", {}).get("task_id", "unknown")

        # 큐에 요청 추가
        future = await self._request_queue.enqueue(
            request_id=request_id,
            func=func,
            *args,
            timeout=timeout,
            priority=priority,
            **kwargs,
        )

        # 결과 대기
        return await future

    async def _queue_worker(self, worker_id: int) -> None:
        """큐에서 요청을 가져와 처리하는 워커."""
        logger.debug(f"Queue worker {worker_id} started")

        while self._is_running:
            try:
                request = await self._request_queue.dequeue()
                if request is None:
                    continue

                try:
                    # 실제 실행
                    result = await self._execute(
                        request.func,
                        *request.args,
                        **request.kwargs,
                    )
                    if not request.future.done():
                        request.future.set_result(result)
                except Exception as e:
                    if not request.future.done():
                        request.future.set_exception(e)
                    logger.error(
                        f"Queue worker {worker_id} execution failed",
                        request_id=request.request_id,
                        error=str(e),
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue worker {worker_id} error: {e}")

        logger.debug(f"Queue worker {worker_id} stopped")

    @abstractmethod
    async def _start_runtime(self) -> None:
        """런타임별 시작 로직 (서브클래스에서 구현)."""
        pass

    @abstractmethod
    async def _stop_runtime(self) -> None:
        """런타임별 중지 로직 (서브클래스에서 구현)."""
        pass

    @abstractmethod
    async def _execute(self, func: Callable, *args, **kwargs) -> Any:
        """실제 실행 로직 (서브클래스에서 구현).

        GPU 세마포어 등의 동시 실행 제어는 이 메서드에서 처리합니다.
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """실행기 통계를 반환합니다."""
        pass

    def get_pool_status(self) -> dict:
        """실시간 풀 상태를 반환합니다."""
        queue_status = self._request_queue.get_status()
        return {
            "runtime": self._runtime_type.value,
            "total_slots": self._pool_size,
            "queue_size": queue_status["queue_size"],
            "max_queue_size": queue_status["max_queue_size"],
            "queue_utilization": queue_status["queue_utilization"],
            "queue_rejected": queue_status["total_rejected"],
            "queue_timeout": queue_status["total_timeout"],
        }

    def get_queue_status(self) -> dict:
        """대기열 상태를 반환합니다."""
        return self._request_queue.get_status()


# 기존 RuntimeExecutor는 호환성을 위해 유지 (별칭)
RuntimeExecutor = QueuedRuntimeExecutor


@dataclass
class WorkerTask:
    """워커에게 전달할 태스크."""
    task_id: str
    text: str
    voice_id: str
    prompt_wav_path: str
    prompt_text: str
    speed: float = 1.0
    num_steps: int = 16
    t_shift: float = 0.5
    guidance_scale: float = 1.0


@dataclass
class WorkerResult:
    """워커로부터 받는 결과."""
    task_id: str
    success: bool
    audio_data: Optional[bytes] = None
    sample_rate: int = 24000
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None
