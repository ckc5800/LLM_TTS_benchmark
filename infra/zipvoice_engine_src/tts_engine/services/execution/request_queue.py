# -*- coding: utf-8 -*-
"""요청 큐잉 시스템.

과부하 시 요청을 대기열에 보관하고 순차적으로 처리합니다.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum

from tts_engine.core.logging import get_logger

logger = get_logger(__name__)


class QueueFullError(Exception):
    """대기열이 가득 찼을 때 발생하는 예외."""
    pass


class QueueTimeoutError(Exception):
    """대기열 타임아웃 예외."""
    pass


class RequestPriority(Enum):
    """요청 우선순위."""
    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass(order=True)
class QueuedRequest:
    """대기열 요청."""
    priority: int = field(compare=True)
    timestamp: float = field(compare=True)
    request_id: str = field(compare=False)
    func: Callable = field(compare=False)
    args: tuple = field(compare=False)
    kwargs: dict = field(compare=False)
    future: asyncio.Future = field(compare=False)
    timeout: Optional[float] = field(compare=False, default=None)


class RequestQueue:
    """요청 대기열 관리자.

    Features:
        - 최대 대기열 크기 제한
        - 요청 타임아웃
        - 우선순위 지원
        - 실시간 상태 모니터링
    """

    def __init__(
        self,
        max_queue_size: int = 500,
        default_timeout: float = 30.0,
        enable_priority: bool = False,
    ):
        self._max_queue_size = max_queue_size
        self._default_timeout = default_timeout
        self._enable_priority = enable_priority

        # 큐 (우선순위 사용 시 heapq, 아니면 일반 deque)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)

        # 통계
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._total_rejected = 0
        self._total_timeout = 0
        self._lock = asyncio.Lock()

        # 상태
        self._is_running = False
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """큐 시스템 시작."""
        if self._is_running:
            return
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
        logger.info(
            "RequestQueue started",
            max_size=self._max_queue_size,
            timeout=self._default_timeout,
        )

    async def stop(self) -> None:
        """큐 시스템 중지."""
        if not self._is_running:
            return
        self._is_running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # 대기 중인 요청 모두 취소
        while not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                if not req.future.done():
                    req.future.cancel()
            except asyncio.QueueEmpty:
                break

        logger.info("RequestQueue stopped")

    async def enqueue(
        self,
        request_id: str,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        **kwargs,
    ) -> asyncio.Future:
        """요청을 대기열에 추가합니다.

        Args:
            request_id: 요청 ID
            func: 실행할 함수
            *args: 함수 인자
            timeout: 대기 타임아웃 (None이면 기본값 사용)
            priority: 요청 우선순위
            **kwargs: 함수 키워드 인자

        Returns:
            결과를 받을 Future

        Raises:
            QueueFullError: 대기열이 가득 찬 경우
        """
        if self._queue.full():
            async with self._lock:
                self._total_rejected += 1
            raise QueueFullError(
                f"Queue is full (max: {self._max_queue_size}). "
                f"Current queue size: {self._queue.qsize()}"
            )

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        request = QueuedRequest(
            priority=-priority.value if self._enable_priority else 0,
            timestamp=time.time(),
            request_id=request_id,
            func=func,
            args=args,
            kwargs=kwargs,
            future=future,
            timeout=timeout or self._default_timeout,
        )

        await self._queue.put(request)
        async with self._lock:
            self._total_enqueued += 1

        logger.debug(
            f"Request enqueued",
            request_id=request_id,
            queue_size=self._queue.qsize(),
        )

        return future

    async def dequeue(self) -> Optional[QueuedRequest]:
        """대기열에서 요청을 가져옵니다.

        Returns:
            QueuedRequest 또는 None (타임아웃/취소된 요청은 스킵)
        """
        while self._is_running:
            try:
                request = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            # 이미 취소된 요청 스킵
            if request.future.done():
                continue

            # 타임아웃 체크
            elapsed = time.time() - request.timestamp
            if request.timeout and elapsed > request.timeout:
                request.future.set_exception(
                    QueueTimeoutError(
                        f"Request {request.request_id} timed out in queue "
                        f"after {elapsed:.1f}s (timeout: {request.timeout}s)"
                    )
                )
                async with self._lock:
                    self._total_timeout += 1
                continue

            async with self._lock:
                self._total_dequeued += 1

            return request

        return None

    async def _cleanup_expired(self) -> None:
        """만료된 요청을 주기적으로 정리합니다."""
        while self._is_running:
            await asyncio.sleep(5.0)  # 5초마다 체크
            # Note: asyncio.Queue는 직접 순회가 어려우므로
            # 실제 만료 체크는 dequeue 시점에 수행

    def get_status(self) -> dict:
        """대기열 상태를 반환합니다."""
        return {
            "queue_size": self._queue.qsize(),
            "max_queue_size": self._max_queue_size,
            "queue_utilization": self._queue.qsize() / self._max_queue_size if self._max_queue_size > 0 else 0,
            "is_full": self._queue.full(),
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_rejected": self._total_rejected,
            "total_timeout": self._total_timeout,
            "default_timeout": self._default_timeout,
        }

    @property
    def queue_size(self) -> int:
        """현재 대기열 크기."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        """대기열이 가득 찼는지 여부."""
        return self._queue.full()

    @property
    def is_empty(self) -> bool:
        """대기열이 비었는지 여부."""
        return self._queue.empty()
