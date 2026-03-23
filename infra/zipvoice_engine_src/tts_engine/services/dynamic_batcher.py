# -*- coding: utf-8 -*-
"""Dynamic Batcher - 동적 배치 처리.

여러 TTS 요청을 모아서 단일 GPU 배치로 처리합니다.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from tts_engine.core.logging import get_logger
from tts_engine.domain.synthesis import SynthesisRequest, SynthesisResult, SynthesisStatus

logger = get_logger(__name__)


@dataclass
class PendingRequest:
    """대기 중인 요청."""
    request: SynthesisRequest
    future: asyncio.Future
    submit_time: float = field(default_factory=time.time)
    voice_id: str = ""

    def __post_init__(self) -> None:
        self.voice_id = self.request.voice_id


class DynamicBatcher:
    """동적 배치 처리기.

    요청을 일정 시간 동안 모아서 배치로 처리합니다.

    Attributes:
        max_batch_size: 최대 배치 크기
        max_wait_ms: 최대 대기 시간 (밀리초)
        group_by_voice: 음성별 그룹화 여부
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_ms: int = 30,
        group_by_voice: bool = False,
    ):
        """동적 배치 처리기 초기화.

        Args:
            max_batch_size: 최대 배치 크기
            max_wait_ms: 최대 대기 시간 (밀리초)
            group_by_voice: 음성별 그룹화 여부
        """
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._group_by_voice = group_by_voice

        self._pending: List[PendingRequest] = []
        self._lock = asyncio.Lock()
        self._batch_event = asyncio.Event()
        self._is_running = False
        self._worker_task: Optional[asyncio.Task] = None

        # 배치 처리 함수 (외부에서 설정)
        self._batch_synthesize_func: Optional[Callable[[List[SynthesisRequest]], Awaitable[List[SynthesisResult]]]] = None

        # 통계
        self._total_batches = 0
        self._total_requests = 0
        self._avg_batch_size = 0.0

        logger.info(
            "DynamicBatcher initialized",
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
            group_by_voice=group_by_voice,
        )

    def set_batch_handler(self, func: Callable[[List[SynthesisRequest]], Awaitable[List[SynthesisResult]]]) -> None:
        """배치 처리 함수를 설정합니다.

        Args:
            func: async def batch_synthesize(requests: List[SynthesisRequest]) -> List[SynthesisResult]
        """
        self._batch_synthesize_func = func

    async def start(self) -> None:
        """배치 처리기를 시작합니다."""
        if self._is_running:
            return

        self._is_running = True
        self._worker_task = asyncio.create_task(self._batch_worker())
        logger.info("DynamicBatcher started")

    async def stop(self) -> None:
        """배치 처리기를 중지합니다."""
        if not self._is_running:
            return

        self._is_running = False
        self._batch_event.set()  # 워커 깨우기

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # 대기 중인 요청 취소
        async with self._lock:
            for pending in self._pending:
                if not pending.future.done():
                    pending.future.cancel()
            self._pending.clear()

        logger.info("DynamicBatcher stopped")

    async def submit(self, request: SynthesisRequest) -> SynthesisResult:
        """요청을 제출하고 결과를 기다립니다.

        Args:
            request: 합성 요청

        Returns:
            합성 결과
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        pending = PendingRequest(
            request=request,
            future=future,
        )

        async with self._lock:
            self._pending.append(pending)
            self._total_requests += 1

            # 배치 크기에 도달하면 즉시 처리
            if len(self._pending) >= self._max_batch_size:
                self._batch_event.set()

        # 결과 대기
        return await future

    async def _batch_worker(self) -> None:
        """배치 처리 워커."""
        while self._is_running:
            try:
                # 최대 대기 시간 또는 배치 이벤트까지 대기
                try:
                    await asyncio.wait_for(
                        self._batch_event.wait(),
                        timeout=self._max_wait_ms / 1000.0,
                    )
                except asyncio.TimeoutError:
                    pass

                self._batch_event.clear()

                # 대기 중인 요청 수집
                async with self._lock:
                    if not self._pending:
                        continue

                    # 배치 크기만큼 가져오기
                    batch = self._pending[:self._max_batch_size]
                    self._pending = self._pending[self._max_batch_size:]

                if batch:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Batch worker error", error=str(e))

    async def _process_batch(self, batch: List[PendingRequest]) -> None:
        """배치를 처리합니다.

        Args:
            batch: 대기 중인 요청 목록
        """
        if not self._batch_synthesize_func:
            # 배치 핸들러가 없으면 실패 처리
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(
                        RuntimeError("Batch handler not configured")
                    )
            return

        batch_size = len(batch)
        start_time = time.time()

        logger.debug(
            "Processing batch",
            batch_size=batch_size,
        )

        try:
            # 요청 목록 추출
            requests = [p.request for p in batch]

            # 배치 합성 실행
            results = await self._batch_synthesize_func(requests)

            # 결과 분배
            for pending, result in zip(batch, results):
                if not pending.future.done():
                    pending.future.set_result(result)

            # 통계 업데이트
            self._total_batches += 1
            self._avg_batch_size = (
                (self._avg_batch_size * (self._total_batches - 1) + batch_size)
                / self._total_batches
            )

            elapsed = (time.time() - start_time) * 1000
            logger.info(
                "Batch processed",
                batch_size=batch_size,
                duration_ms=f"{elapsed:.1f}",
                avg_per_request_ms=f"{elapsed / batch_size:.1f}",
            )

        except Exception as e:
            logger.error("Batch processing failed", error=str(e), batch_size=batch_size)
            # 모든 요청에 에러 전파
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(e)

    def get_stats(self) -> dict:
        """통계를 반환합니다."""
        return {
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "avg_batch_size": round(self._avg_batch_size, 2),
            "pending_count": len(self._pending),
            "is_running": self._is_running,
            "max_batch_size": self._max_batch_size,
            "max_wait_ms": self._max_wait_ms,
        }
