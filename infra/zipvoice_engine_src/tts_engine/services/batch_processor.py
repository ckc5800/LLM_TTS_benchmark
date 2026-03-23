# -*- coding: utf-8 -*-
"""Batch Processor - 배치 합성 처리.

여러 TTS 요청을 효율적으로 배치 처리합니다.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from tts_engine.core.config import Settings
from tts_engine.core.logging import get_logger
from tts_engine.domain.synthesis import (
    BatchSynthesisRequest,
    BatchSynthesisResult,
    SynthesisRequest,
    SynthesisResult,
    SynthesisStatus,
)

logger = get_logger(__name__)


class BatchStatus(str, Enum):
    """배치 상태."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass
class BatchJob:
    """배치 작업 정보."""

    batch_id: str
    requests: List[SynthesisRequest]
    status: BatchStatus = BatchStatus.QUEUED
    results: List[Optional[SynthesisResult]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    priority: int = 1  # 0=low, 1=normal, 2=high

    @property
    def total_count(self) -> int:
        return len(self.requests)

    @property
    def completed_count(self) -> int:
        return sum(1 for r in self.results if r and r.status == SynthesisStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r and r.status == SynthesisStatus.FAILED)

    @property
    def progress_percent(self) -> int:
        if not self.results:
            return 0
        completed = sum(1 for r in self.results if r is not None)
        return int((completed / self.total_count) * 100)


class BatchProcessor:
    """배치 처리기.

    TTS 요청을 배치로 처리하고 상태를 관리합니다.

    Attributes:
        _settings: 애플리케이션 설정
        _jobs: 배치 작업 저장소
        _queue: 처리 대기 큐
        _max_concurrent: 최대 동시 처리 수
    """

    def __init__(
        self,
        settings: Settings,
        max_concurrent: int = 4,
    ):
        """배치 처리기 초기화.

        Args:
            settings: 애플리케이션 설정
            max_concurrent: 최대 동시 배치 처리 수
        """
        self._settings = settings
        self._jobs: Dict[str, BatchJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._is_running = False
        self._worker_task: Optional[asyncio.Task] = None

        logger.info(
            "BatchProcessor initialized",
            max_concurrent=max_concurrent,
        )

    async def start(self) -> None:
        """배치 처리기를 시작합니다."""
        if self._is_running:
            return

        self._is_running = True
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info("BatchProcessor started")

    async def stop(self) -> None:
        """배치 처리기를 중지합니다."""
        if not self._is_running:
            return

        self._is_running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info("BatchProcessor stopped")

    async def submit_batch(
        self,
        batch_request: BatchSynthesisRequest,
        synthesize_func,
        priority: int = 1,
    ) -> str:
        """배치 작업을 제출합니다.

        Args:
            batch_request: 배치 합성 요청
            synthesize_func: 개별 합성 함수
            priority: 우선순위 (0=low, 1=normal, 2=high)

        Returns:
            배치 ID
        """
        batch_id = batch_request.batch_id or str(uuid.uuid4())

        job = BatchJob(
            batch_id=batch_id,
            requests=list(batch_request.requests),
            priority=priority,
        )

        self._jobs[batch_id] = job

        # 큐에 추가하지 않고 바로 처리 (동기 모드)
        await self._process_batch(job, synthesize_func)

        logger.info(
            "Batch submitted",
            batch_id=batch_id,
            total_count=job.total_count,
            priority=priority,
        )

        return batch_id

    async def _process_queue(self) -> None:
        """큐에서 배치를 가져와 처리합니다."""
        while self._is_running:
            try:
                batch_id = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )

                job = self._jobs.get(batch_id)
                if job and job.status == BatchStatus.QUEUED:
                    async with self._semaphore:
                        await self._process_batch(job, None)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error processing batch queue", error=str(e))

    async def _process_batch(
        self,
        job: BatchJob,
        synthesize_func,
    ) -> None:
        """배치를 처리합니다.

        Args:
            job: 배치 작업
            synthesize_func: 개별 합성 함수
        """
        job.status = BatchStatus.PROCESSING
        job.started_at = time.time()
        job.results = [None] * job.total_count

        logger.info(
            "Processing batch",
            batch_id=job.batch_id,
            total_count=job.total_count,
        )

        # 동시 처리를 위한 세마포어
        batch_semaphore = asyncio.Semaphore(
            self._settings.performance.auto_batch.max_batch_size
            if self._settings.performance.auto_batch.enabled
            else 4
        )

        async def process_single(index: int, request: SynthesisRequest):
            async with batch_semaphore:
                try:
                    if synthesize_func:
                        result = await synthesize_func(request)
                    else:
                        # 기본 더미 결과
                        result = SynthesisResult(
                            request_id=request.request_id or str(uuid.uuid4()),
                            status=SynthesisStatus.COMPLETED,
                        )
                    job.results[index] = result
                except Exception as e:
                    logger.error(
                        "Batch item failed",
                        batch_id=job.batch_id,
                        index=index,
                        error=str(e),
                    )
                    job.results[index] = SynthesisResult(
                        request_id=request.request_id or str(uuid.uuid4()),
                        status=SynthesisStatus.FAILED,
                        error_message=str(e),
                    )

        # 모든 요청 병렬 처리
        tasks = [
            process_single(i, req)
            for i, req in enumerate(job.requests)
        ]
        await asyncio.gather(*tasks)

        # 상태 업데이트
        job.completed_at = time.time()

        if job.failed_count == 0:
            job.status = BatchStatus.COMPLETED
        elif job.completed_count == 0:
            job.status = BatchStatus.FAILED
        else:
            job.status = BatchStatus.PARTIAL

        logger.info(
            "Batch completed",
            batch_id=job.batch_id,
            status=job.status.value,
            completed=job.completed_count,
            failed=job.failed_count,
            duration_ms=(job.completed_at - job.started_at) * 1000,
        )

    def get_batch_status(self, batch_id: str) -> Optional[BatchJob]:
        """배치 상태를 조회합니다.

        Args:
            batch_id: 배치 ID

        Returns:
            BatchJob 또는 None
        """
        return self._jobs.get(batch_id)

    def get_batch_result(self, batch_id: str) -> Optional[BatchSynthesisResult]:
        """배치 결과를 반환합니다.

        Args:
            batch_id: 배치 ID

        Returns:
            BatchSynthesisResult 또는 None
        """
        job = self._jobs.get(batch_id)
        if not job:
            return None

        processing_time_ms = 0.0
        if job.started_at and job.completed_at:
            processing_time_ms = (job.completed_at - job.started_at) * 1000

        return BatchSynthesisResult(
            batch_id=job.batch_id,
            results=[r for r in job.results if r is not None],
            total_count=job.total_count,
            completed_count=job.completed_count,
            failed_count=job.failed_count,
            processing_time_ms=processing_time_ms,
        )

    def cancel_batch(self, batch_id: str) -> bool:
        """배치를 취소합니다.

        Args:
            batch_id: 배치 ID

        Returns:
            취소 성공 여부
        """
        job = self._jobs.get(batch_id)
        if not job:
            return False

        if job.status in (BatchStatus.QUEUED, BatchStatus.PROCESSING):
            job.status = BatchStatus.CANCELLED
            job.completed_at = time.time()
            logger.info("Batch cancelled", batch_id=batch_id)
            return True

        return False

    def cleanup_old_jobs(self, max_age_seconds: float = 3600) -> int:
        """오래된 작업을 정리합니다.

        Args:
            max_age_seconds: 최대 보관 시간 (초)

        Returns:
            정리된 작업 수
        """
        now = time.time()
        old_jobs = [
            batch_id
            for batch_id, job in self._jobs.items()
            if job.completed_at and (now - job.completed_at) > max_age_seconds
        ]

        for batch_id in old_jobs:
            del self._jobs[batch_id]

        if old_jobs:
            logger.info("Cleaned up old batch jobs", count=len(old_jobs))

        return len(old_jobs)

    def get_stats(self) -> dict:
        """배치 처리기 통계를 반환합니다.

        Returns:
            통계 딕셔너리
        """
        status_counts = {}
        for job in self._jobs.values():
            status = job.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_jobs": len(self._jobs),
            "is_running": self._is_running,
            "max_concurrent": self._max_concurrent,
            "status_counts": status_counts,
            "queue_size": self._queue.qsize(),
        }
