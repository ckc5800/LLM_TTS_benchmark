# -*- coding: utf-8 -*-
"""Thread Executor - 스레드 풀 기반 실행기.

asyncio와 통합된 스레드 풀 실행기를 제공합니다.
GPU 통합 메모리 환경(DGX Spark)에서 권장됩니다.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, List, Optional, TypeVar

from tts_engine.core.exceptions import ExecutionError, ExecutionTimeoutError
from tts_engine.core.logging import get_logger
from tts_engine.services.execution.executor_base import ExecutorBase

logger = get_logger(__name__)

T = TypeVar("T")


class ThreadExecutor(ExecutorBase):
    """스레드 풀 기반 실행기.

    CPU/GPU 통합 메모리 환경에서 효율적인 병렬 처리를 제공합니다.
    프로세스 간 데이터 복사 오버헤드가 없어 DGX Spark에 적합합니다.

    Attributes:
        _executor: ThreadPoolExecutor 인스턴스
        _submitted_count: 제출된 작업 수
        _completed_count: 완료된 작업 수
        _failed_count: 실패한 작업 수
    """

    def __init__(self, pool_size: int = 8):
        """스레드 실행기 초기화.

        Args:
            pool_size: 스레드 풀 크기 (기본: 8)
        """
        super().__init__(pool_size=pool_size)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._submitted_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._total_execution_time = 0.0

        logger.info("ThreadExecutor initialized", pool_size=pool_size)

    async def start(self) -> None:
        """스레드 풀을 시작합니다."""
        if self._is_running:
            logger.warning("ThreadExecutor is already running")
            return

        self._executor = ThreadPoolExecutor(
            max_workers=self._pool_size,
            thread_name_prefix="tts_worker_",
        )
        self._is_running = True

        logger.info("ThreadExecutor started", pool_size=self._pool_size)

    async def stop(self, wait: bool = True) -> None:
        """스레드 풀을 중지합니다.

        Args:
            wait: 진행 중인 작업 완료 대기 여부
        """
        if not self._is_running or not self._executor:
            logger.warning("ThreadExecutor is not running")
            return

        logger.info("Stopping ThreadExecutor", wait=wait)

        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        self._executor = None
        self._is_running = False

        logger.info(
            "ThreadExecutor stopped",
            submitted=self._submitted_count,
            completed=self._completed_count,
            failed=self._failed_count,
        )

    async def submit(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """작업을 스레드 풀에 제출합니다.

        Args:
            func: 실행할 함수
            *args: 위치 인자
            timeout: 타임아웃 (초)
            **kwargs: 키워드 인자

        Returns:
            함수 실행 결과

        Raises:
            ExecutionTimeoutError: 타임아웃 초과
            ExecutionError: 실행 실패
        """
        if not self._is_running or not self._executor:
            raise ExecutionError("ThreadExecutor is not running")

        self._submitted_count += 1
        start_time = time.time()

        loop = asyncio.get_event_loop()

        try:
            # 스레드 풀에서 실행
            future = loop.run_in_executor(
                self._executor,
                lambda: func(*args, **kwargs),
            )

            # 타임아웃 적용
            if timeout:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future

            self._completed_count += 1
            self._total_execution_time += time.time() - start_time

            return result

        except asyncio.TimeoutError:
            self._failed_count += 1
            raise ExecutionTimeoutError(
                f"Task execution timed out after {timeout}s"
            )

        except Exception as e:
            self._failed_count += 1
            raise ExecutionError(f"Task execution failed: {e}")

    async def submit_many(
        self,
        func: Callable[..., T],
        args_list: List[tuple],
        timeout: Optional[float] = None,
    ) -> List[T]:
        """여러 작업을 병렬로 제출합니다.

        Args:
            func: 실행할 함수
            args_list: 인자 튜플 리스트
            timeout: 각 작업의 타임아웃 (초)

        Returns:
            결과 리스트 (입력 순서 유지)
        """
        if not self._is_running or not self._executor:
            raise ExecutionError("ThreadExecutor is not running")

        tasks = [
            self.submit(func, *args, timeout=timeout)
            for args in args_list
        ]

        # 모든 작업 병렬 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 예외 처리
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Task failed in batch",
                    task_index=i,
                    error=str(result),
                )
                processed_results.append(None)
            else:
                processed_results.append(result)

        return processed_results

    def get_stats(self) -> dict:
        """실행기 통계를 반환합니다.

        Returns:
            통계 딕셔너리
        """
        avg_time = 0.0
        if self._completed_count > 0:
            avg_time = self._total_execution_time / self._completed_count

        return {
            "type": "thread",
            "pool_size": self._pool_size,
            "is_running": self._is_running,
            "submitted_count": self._submitted_count,
            "completed_count": self._completed_count,
            "failed_count": self._failed_count,
            "success_rate": (
                self._completed_count / self._submitted_count
                if self._submitted_count > 0
                else 0.0
            ),
            "avg_execution_time_ms": avg_time * 1000,
        }
