# -*- coding: utf-8 -*-
"""Process Executor - 프로세스 풀 기반 실행기.

멀티프로세싱을 활용한 CPU 바운드 작업 병렬 처리를 제공합니다.
GIL 우회가 필요한 환경에서 사용합니다.
"""

import asyncio
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, List, Optional, TypeVar

from tts_engine.core.exceptions import ExecutionError, ExecutionTimeoutError
from tts_engine.core.logging import get_logger
from tts_engine.services.execution.executor_base import ExecutorBase

logger = get_logger(__name__)

T = TypeVar("T")


def _worker_initializer():
    """워커 프로세스 초기화 함수."""
    # GPU 컨텍스트 등 프로세스별 초기화가 필요한 경우 여기서 수행
    pass


class ProcessExecutor(ExecutorBase):
    """프로세스 풀 기반 실행기.

    CPU 바운드 작업이나 GIL 우회가 필요한 경우 사용합니다.
    GPU 작업의 경우 프로세스 간 메모리 복사 오버헤드에 주의해야 합니다.

    Attributes:
        _executor: ProcessPoolExecutor 인스턴스
        _mp_context: 멀티프로세싱 컨텍스트
    """

    def __init__(
        self,
        pool_size: int = 4,
        mp_context: Optional[str] = None,
    ):
        """프로세스 실행기 초기화.

        Args:
            pool_size: 프로세스 풀 크기 (기본: 4)
            mp_context: 멀티프로세싱 컨텍스트 ('spawn', 'fork', 'forkserver')
        """
        super().__init__(pool_size=pool_size)
        self._executor: Optional[ProcessPoolExecutor] = None
        self._mp_context = mp_context or "spawn"  # CUDA 호환성을 위해 spawn 기본값
        self._submitted_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._total_execution_time = 0.0

        logger.info(
            "ProcessExecutor initialized",
            pool_size=pool_size,
            mp_context=self._mp_context,
        )

    async def start(self) -> None:
        """프로세스 풀을 시작합니다."""
        if self._is_running:
            logger.warning("ProcessExecutor is already running")
            return

        ctx = mp.get_context(self._mp_context)
        self._executor = ProcessPoolExecutor(
            max_workers=self._pool_size,
            mp_context=ctx,
            initializer=_worker_initializer,
        )
        self._is_running = True

        logger.info(
            "ProcessExecutor started",
            pool_size=self._pool_size,
            mp_context=self._mp_context,
        )

    async def stop(self, wait: bool = True) -> None:
        """프로세스 풀을 중지합니다.

        Args:
            wait: 진행 중인 작업 완료 대기 여부
        """
        if not self._is_running or not self._executor:
            logger.warning("ProcessExecutor is not running")
            return

        logger.info("Stopping ProcessExecutor", wait=wait)

        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        self._executor = None
        self._is_running = False

        logger.info(
            "ProcessExecutor stopped",
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
        """작업을 프로세스 풀에 제출합니다.

        Args:
            func: 실행할 함수 (pickle 가능해야 함)
            *args: 위치 인자 (pickle 가능해야 함)
            timeout: 타임아웃 (초)
            **kwargs: 키워드 인자 (pickle 가능해야 함)

        Returns:
            함수 실행 결과

        Raises:
            ExecutionTimeoutError: 타임아웃 초과
            ExecutionError: 실행 실패
        """
        if not self._is_running or not self._executor:
            raise ExecutionError("ProcessExecutor is not running")

        self._submitted_count += 1
        start_time = time.time()

        loop = asyncio.get_event_loop()

        try:
            # kwargs를 포함하는 래퍼 함수 사용
            if kwargs:
                # kwargs가 있는 경우 래퍼 사용
                wrapper = _FunctionWrapper(func, kwargs)
                future = loop.run_in_executor(
                    self._executor,
                    wrapper,
                    *args,
                )
            else:
                future = loop.run_in_executor(
                    self._executor,
                    func,
                    *args,
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
            raise ExecutionError("ProcessExecutor is not running")

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
            "type": "process",
            "pool_size": self._pool_size,
            "mp_context": self._mp_context,
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


class _FunctionWrapper:
    """kwargs를 지원하기 위한 함수 래퍼 (pickle 가능)."""

    def __init__(self, func: Callable, kwargs: dict):
        self.func = func
        self.kwargs = kwargs

    def __call__(self, *args):
        return self.func(*args, **self.kwargs)
