# -*- coding: utf-8 -*-
"""Executor Base - 실행기 추상 클래스.

TTS 합성 작업 실행을 위한 추상 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, TypeVar

from tts_engine.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class ExecutorBase(ABC):
    """실행기 추상 베이스 클래스.

    다양한 실행 전략(스레드, 프로세스)을 통일된 인터페이스로 제공합니다.

    Attributes:
        _pool_size: 풀 크기
        _is_running: 실행 상태
    """

    def __init__(self, pool_size: int = 4):
        """실행기 초기화.

        Args:
            pool_size: 워커 풀 크기
        """
        self._pool_size = pool_size
        self._is_running = False

    @abstractmethod
    async def start(self) -> None:
        """실행기를 시작합니다."""
        pass

    @abstractmethod
    async def stop(self, wait: bool = True) -> None:
        """실행기를 중지합니다.

        Args:
            wait: 진행 중인 작업 완료 대기 여부
        """
        pass

    @abstractmethod
    async def submit(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """작업을 제출하고 결과를 대기합니다.

        Args:
            func: 실행할 함수
            *args: 위치 인자
            timeout: 타임아웃 (초)
            **kwargs: 키워드 인자

        Returns:
            함수 실행 결과

        Raises:
            TimeoutError: 타임아웃 초과
            ExecutionError: 실행 실패
        """
        pass

    @abstractmethod
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
        pass

    @property
    def is_running(self) -> bool:
        """실행 상태."""
        return self._is_running

    @property
    def pool_size(self) -> int:
        """풀 크기."""
        return self._pool_size

    @abstractmethod
    def get_stats(self) -> dict:
        """실행기 통계를 반환합니다.

        Returns:
            통계 딕셔너리
        """
        pass

    async def __aenter__(self) -> "ExecutorBase":
        """비동기 컨텍스트 매니저 진입."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """비동기 컨텍스트 매니저 종료."""
        await self.stop()
