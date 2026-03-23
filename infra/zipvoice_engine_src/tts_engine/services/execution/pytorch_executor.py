# -*- coding: utf-8 -*-
"""PyTorch 멀티프로세스 실행기."""

import asyncio
import multiprocessing as mp
import queue
import time
import uuid
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, List, Optional

from tts_engine.core.constants import RuntimeType
from tts_engine.core.exceptions import ExecutionError, ExecutionTimeoutError
from tts_engine.core.logging import get_logger
from tts_engine.services.execution.base import QueuedRuntimeExecutor, WorkerTask

logger = get_logger(__name__)


def _pytorch_worker_process(
    worker_id: int,
    task_queue: Queue,
    result_queue: Queue,
    model_config: dict,
    shutdown_event: mp.Event,
):
    """PyTorch 워커 프로세스 메인 함수."""
    from tts_engine.services.execution.pytorch_worker import pytorch_worker_main
    pytorch_worker_main(
        worker_id=worker_id,
        task_queue=task_queue,
        result_queue=result_queue,
        model_config=model_config,
        shutdown_event=shutdown_event,
    )


class PyTorchExecutor(QueuedRuntimeExecutor):
    """PyTorch 멀티프로세스 실행기.

    내장 큐로 요청을 관리하고, 멀티프로세스 워커로 실행합니다.
    """

    def __init__(
        self,
        pool_size: int = 4,
        model_config: Optional[dict] = None,
        max_queue_size: int = 500,
        queue_timeout: float = 30.0,
    ):
        self._model_config = model_config or {}

        super().__init__(
            runtime_type=RuntimeType.PYTORCH,
            pool_size=pool_size,
            max_queue_size=max_queue_size,
            queue_timeout=queue_timeout,
        )

        # 멀티프로세스 관련
        self._workers: List[Process] = []
        self._task_queues: List[Queue] = []
        self._result_queue: Optional[Queue] = None
        self._shutdown_event: Optional[mp.Event] = None
        self._pending_tasks: Dict[str, asyncio.Future] = {}
        self._result_handler_task: Optional[asyncio.Task] = None
        self._current_worker_idx = 0

        # 통계
        self._submitted_count = 0
        self._completed_count = 0
        self._failed_count = 0

    async def _start_runtime(self) -> None:
        """PyTorch 멀티프로세스 워커 풀 시작."""
        logger.info("Starting PyTorch runtime", pool_size=self._pool_size)

        ctx = mp.get_context("spawn")
        self._result_queue = ctx.Queue()
        self._shutdown_event = ctx.Event()

        for i in range(self._pool_size):
            task_queue = ctx.Queue()
            self._task_queues.append(task_queue)

            worker = ctx.Process(
                target=_pytorch_worker_process,
                args=(i, task_queue, self._result_queue, self._model_config, self._shutdown_event),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        # 워커 초기화 대기
        initialized = 0
        timeout = 300
        start_time = time.time()

        while initialized < self._pool_size:
            if time.time() - start_time > timeout:
                raise ExecutionError(f"Worker initialization timeout: {initialized}/{self._pool_size}")
            try:
                result = self._result_queue.get(timeout=5.0)
                if result.task_id == "__init__":
                    if result.success:
                        initialized += 1
                        logger.info(f"Worker initialized: {initialized}/{self._pool_size}")
                    else:
                        raise ExecutionError(f"Worker failed: {result.error_message}")
            except queue.Empty:
                continue

        self._result_handler_task = asyncio.create_task(self._handle_results())
        logger.info("PyTorch runtime started", workers=self._pool_size)

    async def _stop_runtime(self) -> None:
        """PyTorch 멀티프로세스 워커 풀 중지."""
        logger.info("Stopping PyTorch runtime")

        if self._result_handler_task:
            self._result_handler_task.cancel()
            try:
                await self._result_handler_task
            except asyncio.CancelledError:
                pass

        if self._shutdown_event:
            self._shutdown_event.set()

        for q in self._task_queues:
            try:
                q.put(None)
            except:
                pass

        for worker in self._workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()

        self._workers.clear()
        self._task_queues.clear()
        logger.info("PyTorch runtime stopped")

    async def _handle_results(self) -> None:
        """결과 큐에서 결과를 수신합니다."""
        while self._is_running:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._result_queue.get(timeout=0.1) if self._result_queue else None,
                )

                if result is None:
                    continue

                if result.task_id in self._pending_tasks:
                    future = self._pending_tasks.pop(result.task_id)
                    if result.success:
                        result_data = {
                            "audio_data": result.audio_data,
                            "sample_rate": result.sample_rate,
                            "processing_time_ms": result.processing_time_ms,
                        }
                        future.set_result(result_data)
                        self._completed_count += 1
                    else:
                        future.set_exception(ExecutionError(result.error_message))
                        self._failed_count += 1

            except queue.Empty:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in result handler", error=str(e))

    async def _execute(self, func: Callable, *args, **kwargs) -> Any:
        """멀티프로세스 워커에게 작업을 전달합니다."""
        request_data = kwargs.get("request_data", {})
        task_id = request_data.get("task_id", str(uuid.uuid4()))

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending_tasks[task_id] = future

        worker_idx = self._current_worker_idx
        self._current_worker_idx = (self._current_worker_idx + 1) % self._pool_size

        task = WorkerTask(
            task_id=task_id,
            text=request_data.get("text", ""),
            voice_id=request_data.get("voice_id", ""),
            prompt_wav_path=request_data.get("prompt_wav_path", ""),
            prompt_text=request_data.get("prompt_text", ""),
            speed=request_data.get("speed", 1.0),
            num_steps=request_data.get("num_steps", 16),
            t_shift=request_data.get("t_shift", 0.5),
            guidance_scale=request_data.get("guidance_scale", 1.0),
        )
        self._task_queues[worker_idx].put(task)
        self._submitted_count += 1

        # 결과 대기 (타임아웃 없이 - 큐 시스템에서 타임아웃 처리)
        return await future

    def get_stats(self) -> dict:
        workers_alive = sum(1 for w in self._workers if w.is_alive())
        pending = len(self._pending_tasks)
        active = self._submitted_count - self._completed_count - self._failed_count

        queue_status = self._request_queue.get_status()

        return {
            "type": "pytorch_multiprocess",
            "runtime": self._runtime_type.value,
            "pool_size": self._pool_size,
            "is_running": self._is_running,
            "workers_alive": workers_alive,
            "submitted_count": self._submitted_count,
            "completed_count": self._completed_count,
            "failed_count": self._failed_count,
            "pending_count": pending,
            # 실시간 상태
            "active_count": max(0, active),
            "available_slots": max(0, workers_alive - active),
            # 큐 상태
            "queue_size": queue_status["queue_size"],
            "queue_rejected": queue_status["total_rejected"],
            "queue_timeout": queue_status["total_timeout"],
        }

    def get_pool_status(self) -> dict:
        """실시간 풀 상태를 반환합니다."""
        workers_alive = sum(1 for w in self._workers if w.is_alive())
        pending = len(self._pending_tasks)
        active = self._submitted_count - self._completed_count - self._failed_count

        queue_status = self._request_queue.get_status()

        return {
            "runtime": "pytorch",
            "total_slots": self._pool_size,
            "workers_alive": workers_alive,
            "active": max(0, active),
            "waiting": pending,
            "available": max(0, workers_alive - active),
            # 큐 상태
            "queue_size": queue_status["queue_size"],
            "max_queue_size": queue_status["max_queue_size"],
            "queue_utilization": queue_status["queue_utilization"],
            "queue_rejected": queue_status["total_rejected"],
            "queue_timeout": queue_status["total_timeout"],
        }
