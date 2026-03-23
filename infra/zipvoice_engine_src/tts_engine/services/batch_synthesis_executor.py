# -*- coding: utf-8 -*-
"""배치 합성 실행기 - 런타임별 배치 합성 처리.

DynamicBatcher의 배치 합성 핸들러를 런타임별로 구현합니다:
- PyTorch: 멀티프로세스 워커 풀
- ONNX: 세션 풀 + 스레드 풀
- TensorRT: GPU 세마포어 + 스레드 풀
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from tts_engine.core.constants import RuntimeType
from tts_engine.core.logging import get_logger
from tts_engine.domain.synthesis import SynthesisRequest, SynthesisResult, SynthesisStatus

if TYPE_CHECKING:
    from tts_engine.core.config import Settings
    from tts_engine.services.model_manager import ModelManager
    from tts_engine.services.voice_manager import VoiceManager
    from tts_engine.services.execution import RuntimeExecutor

logger = get_logger(__name__)


class BatchSynthesisExecutor:
    """배치 합성 실행기.

    런타임 타입에 따라 최적화된 배치 합성을 수행합니다.
    """

    def __init__(
        self,
        settings: "Settings",
        model_manager: "ModelManager",
        voice_manager: "VoiceManager",
        runtime_executor: Optional["RuntimeExecutor"] = None,
        thread_executor: Optional[ThreadPoolExecutor] = None,
        runtime_type: Optional[RuntimeType] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """배치 합성 실행기 초기화.

        Args:
            settings: 애플리케이션 설정
            model_manager: 모델 매니저
            voice_manager: 음성 매니저
            runtime_executor: 런타임 실행기
            thread_executor: 스레드 풀 실행기
            runtime_type: 런타임 타입
            model_config: 모델 설정
        """
        self._settings = settings
        self._model_manager = model_manager
        self._voice_manager = voice_manager
        self._runtime_executor = runtime_executor
        self._executor = thread_executor
        self._runtime_type = runtime_type
        self._model_config = model_config

    def update_executor(
        self,
        runtime_executor: Optional["RuntimeExecutor"],
        thread_executor: Optional[ThreadPoolExecutor],
        runtime_type: Optional[RuntimeType],
        model_config: Optional[Dict[str, Any]],
    ) -> None:
        """실행기 업데이트."""
        self._runtime_executor = runtime_executor
        self._executor = thread_executor
        self._runtime_type = runtime_type
        self._model_config = model_config

    async def handle_batch(
        self,
        requests: List[SynthesisRequest],
    ) -> List[SynthesisResult]:
        """DynamicBatcher의 배치 합성 핸들러.

        런타임별로 최적화된 실행 전략을 사용합니다.

        Args:
            requests: 배치로 모인 합성 요청들

        Returns:
            합성 결과 목록
        """
        if not requests:
            return []

        # PyTorch 런타임: 멀티프로세스 실행기 사용
        if self._runtime_type == RuntimeType.PYTORCH and self._runtime_executor:
            return await self._batch_synthesize_pytorch(requests)

        # ONNX 런타임: 세션 풀 + 스레드 풀 사용
        if self._runtime_type == RuntimeType.ONNX and self._runtime_executor:
            return await self._batch_synthesize_onnx(requests)

        # TensorRT 런타임: TensorRTExecutor + GPU 세마포어 사용
        if self._runtime_type == RuntimeType.TENSORRT and self._runtime_executor:
            return await self._batch_synthesize_tensorrt(requests)

        # 기타 런타임: 기존 ThreadPoolExecutor 사용
        return await self._batch_synthesize_threadpool(requests)

    async def _batch_synthesize_pytorch(
        self,
        requests: List[SynthesisRequest],
    ) -> List[SynthesisResult]:
        """PyTorch 멀티프로세스 실행기로 배치 합성."""
        all_results: Dict[str, SynthesisResult] = {}
        pool_timeout = self._settings.performance.execution.task_timeout_seconds

        # 각 요청을 워커 태스크로 변환
        tasks = []
        for req in requests:
            try:
                voice = self._voice_manager.get_voice_or_fallback(req.voice_id)

                task_data = {
                    "task_id": req.request_id,
                    "text": req.text,
                    "voice_id": req.voice_id,
                    "prompt_wav_path": voice.prompt_wav,
                    "prompt_text": voice.prompt_text,
                    "speed": req.speed or 1.0,
                    "num_steps": self._model_config.get("num_steps", 16) if self._model_config else 16,
                    "t_shift": self._model_config.get("t_shift", 0.5) if self._model_config else 0.5,
                    "guidance_scale": self._model_config.get("guidance_scale", 1.0) if self._model_config else 1.0,
                }
                tasks.append((req, task_data))

            except Exception as e:
                logger.error(
                    "Failed to prepare PyTorch task",
                    request_id=req.request_id,
                    error=str(e),
                )
                all_results[req.request_id] = SynthesisResult(
                    request_id=req.request_id,
                    status=SynthesisStatus.FAILED,
                    error_message=str(e),
                )

        # 멀티프로세스 실행기로 병렬 실행
        if tasks and self._runtime_executor:
            async def submit_task(req: SynthesisRequest, task_data: dict) -> tuple:
                try:
                    result_data = await self._runtime_executor.submit(
                        lambda: None,  # 실제 함수는 워커에서 실행
                        timeout=pool_timeout,
                        request_data=task_data,
                    )
                    return (req.request_id, SynthesisResult(
                        request_id=req.request_id,
                        status=SynthesisStatus.COMPLETED,
                        audio_data=result_data.get("audio_data"),
                        sample_rate=result_data.get("sample_rate", 24000),
                        processing_time_ms=result_data.get("processing_time_ms", 0),
                    ))
                except Exception as e:
                    return (req.request_id, SynthesisResult(
                        request_id=req.request_id,
                        status=SynthesisStatus.FAILED,
                        error_message=str(e),
                    ))

            # 모든 태스크 병렬 실행
            results = await asyncio.gather(
                *[submit_task(req, task_data) for req, task_data in tasks],
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, tuple):
                    request_id, synth_result = result
                    all_results[request_id] = synth_result
                elif isinstance(result, Exception):
                    logger.error("PyTorch batch error", error=str(result))

        logger.debug(
            "PyTorch multiprocess batch completed",
            total_requests=len(requests),
            completed=len(all_results),
        )

        return self._return_ordered_results(requests, all_results)

    async def _batch_synthesize_onnx(
        self,
        requests: List[SynthesisRequest],
    ) -> List[SynthesisResult]:
        """ONNX 세션 풀 + 스레드 풀로 배치 합성."""
        all_results: Dict[str, SynthesisResult] = {}
        pool_timeout = self._settings.performance.execution.task_timeout_seconds

        # 각 요청을 ONNX 태스크로 변환
        tasks = []
        for req in requests:
            try:
                voice = self._voice_manager.get_voice_or_fallback(req.voice_id)

                task_data = {
                    "task_id": req.request_id,
                    "text": req.text,
                    "voice_id": req.voice_id,
                    "prompt_wav_path": voice.prompt_wav,
                    "prompt_text": voice.prompt_text,
                    "speed": req.speed or 1.0,
                    "num_steps": self._model_config.get("num_steps", 16) if self._model_config else 16,
                    "t_shift": self._model_config.get("t_shift", 0.5) if self._model_config else 0.5,
                    "guidance_scale": self._model_config.get("guidance_scale", 1.0) if self._model_config else 1.0,
                }
                tasks.append((req, task_data))

            except Exception as e:
                logger.error(
                    "Failed to prepare ONNX task",
                    request_id=req.request_id,
                    error=str(e),
                )
                all_results[req.request_id] = SynthesisResult(
                    request_id=req.request_id,
                    status=SynthesisStatus.FAILED,
                    error_message=str(e),
                )

        # ONNX 세션 풀 실행기로 병렬 실행
        if tasks and self._runtime_executor:
            async def submit_task(req: SynthesisRequest, task_data: dict) -> tuple:
                try:
                    result_data = await self._runtime_executor.submit(
                        lambda: None,
                        timeout=pool_timeout,
                        request_data=task_data,
                    )
                    return (req.request_id, SynthesisResult(
                        request_id=req.request_id,
                        status=SynthesisStatus.COMPLETED,
                        audio_data=result_data.get("audio_data"),
                        sample_rate=result_data.get("sample_rate", 24000),
                        processing_time_ms=result_data.get("processing_time_ms", 0),
                    ))
                except Exception as e:
                    return (req.request_id, SynthesisResult(
                        request_id=req.request_id,
                        status=SynthesisStatus.FAILED,
                        error_message=str(e),
                    ))

            results = await asyncio.gather(
                *[submit_task(req, task_data) for req, task_data in tasks],
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, tuple):
                    request_id, synth_result = result
                    all_results[request_id] = synth_result
                elif isinstance(result, Exception):
                    logger.error("ONNX batch error", error=str(result))

        logger.debug(
            "ONNX session pool batch completed",
            total_requests=len(requests),
            completed=len(all_results),
        )

        return self._return_ordered_results(requests, all_results)

    async def _batch_synthesize_tensorrt(
        self,
        requests: List[SynthesisRequest],
    ) -> List[SynthesisResult]:
        """TensorRT 세마포어 + 스레드 풀로 배치 합성."""
        all_results: Dict[str, SynthesisResult] = {}
        pool_timeout = self._settings.performance.execution.task_timeout_seconds

        # 음성별 모델 인스턴스 매핑
        request_instances: List[tuple] = []
        for req in requests:
            try:
                voice = self._voice_manager.get_voice_or_fallback(req.voice_id)
                instance_name = voice.model_instance
                request_instances.append((req, instance_name))
            except Exception as e:
                logger.error(
                    "Failed to get voice for TensorRT batch",
                    voice_id=req.voice_id,
                    error=str(e),
                )
                all_results[req.request_id] = SynthesisResult(
                    request_id=req.request_id,
                    status=SynthesisStatus.FAILED,
                    error_message=f"Voice not found: {e}",
                )

        model_manager = self._model_manager
        runtime_executor = self._runtime_executor

        async def submit_request(req: SynthesisRequest, instance_name: str) -> SynthesisResult:
            """단일 요청을 TensorRTExecutor에 제출."""
            task_data = {"task_id": req.request_id}

            def _run_synthesis():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    async def _synthesize():
                        async with model_manager.get_model(instance_name, timeout=pool_timeout / 2) as model:
                            return await model.synthesize(req)
                    return loop.run_until_complete(_synthesize())
                finally:
                    loop.close()

            try:
                result = await runtime_executor.submit(
                    _run_synthesis,
                    timeout=pool_timeout,
                    request_data=task_data,
                )
                return result
            except Exception as e:
                logger.error(
                    "TensorRT synthesis failed",
                    request_id=req.request_id,
                    error=str(e),
                )
                return SynthesisResult(
                    request_id=req.request_id,
                    status=SynthesisStatus.FAILED,
                    error_message=str(e),
                )

        # 모든 요청을 병렬로 제출
        tasks = [submit_request(req, inst) for req, inst in request_instances]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (req, _), result in zip(request_instances, results):
            if isinstance(result, Exception):
                all_results[req.request_id] = SynthesisResult(
                    request_id=req.request_id,
                    status=SynthesisStatus.FAILED,
                    error_message=str(result),
                )
            else:
                all_results[req.request_id] = result

        logger.debug(
            "TensorRT batch completed",
            total_requests=len(requests),
            completed=len(all_results),
        )

        return self._return_ordered_results(requests, all_results)

    async def _batch_synthesize_threadpool(
        self,
        requests: List[SynthesisRequest],
    ) -> List[SynthesisResult]:
        """ThreadPoolExecutor로 배치 합성 (레거시)."""
        effective_batch_size = 2

        # 음성별로 모델 인스턴스 그룹화
        instance_groups: Dict[str, List[SynthesisRequest]] = {}

        for req in requests:
            try:
                voice = self._voice_manager.get_voice_or_fallback(req.voice_id)
                instance_name = voice.model_instance
                if instance_name not in instance_groups:
                    instance_groups[instance_name] = []
                instance_groups[instance_name].append(req)
            except Exception as e:
                logger.error(
                    "Failed to get voice for batch",
                    voice_id=req.voice_id,
                    error=str(e),
                )

        all_results: Dict[str, SynthesisResult] = {}
        pool_timeout = self._settings.performance.execution.task_timeout_seconds / 2
        model_manager = self._model_manager

        def process_sub_batch_sync(
            instance_name: str,
            sub_batch: List[SynthesisRequest],
        ) -> List[tuple]:
            """별도 스레드에서 sub-batch 처리."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    async def _run():
                        async with model_manager.get_model(instance_name, timeout=pool_timeout) as model:
                            return await model.synthesize_batch(sub_batch)

                    results = loop.run_until_complete(_run())
                    return [(req.request_id, result) for req, result in zip(sub_batch, results)]
                finally:
                    loop.close()
            except Exception as e:
                logger.error(
                    "Sub-batch synthesis failed",
                    instance_name=instance_name,
                    batch_size=len(sub_batch),
                    error=str(e),
                )
                return [(req.request_id, SynthesisResult(
                    request_id=req.request_id,
                    status=SynthesisStatus.FAILED,
                    error_message=str(e),
                )) for req in sub_batch]

        # sub-batch 목록 생성
        sub_batches = []
        for instance_name, group_requests in instance_groups.items():
            for i in range(0, len(group_requests), effective_batch_size):
                sub_batch = group_requests[i:i + effective_batch_size]
                sub_batches.append((instance_name, sub_batch))

        logger.debug(
            "ThreadPool parallel dispatch",
            total_requests=len(requests),
            num_sub_batches=len(sub_batches),
            effective_batch_size=effective_batch_size,
            runtime_type=self._runtime_type.value if self._runtime_type else "unknown",
        )

        # ThreadPoolExecutor로 병렬 실행
        if sub_batches and self._executor:
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(
                    self._executor,
                    process_sub_batch_sync,
                    instance_name,
                    sub_batch,
                )
                for instance_name, sub_batch in sub_batches
            ]

            batch_results = await asyncio.gather(*futures, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, list):
                    for request_id, synth_result in result:
                        all_results[request_id] = synth_result
                elif isinstance(result, Exception):
                    logger.error("ThreadPool batch error", error=str(result))

        return self._return_ordered_results(requests, all_results)

    def _return_ordered_results(
        self,
        requests: List[SynthesisRequest],
        results: Dict[str, SynthesisResult],
    ) -> List[SynthesisResult]:
        """원래 순서대로 결과 반환."""
        return [results.get(req.request_id, SynthesisResult(
            request_id=req.request_id,
            status=SynthesisStatus.FAILED,
            error_message="Result not found",
        )) for req in requests]
