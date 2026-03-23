# -*- coding: utf-8 -*-
"""합성 실행기 - 런타임별 단일 합성 실행.

ONNX, TensorRT, PyTorch 런타임별로 최적화된 합성 실행을 제공합니다.
"""

import asyncio
from typing import Any, Dict, Optional, TYPE_CHECKING

from tts_engine.core.constants import AudioFormat, RuntimeType
from tts_engine.core.exceptions import SynthesisError
from tts_engine.core.logging import get_logger
from tts_engine.domain.synthesis import SynthesisRequest, SynthesisResult, SynthesisStatus

if TYPE_CHECKING:
    from tts_engine.core.config import Settings
    from tts_engine.services.model_manager import ModelManager
    from tts_engine.services.voice_manager import VoiceManager
    from tts_engine.services.execution import RuntimeExecutor

logger = get_logger(__name__)


class SynthesisExecutor:
    """런타임별 단일 합성 실행기.

    ONNX, TensorRT, PyTorch 런타임에 최적화된 합성 실행을 제공합니다.
    """

    def __init__(
        self,
        settings: "Settings",
        model_manager: "ModelManager",
        voice_manager: "VoiceManager",
    ):
        """합성 실행기 초기화.

        Args:
            settings: 애플리케이션 설정
            model_manager: 모델 매니저
            voice_manager: 음성 매니저
        """
        self._settings = settings
        self._model_manager = model_manager
        self._voice_manager = voice_manager
        self._runtime_executor: Optional["RuntimeExecutor"] = None
        self._runtime_type: Optional[RuntimeType] = None
        self._model_config: Optional[Dict[str, Any]] = None

    def update_executor(
        self,
        runtime_executor: Optional["RuntimeExecutor"],
        runtime_type: Optional[RuntimeType],
        model_config: Optional[Dict[str, Any]],
    ) -> None:
        """실행기 업데이트."""
        self._runtime_executor = runtime_executor
        self._runtime_type = runtime_type
        self._model_config = model_config

    async def execute(
        self,
        request: SynthesisRequest,
        instance_name: str,
    ) -> SynthesisResult:
        """풀에서 모델을 가져와 합성을 실행합니다.

        런타임별 최적화된 실행 전략을 사용합니다.

        Args:
            request: 합성 요청
            instance_name: 모델 인스턴스 이름

        Returns:
            SynthesisResult
        """
        import time
        start_time = time.time()
        request_id = request.request_id or "unknown"
        pool_timeout = self._settings.performance.execution.task_timeout_seconds / 2

        runtime_name = self._runtime_type.value if self._runtime_type else "pytorch"
        logger.debug(
            f"[{request_id}] Executor: Starting synthesis",
            runtime=runtime_name,
            instance_name=instance_name,
        )

        # ONNX 런타임
        if self._runtime_type == RuntimeType.ONNX and self._runtime_executor:
            result = await self._execute_onnx(request)
            logger.debug(
                f"[{request_id}] Executor: ONNX synthesis completed",
                elapsed_ms=int((time.time() - start_time) * 1000),
            )
            return result

        # TensorRT 런타임
        if self._runtime_type == RuntimeType.TENSORRT and self._runtime_executor:
            result = await self._execute_tensorrt(request, instance_name)
            logger.debug(
                f"[{request_id}] Executor: TensorRT synthesis completed",
                elapsed_ms=int((time.time() - start_time) * 1000),
            )
            return result

        # PyTorch/기본: ModelManager 모델 풀 사용
        async with self._model_manager.get_model(instance_name, timeout=pool_timeout) as model:
            result = await model.synthesize(request)
            logger.debug(
                f"[{request_id}] Executor: PyTorch synthesis completed",
                elapsed_ms=int((time.time() - start_time) * 1000),
            )
            return result

    async def _execute_onnx(
        self,
        request: SynthesisRequest,
    ) -> SynthesisResult:
        """ONNX 세션 풀로 단일 합성을 실행합니다."""
        pool_timeout = self._settings.performance.execution.task_timeout_seconds

        # 음성 정보 조회
        voice = self._voice_manager.get_voice_or_fallback(request.voice_id)

        # 태스크 데이터 생성
        task_data = {
            "task_id": request.request_id,
            "text": request.text,
            "voice_id": request.voice_id,
            "prompt_wav_path": voice.prompt_wav,
            "prompt_text": voice.prompt_text,
            "speed": request.speed or 1.0,
            "num_steps": self._model_config.get("num_steps", 16) if self._model_config else 16,
            "t_shift": self._model_config.get("t_shift", 0.5) if self._model_config else 0.5,
            "guidance_scale": self._model_config.get("guidance_scale", 1.0) if self._model_config else 1.0,
        }

        # ONNXExecutor를 통해 합성 실행
        result_data = await self._runtime_executor.submit(
            lambda: None,
            timeout=pool_timeout,
            request_data=task_data,
        )

        # 결과 변환
        if result_data.get("success"):
            return SynthesisResult(
                request_id=request.request_id,
                status=SynthesisStatus.COMPLETED,
                audio_data=result_data.get("audio_data"),
                sample_rate=result_data.get("sample_rate", 24000),
                format=AudioFormat.WAV,
                processing_time_ms=result_data.get("processing_time_ms", 0),
            )
        else:
            raise SynthesisError(
                f"ONNX synthesis failed: {result_data.get('error_message', 'Unknown error')}",
                request_id=request.request_id,
            )

    async def _execute_tensorrt(
        self,
        request: SynthesisRequest,
        instance_name: str,
    ) -> SynthesisResult:
        """TensorRT로 단일 합성을 실행합니다."""
        pool_timeout = self._settings.performance.execution.task_timeout_seconds
        model_manager = self._model_manager
        runtime_executor = self._runtime_executor

        task_data = {"task_id": request.request_id}

        async def _synthesize():
            async with model_manager.get_model(instance_name, timeout=pool_timeout / 2) as model:
                return await model.synthesize(request)

        def _run_synthesis():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(_synthesize())
            finally:
                loop.close()

        result = await runtime_executor.submit(
            _run_synthesis,
            timeout=pool_timeout,
            request_data=task_data,
        )

        return result
