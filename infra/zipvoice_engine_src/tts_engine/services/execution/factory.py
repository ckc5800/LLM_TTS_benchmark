# -*- coding: utf-8 -*-
"""런타임별 실행기 팩토리."""

from typing import Optional

from tts_engine.core.constants import RuntimeType
from tts_engine.services.execution.base import RuntimeExecutor
from tts_engine.services.execution.pytorch_executor import PyTorchExecutor
from tts_engine.services.execution.onnx_executor import ONNXExecutor
from tts_engine.services.execution.tensorrt_executor import TensorRTExecutor


class RuntimeExecutorFactory:
    """런타임별 실행기 팩토리."""

    @staticmethod
    def create(
        runtime_type: RuntimeType,
        pool_size: int = 4,
        model_config: Optional[dict] = None,
    ) -> RuntimeExecutor:
        """런타임 타입에 맞는 실행기를 생성합니다."""
        if runtime_type == RuntimeType.PYTORCH:
            return PyTorchExecutor(pool_size=pool_size, model_config=model_config)
        elif runtime_type == RuntimeType.ONNX:
            return ONNXExecutor(pool_size=pool_size, model_config=model_config)
        elif runtime_type == RuntimeType.TENSORRT:
            return TensorRTExecutor(pool_size=pool_size, model_config=model_config)
        else:
            raise ValueError(f"Unknown runtime type: {runtime_type}")
