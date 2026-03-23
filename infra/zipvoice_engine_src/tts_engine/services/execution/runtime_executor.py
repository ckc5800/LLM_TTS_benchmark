# -*- coding: utf-8 -*-
"""Runtime-specific Executor - 런타임별 최적화된 실행기.

이 모듈은 하위 모듈들을 통합하여 기존 API와의 호환성을 유지합니다.
"""

# 베이스 클래스와 데이터 클래스
from tts_engine.services.execution.base import (
    RuntimeExecutor,
    WorkerTask,
    WorkerResult,
)

# 각 런타임별 실행기
from tts_engine.services.execution.pytorch_executor import PyTorchExecutor
from tts_engine.services.execution.onnx_executor import ONNXExecutor
from tts_engine.services.execution.tensorrt_executor import TensorRTExecutor

# 팩토리
from tts_engine.services.execution.factory import RuntimeExecutorFactory


__all__ = [
    "RuntimeExecutor",
    "WorkerTask",
    "WorkerResult",
    "PyTorchExecutor",
    "ONNXExecutor",
    "TensorRTExecutor",
    "RuntimeExecutorFactory",
]
