# -*- coding: utf-8 -*-
"""AICESS TTS Engine - 고성능 ZipVoice 기반 TTS 엔진.

NVIDIA DGX Spark (GB10 Blackwell, sm_121) 최적화:
- PyTorch, ONNX, TensorRT 멀티 런타임 지원
- 모델 풀 기반 고가용성 아키텍처
- gRPC 기반 서비스 인터페이스
"""

__version__ = "1.0.0"
__author__ = "AICESS Team"

from tts_engine.core.logging import get_logger, setup_logging

__all__ = [
    "__version__",
    "__author__",
    "setup_logging",
    "get_logger",
]
