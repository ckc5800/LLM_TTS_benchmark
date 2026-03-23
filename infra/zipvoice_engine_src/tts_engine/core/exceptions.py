"""Custom exceptions for TTS Engine."""

import json
from typing import Optional


class TTSEngineError(Exception):
    """TTS 엔진 기본 예외.

    모든 TTS 엔진 예외의 베이스 클래스입니다.
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            # JSON 형식으로 직렬화하여 Loguru 포맷 충돌 방지
            details_str = json.dumps(self.details, ensure_ascii=False)
            return f"{self.message} - {details_str}"
        return self.message


# ============================================================
# Configuration Errors
# ============================================================


class ConfigurationError(TTSEngineError):
    """설정 관련 오류.

    설정 파일 로드 실패, 유효성 검증 실패 등.
    """

    pass


class ConfigFileNotFoundError(ConfigurationError):
    """설정 파일을 찾을 수 없음."""

    def __init__(self, file_path: str):
        super().__init__(
            f"설정 파일을 찾을 수 없습니다: {file_path}", {"file_path": file_path}
        )


class ConfigValidationError(ConfigurationError):
    """설정 유효성 검증 실패."""

    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(f"설정 유효성 검증 실패: {message}", details)


# ============================================================
# Model Errors
# ============================================================


class ModelError(TTSEngineError):
    """모델 관련 오류."""

    pass


class ModelNotFoundError(ModelError):
    """모델을 찾을 수 없음."""

    def __init__(self, model_name: str, model_path: Optional[str] = None):
        details = {"model_name": model_name}
        if model_path:
            details["model_path"] = model_path
        super().__init__(f"모델을 찾을 수 없습니다: {model_name}", details)


class ModelLoadError(ModelError):
    """모델 로드 실패."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            f"모델 로드 실패: {model_name}", {"model_name": model_name, "reason": reason}
        )


class ModelInitializationError(ModelError):
    """모델 초기화 실패."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            f"모델 초기화 실패: {model_name}",
            {"model_name": model_name, "reason": reason},
        )


class ModelUnloadError(ModelError):
    """모델 언로드 실패."""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            f"모델 언로드 실패: {model_name}",
            {"model_name": model_name, "reason": reason},
        )


# ============================================================
# Runtime Errors
# ============================================================


class RuntimeError(TTSEngineError):
    """런타임 관련 오류."""

    pass


class RuntimeNotSupportedError(RuntimeError):
    """지원하지 않는 런타임."""

    def __init__(self, runtime: str, supported: list[str]):
        super().__init__(
            f"지원하지 않는 런타임: {runtime}",
            {"runtime": runtime, "supported": supported},
        )


class RuntimeInitializationError(RuntimeError):
    """런타임 초기화 실패."""

    def __init__(self, runtime: str, reason: str):
        super().__init__(
            f"런타임 초기화 실패: {runtime}", {"runtime": runtime, "reason": reason}
        )


class TensorRTEngineError(RuntimeError):
    """TensorRT 엔진 오류."""

    def __init__(self, message: str, engine_path: Optional[str] = None):
        details = {"engine_path": engine_path} if engine_path else {}
        super().__init__(f"TensorRT 오류: {message}", details)


class OnnxRuntimeError(RuntimeError):
    """ONNX Runtime 오류."""

    def __init__(self, message: str, provider: Optional[str] = None):
        details = {"provider": provider} if provider else {}
        super().__init__(f"ONNX Runtime 오류: {message}", details)


# ============================================================
# Synthesis Errors
# ============================================================


class SynthesisError(TTSEngineError):
    """합성 오류."""

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        voice_id: Optional[str] = None,
    ):
        details = {}
        if request_id:
            details["request_id"] = request_id
        if voice_id:
            details["voice_id"] = voice_id
        super().__init__(message, details)
        self.request_id = request_id
        self.voice_id = voice_id


class SynthesisTimeoutError(SynthesisError):
    """합성 타임아웃."""

    def __init__(self, timeout_seconds: float, request_id: Optional[str] = None):
        super().__init__(
            f"합성 타임아웃: {timeout_seconds}초 초과", request_id=request_id
        )
        self.timeout_seconds = timeout_seconds


class TextTooLongError(SynthesisError):
    """텍스트가 너무 김."""

    def __init__(self, text_length: int, max_length: int):
        super().__init__(
            f"텍스트가 너무 깁니다: {text_length}자 (최대: {max_length}자)"
        )
        self.details.update({"text_length": text_length, "max_length": max_length})


class EmptyTextError(SynthesisError):
    """빈 텍스트."""

    def __init__(self):
        super().__init__("합성할 텍스트가 비어있습니다")


# ============================================================
# Voice Errors
# ============================================================


class VoiceError(TTSEngineError):
    """음성 관련 오류."""

    pass


class VoiceNotFoundError(VoiceError):
    """음성을 찾을 수 없음."""

    def __init__(self, voice_id: str):
        super().__init__(f"음성을 찾을 수 없습니다: {voice_id}", {"voice_id": voice_id})


class VoiceLoadError(VoiceError):
    """음성 로드 실패."""

    def __init__(self, voice_id: str, reason: str):
        super().__init__(
            f"음성 로드 실패: {voice_id}", {"voice_id": voice_id, "reason": reason}
        )


class VoiceNotReadyError(VoiceError):
    """음성이 준비되지 않음."""

    def __init__(self, voice_id: str):
        super().__init__(
            f"음성이 아직 준비되지 않았습니다: {voice_id}", {"voice_id": voice_id}
        )


# ============================================================
# Resource Errors
# ============================================================


class ResourceError(TTSEngineError):
    """리소스 관련 오류."""

    pass


class GPUMemoryError(ResourceError):
    """GPU 메모리 부족."""

    def __init__(self, required_mb: float, available_mb: float):
        super().__init__(
            f"GPU 메모리 부족: 필요 {required_mb:.1f}MB, 가용 {available_mb:.1f}MB",
            {"required_mb": required_mb, "available_mb": available_mb},
        )


class PoolExhaustedError(ResourceError):
    """모델 풀 고갈."""

    def __init__(self, pool_name: str, pool_size: int):
        super().__init__(
            f"모델 풀이 고갈되었습니다: {pool_name} (크기: {pool_size})",
            {"pool_name": pool_name, "pool_size": pool_size},
        )


class DeviceNotAvailableError(ResourceError):
    """디바이스를 사용할 수 없음."""

    def __init__(self, device: str, reason: Optional[str] = None):
        message = f"디바이스를 사용할 수 없습니다: {device}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, {"device": device, "reason": reason})


# ============================================================
# Server Errors
# ============================================================


class ServerError(TTSEngineError):
    """서버 관련 오류."""

    pass


class ServerStartupError(ServerError):
    """서버 시작 실패."""

    def __init__(self, reason: str):
        super().__init__(f"서버 시작 실패: {reason}")


class ServerShutdownError(ServerError):
    """서버 종료 실패."""

    def __init__(self, reason: str):
        super().__init__(f"서버 종료 실패: {reason}")


# ============================================================
# Execution Errors
# ============================================================


class ExecutionError(TTSEngineError):
    """실행 관련 오류."""

    pass


class ExecutionTimeoutError(ExecutionError):
    """실행 타임아웃."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None):
        details = {"timeout_seconds": timeout_seconds} if timeout_seconds else {}
        super().__init__(message, details)


class ExecutorNotRunningError(ExecutionError):
    """실행기가 실행 중이 아님."""

    def __init__(self, executor_type: str):
        super().__init__(
            f"실행기가 실행 중이 아닙니다: {executor_type}",
            {"executor_type": executor_type},
        )
