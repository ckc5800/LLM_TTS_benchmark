"""Logging configuration using loguru with colorama.

Features:
- 일자별 로테이션 (자정 기준)
- 보관 주기 설정
- 가독성 높은 구조화된 출력
- 컬러 콘솔 출력 (colorama)
- JSON 파일 출력 (분석용)
- 컨텍스트 바인딩 지원
"""

import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import colorama
from loguru import logger

# colorama 초기화 (Windows 호환)
colorama.init(autoreset=True)

# 컨텍스트 변수 저장소
_context_vars: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})


# ============================================================
# 커스텀 로테이션 압축 (파일명 간결화)
# ============================================================


def _make_compression_func(base_path: str):
    """로테이션된 파일을 간결한 이름으로 압축하는 함수를 생성합니다.

    loguru 기본: tts_engine.2025-12-03_00-00-00_123456.log.gz
    변경 후: tts_engine.2025-12-03.log.gz

    Args:
        base_path: 기본 로그 파일 경로 (예: ./logs/tts_engine.log)

    Returns:
        compression 함수 (loguru compression 파라미터에 전달)
    """
    import gzip
    import os
    import re
    import shutil

    base = Path(base_path)
    log_dir = base.parent
    base_stem = base.stem      # tts_engine
    base_suffix = base.suffix  # .log

    def compress(filepath: str) -> None:
        """로테이션된 파일을 압축하고 간결한 이름으로 변경합니다."""
        # filepath 예: ./logs/tts_engine.2025-12-03_00-00-00_123456.log
        # 날짜 추출 (YYYY-MM-DD 형식)
        match = re.search(r"(\d{4}-\d{2}-\d{2})", filepath)
        if match:
            date_str = match.group(1)
        else:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # 간결한 파일명 생성: tts_engine.2025-12-03.log.gz
        new_name = f"{base_stem}.{date_str}{base_suffix}.gz"
        new_path = log_dir / new_name

        # 이미 같은 날짜 파일이 있으면 번호 추가
        if new_path.exists():
            counter = 1
            while True:
                new_name = f"{base_stem}.{date_str}.{counter}{base_suffix}.gz"
                new_path = log_dir / new_name
                if not new_path.exists():
                    break
                counter += 1

        # gzip 압축
        with open(filepath, "rb") as f_in:
            with gzip.open(str(new_path), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # 원본 파일 삭제
        os.remove(filepath)

    return compress


# ============================================================
# 커스텀 포맷터
# ============================================================


def _format_context(record: dict) -> str:
    """컨텍스트 변수를 포맷팅합니다."""
    ctx = _context_vars.get()
    extra = record.get("extra", {})

    # 컨텍스트와 extra 병합
    merged = {**ctx, **extra}

    # 내부 키 제외
    exclude_keys = {"_name", "_module", "_function", "_line"}
    filtered = {k: v for k, v in merged.items() if k not in exclude_keys}

    if not filtered:
        return ""

    # key=value 형식으로 포맷팅
    parts = [f"{k}={v}" for k, v in filtered.items()]
    return " | " + " ".join(parts)


def _console_format(record: dict) -> str:
    """콘솔 출력용 컬러 포맷.

    Format (show_source_location=False):
    2025-11-27 14:30:45.123 | INFO     | Message | key=value

    Format (show_source_location=True):
    2025-11-27 14:30:45.123 | INFO     | module:function:123 | Message | key=value
    """
    # 레벨별 색상 정의
    level_colors = {
        "TRACE": colorama.Fore.CYAN,
        "DEBUG": colorama.Fore.BLUE,
        "INFO": colorama.Fore.GREEN,
        "SUCCESS": colorama.Fore.GREEN + colorama.Style.BRIGHT,
        "WARNING": colorama.Fore.YELLOW,
        "ERROR": colorama.Fore.RED,
        "CRITICAL": colorama.Fore.RED + colorama.Style.BRIGHT,
    }

    level = record["level"].name
    level_color = level_colors.get(level, "")
    reset = colorama.Style.RESET_ALL

    # 타임스탬프 (밀리초 포함)
    time_str = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # 컨텍스트
    context_str = _format_context(record)

    # 메시지와 컨텍스트에서 중괄호 이스케이프 (loguru 포맷 해석 방지)
    message = record["message"].replace("{", "{{").replace("}", "}}")
    context_str = context_str.replace("{", "{{").replace("}", "}}")

    # 최종 포맷 (소스 경로 옵션에 따라)
    if _show_source_location:
        location = f"{record['name']}:{record['function']}:{record['line']}"
        format_str = (
            f"{colorama.Fore.WHITE}{time_str}{reset} | "
            f"{level_color}{level:<8}{reset} | "
            f"{colorama.Fore.CYAN}{location}{reset} | "
            f"{message}"
            f"{colorama.Fore.MAGENTA}{context_str}{reset}"
        )
    else:
        format_str = (
            f"{colorama.Fore.WHITE}{time_str}{reset} | "
            f"{level_color}{level:<8}{reset} | "
            f"{message}"
            f"{colorama.Fore.MAGENTA}{context_str}{reset}"
        )

    # 예외 정보가 있으면 추가
    if record["exception"]:
        format_str += "\n{exception}"

    return format_str + "\n"


# ============================================================
# 로거 설정
# ============================================================


class LogConfig:
    """로깅 설정 클래스."""

    def __init__(
        self,
        level: str = "INFO",
        log_dir: str = "./logs",
        app_name: str = "tts_engine",
        retention_days: int = 30,
        rotation_time: str = "00:00",  # 자정
        console_enabled: bool = True,
        file_enabled: bool = True,
        json_enabled: bool = True,
        show_source_location: bool = False,  # 소스 경로 표시 (모듈:함수:라인)
    ):
        self.level = level.upper()
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.retention_days = retention_days
        self.rotation_time = rotation_time
        self.console_enabled = console_enabled
        self.file_enabled = file_enabled
        self.json_enabled = json_enabled
        self.show_source_location = show_source_location


# 전역 설정 (콘솔 포맷에서 사용)
_show_source_location: bool = False


def setup_logging(
    level: str = "INFO",
    log_dir: str = "./logs",
    app_name: str = "tts_engine",
    retention_days: int = 30,
    rotation_time: str = "00:00",
    console_enabled: bool = True,
    file_enabled: bool = True,
    json_enabled: bool = True,
    show_source_location: bool = False,
) -> None:
    """로깅 시스템을 초기화합니다.

    Args:
        level: 로그 레벨 (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
        log_dir: 로그 파일 저장 디렉토리
        app_name: 애플리케이션 이름 (로그 파일명에 사용)
        retention_days: 로그 파일 보관 일수 (기본: 30일)
        rotation_time: 로테이션 시간 (기본: "00:00" 자정)
        console_enabled: 콘솔 출력 활성화
        file_enabled: 파일 출력 활성화 (가독성 높은 텍스트)
        json_enabled: JSON 파일 출력 활성화 (분석용)
        show_source_location: 콘솔에 소스 경로 표시 (모듈:함수:라인)

    Example:
        >>> setup_logging(
        ...     level="DEBUG",
        ...     log_dir="./logs",
        ...     app_name="tts_engine",
        ...     retention_days=30,
        ... )
    """
    global _show_source_location
    _show_source_location = show_source_location

    config = LogConfig(
        level=level,
        log_dir=log_dir,
        app_name=app_name,
        retention_days=retention_days,
        rotation_time=rotation_time,
        console_enabled=console_enabled,
        file_enabled=file_enabled,
        json_enabled=json_enabled,
        show_source_location=show_source_location,
    )

    # 기존 핸들러 제거
    logger.remove()

    # 로그 디렉토리 생성
    config.log_dir.mkdir(parents=True, exist_ok=True)

    # 1. 콘솔 핸들러 (컬러 출력)
    if config.console_enabled:
        logger.add(
            sys.stderr,
            format=_console_format,
            level=config.level,
            colorize=False,  # 커스텀 컬러 사용
            backtrace=True,
            diagnose=True,
        )

    # 2. 텍스트 파일 핸들러
    # 현재: tts_engine.log
    # 로테이션 후: tts_engine.2025-12-03.log.gz
    if config.file_enabled:
        text_log_path = str(config.log_dir / f"{config.app_name}.log")
        logger.add(
            text_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
            level=config.level,
            rotation=config.rotation_time,  # 자정에 로테이션
            retention=f"{config.retention_days} days",
            compression=_make_compression_func(text_log_path),
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
        )

    # 3. JSON 파일 핸들러 (loguru serialize 사용)
    # 현재: tts_engine.json
    # 로테이션 후: tts_engine.2025-12-03.json.gz
    if config.json_enabled:
        json_log_path = str(config.log_dir / f"{config.app_name}.json")
        logger.add(
            json_log_path,
            level=config.level,
            rotation=config.rotation_time,
            retention=f"{config.retention_days} days",
            compression=_make_compression_func(json_log_path),
            encoding="utf-8",
            serialize=True,  # loguru 기본 JSON 직렬화 사용
        )

    # 4. 에러 전용 파일
    # 현재: tts_engine_error.log
    # 로테이션 후: tts_engine_error.2025-12-03.log.gz
    if config.file_enabled:
        error_log_path = str(config.log_dir / f"{config.app_name}_error.log")
        logger.add(
            error_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}\n{exception}",
            level="ERROR",
            rotation=config.rotation_time,
            retention=f"{config.retention_days * 2} days",
            compression=_make_compression_func(error_log_path),
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
        )

    logger.info(
        f"Logging initialized: level={config.level}, "
        f"log_dir={config.log_dir}, retention={config.retention_days}days"
    )


# ============================================================
# 로거 인터페이스
# ============================================================


def get_logger(name: Optional[str] = None) -> "logger.__class__":
    """모듈별 로거를 반환합니다.

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        loguru logger 인스턴스

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("Server started", port=8220)
        >>> log.error("Connection failed", error=str(e), retry_count=3)
    """
    if name:
        return logger.bind(_name=name)
    return logger


def bind_context(**kwargs: Any) -> None:
    """현재 컨텍스트에 로깅 변수를 바인딩합니다.

    바인딩된 변수는 해당 스레드/코루틴의 모든 로그에 자동 포함됩니다.

    Args:
        **kwargs: 바인딩할 키-값 쌍

    Example:
        >>> bind_context(request_id="req-123", user_id="user-456")
        >>> log.info("Processing request")  # request_id, user_id 자동 포함
    """
    current = _context_vars.get().copy()
    current.update(kwargs)
    _context_vars.set(current)


def unbind_context(*keys: str) -> None:
    """현재 컨텍스트에서 로깅 변수를 제거합니다.

    Args:
        *keys: 제거할 키들

    Example:
        >>> unbind_context("request_id", "user_id")
    """
    current = _context_vars.get().copy()
    for key in keys:
        current.pop(key, None)
    _context_vars.set(current)


def clear_context() -> None:
    """현재 컨텍스트의 모든 로깅 변수를 제거합니다."""
    _context_vars.set({})


# ============================================================
# 편의 함수들
# ============================================================


def log_request(
    request_id: str,
    method: str,
    voice_id: Optional[str] = None,
    text_length: Optional[int] = None,
) -> None:
    """요청 시작 로그를 기록합니다."""
    bind_context(request_id=request_id)
    logger.info(
        f"Request started: {method}",
        voice_id=voice_id,
        text_length=text_length,
    )


def log_response(
    request_id: str,
    status: str,
    duration_ms: float,
    audio_duration_s: Optional[float] = None,
) -> None:
    """요청 완료 로그를 기록합니다."""
    extra = {"status": status, "duration_ms": f"{duration_ms:.2f}"}
    if audio_duration_s:
        extra["audio_duration_s"] = f"{audio_duration_s:.2f}"
        extra["rtf"] = f"{duration_ms / 1000 / audio_duration_s:.4f}"

    logger.info(f"Request completed: {status}", **extra)
    unbind_context("request_id")


def log_synthesis(
    voice_id: str,
    text_length: int,
    duration_ms: float,
    audio_duration_s: Optional[float] = None,
    runtime: Optional[str] = None,
    request_id: Optional[str] = None,
    success: bool = True,
    error: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """합성 완료 로그를 기록합니다."""
    extra: dict[str, Any] = {
        "voice_id": voice_id,
        "text_length": text_length,
        "duration_ms": f"{duration_ms:.2f}",
        "success": success,
    }

    if request_id:
        extra["request_id"] = request_id

    if audio_duration_s and audio_duration_s > 0:
        extra["audio_duration_s"] = f"{audio_duration_s:.2f}"
        rtf = duration_ms / 1000 / audio_duration_s
        extra["rtf"] = f"{rtf:.4f}"

    if runtime:
        extra["runtime"] = runtime

    if error:
        extra["error"] = error

    extra.update(kwargs)

    # 요청ID가 있으면 메시지 앞에 포함
    prefix = f"[{request_id}] " if request_id else ""

    if success:
        logger.info(f"{prefix}Synthesis completed", **extra)
    else:
        logger.error(f"{prefix}Synthesis failed", **extra)


def log_model_event(
    event: str,
    model_name: str,
    runtime: Optional[str] = None,
    duration_ms: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """모델 관련 이벤트 로그를 기록합니다."""
    extra = {"model_name": model_name}
    if runtime:
        extra["runtime"] = runtime
    if duration_ms:
        extra["duration_ms"] = f"{duration_ms:.2f}"
    extra.update(kwargs)

    logger.info(f"Model {event}", **extra)


def log_error(
    error: Exception,
    context: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """에러 로그를 기록합니다."""
    error_type = type(error).__name__
    error_msg = str(error)

    extra = {"error_type": error_type}
    if context:
        extra["context"] = context
    extra.update(kwargs)

    logger.error(f"{error_type}: {error_msg}", **extra)


# 모듈 레벨 로거 (간편 사용)
log = get_logger("tts_engine")
