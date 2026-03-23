# -*- coding: utf-8 -*-
"""TTS Engine Main - 애플리케이션 엔트리포인트.

gRPC TTS 서버를 시작하고 관리합니다.

Usage:
    python -m tts_engine.main [--config CONFIG_PATH]
    python -m tts_engine.main --help
"""

import argparse
import asyncio
import os
import platform
import sys
from pathlib import Path
from typing import Optional

# Windows에서 PosixPath 에러 방지
if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

# ONNX Runtime 로그 레벨 설정 (다른 import 전에 설정)
os.environ.setdefault('ORT_LOG_LEVEL', '3')
try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)  # ERROR 이상만 표시
except ImportError:
    pass

# TensorRT 로그 레벨 설정 (ERROR만 표시)
os.environ.setdefault('TRT_LOGGER_LEVEL', '3')
try:
    import tensorrt as trt
    _trt_logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(_trt_logger, "")
except ImportError:
    pass
except Exception:
    pass  # TensorRT 초기화 실패해도 계속 진행

from tts_engine.core.config import Settings, VoicesConfig, load_settings
from tts_engine.core.logging import get_logger, setup_logging
from tts_engine.server.grpc_server import run_server

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱합니다.

    Returns:
        파싱된 인자
    """
    parser = argparse.ArgumentParser(
        description="AICESS TTS Engine - gRPC TTS Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 기본 설정으로 시작
    python -m tts_engine.main

    # 커스텀 설정 파일 사용
    python -m tts_engine.main --config /path/to/settings.yaml

    # 개발 모드 (상세 로깅)
    python -m tts_engine.main --debug

Environment Variables:
    TTS_ENGINE_CONFIG_PATH   설정 파일 경로
    TTS_ENGINE_LOG_LEVEL     로그 레벨 (DEBUG, INFO, WARNING, ERROR)
    TTS_ENGINE_GRPC_PORT     gRPC 서버 포트
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="설정 파일 경로 (기본: config/settings.yaml)",
    )

    parser.add_argument(
        "--voices",
        "-v",
        type=str,
        default=None,
        help="음성 설정 파일 경로 (기본: config/voices.yaml)",
    )

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="디버그 모드 활성화 (상세 로깅)",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        help="gRPC 서버 포트 (설정 파일 오버라이드)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="gRPC 서버 호스트 (설정 파일 오버라이드)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="AICESS TTS Engine 1.0.0",
    )

    return parser.parse_args()


def load_configuration(args: argparse.Namespace) -> tuple[Settings, VoicesConfig]:
    """설정을 로드합니다.

    Args:
        args: 명령행 인자

    Returns:
        (Settings, VoicesConfig) 튜플

    Raises:
        SystemExit: 설정 로드 실패 시
    """
    try:
        # 설정 파일 경로 결정
        config_dir = args.config
        voices_path = args.voices

        # 설정 로드
        settings, voices = load_settings(
            config_dir=config_dir,
            voices_path=voices_path,
        )

        # 명령행 인자로 오버라이드
        if args.port:
            settings.server.grpc.port = args.port

        if args.host:
            settings.server.grpc.host = args.host

        if args.debug:
            settings.logging.level = "DEBUG"

        return settings, voices

    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}", file=sys.stderr)
        sys.exit(1)

    except ValueError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"Error: Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)


async def main_async(settings: Settings, voices: VoicesConfig) -> int:
    """비동기 메인 함수.

    Args:
        settings: 애플리케이션 설정
        voices: 음성 설정

    Returns:
        종료 코드 (0: 성공, 1: 실패)
    """
    try:
        logger.info(
            "Starting AICESS TTS Engine",
            host=settings.server.grpc.host,
            port=settings.server.grpc.port,
            log_level=settings.logging.level,
        )

        # 서버 실행
        await run_server(settings, voices)

        return 0

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        return 0

    except Exception as e:
        logger.error("Application error", error=str(e), exc_info=True)
        return 1


def main() -> int:
    """메인 엔트리포인트.

    Returns:
        종료 코드
    """
    # 명령행 인자 파싱
    args = parse_args()

    # 설정 로드
    settings, voices = load_configuration(args)

    # 로깅 설정
    setup_logging(
        level=settings.logging.level,
        log_dir=settings.logging.log_dir,
        console_enabled=settings.logging.console_enabled,
        file_enabled=settings.logging.file_enabled,
        json_enabled=settings.logging.json_enabled,
        show_source_location=settings.logging.show_source_location,
    )

    # 시작 배너
    print_banner(settings)

    # 비동기 메인 실행
    try:
        return asyncio.run(main_async(settings, voices))
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        return 0


def print_banner(settings: Settings) -> None:
    """시작 배너를 출력합니다.

    Args:
        settings: 애플리케이션 설정
    """
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    AICESS TTS Engine                         ║
║                      Version 1.0.0                           ║
╠══════════════════════════════════════════════════════════════╣
║  gRPC Server: {host}:{port:<36}║
║  Log Level:   {log_level:<42}║
║  Execution:   {exec_mode:<42}║
╚══════════════════════════════════════════════════════════════╝
    """.format(
        host=settings.server.grpc.host,
        port=settings.server.grpc.port,
        log_level=settings.logging.level,
        exec_mode=settings.performance.execution.mode,
    )
    print(banner)


if __name__ == "__main__":
    sys.exit(main())
