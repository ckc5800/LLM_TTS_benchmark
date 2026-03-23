# -*- coding: utf-8 -*-
"""오디오 파일 저장 서비스.

합성된 오디오를 파일 시스템에 저장하고 관리합니다.

디렉토리 구조:
    audio_output/
    └── 20251203/           # 날짜별 디렉토리 (YYYYMMDD)
        ├── req_abc123.wav  # 요청 ID 기반 파일명
        ├── req_def456.mp3
        └── ...

지원 포맷:
    - WAV: 기본 포맷 (16-bit PCM)
    - PCM: Raw PCM 데이터
    - MP3: 압축 포맷 (128kbps)
    - OGG: OGG Vorbis
    - FLAC: 무손실 압축
"""

import asyncio
import io
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np

from tts_engine.core.config import FileStorageConfig
from tts_engine.core.constants import AudioFormat
from tts_engine.core.logging import get_logger

logger = get_logger(__name__)


class FileStorageService:
    """오디오 파일 저장 서비스."""

    # 오디오 포맷별 파일 확장자 매핑
    FORMAT_EXTENSIONS: dict[AudioFormat, str] = {
        AudioFormat.WAV: ".wav",
        AudioFormat.MP3: ".mp3",
        AudioFormat.OGG: ".ogg",
        AudioFormat.PCM: ".pcm",
        AudioFormat.FLAC: ".flac",
    }

    def __init__(self, config: FileStorageConfig):
        """초기화.

        Args:
            config: 파일 저장소 설정
        """
        self._config = config
        self._base_path = Path(config.path)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def enabled(self) -> bool:
        """파일 저장 활성화 여부."""
        return self._config.enabled

    async def start(self) -> None:
        """서비스 시작."""
        if not self._config.enabled:
            logger.info("FileStorageService disabled")
            return

        # 기본 디렉토리 생성
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._running = True

        # 자동 정리 태스크 시작
        if self._config.cleanup_enabled:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info(
            "FileStorageService started",
            path=str(self._base_path),
            cleanup_enabled=self._config.cleanup_enabled,
            retention_hours=self._config.retention_hours,
        )

    async def stop(self) -> None:
        """서비스 중지."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        logger.info("FileStorageService stopped")

    def _get_extension(self, audio_format: Union[AudioFormat, str]) -> str:
        """오디오 포맷에 맞는 파일 확장자를 반환합니다.

        Args:
            audio_format: 오디오 포맷

        Returns:
            파일 확장자 (예: ".wav", ".mp3")
        """
        if isinstance(audio_format, str):
            try:
                audio_format = AudioFormat(audio_format.lower())
            except ValueError:
                return ".wav"  # 기본값

        return self.FORMAT_EXTENSIONS.get(audio_format, ".wav")

    def _get_storage_path(
        self,
        request_id: str,
        audio_format: Union[AudioFormat, str] = AudioFormat.WAV,
    ) -> Path:
        """저장 경로를 생성합니다.

        Args:
            request_id: 요청 ID
            audio_format: 오디오 포맷 (확장자 결정에 사용)

        Returns:
            파일 저장 경로
        """
        if self._config.use_date_subdirs:
            # 날짜별 디렉토리: audio_output/20251203/
            date_dir = datetime.now().strftime("%Y%m%d")
            dir_path = self._base_path / date_dir
        else:
            dir_path = self._base_path

        # 디렉토리 생성
        dir_path.mkdir(parents=True, exist_ok=True)

        # 파일명: request_id.{ext}
        # 특수문자 제거
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in request_id)
        extension = self._get_extension(audio_format)
        return dir_path / f"{safe_id}{extension}"

    async def save(
        self,
        request_id: str,
        audio_data: bytes,
        audio_format: Union[AudioFormat, str] = AudioFormat.WAV,
        sample_rate: int = 24000,
    ) -> Optional[str]:
        """오디오 데이터를 파일로 저장합니다.

        입력 데이터가 WAV 포맷이고 요청 포맷이 다른 경우 변환을 수행합니다.

        Args:
            request_id: 요청 ID
            audio_data: 오디오 데이터 (WAV 포맷)
            audio_format: 저장할 오디오 포맷
            sample_rate: 샘플레이트 (포맷 변환 시 사용)

        Returns:
            저장된 파일 경로 (비활성화 시 None)
        """
        if not self._config.enabled:
            return None

        try:
            # 포맷 정규화
            if isinstance(audio_format, str):
                try:
                    audio_format = AudioFormat(audio_format.lower())
                except ValueError:
                    audio_format = AudioFormat.WAV

            file_path = self._get_storage_path(request_id, audio_format)

            # 이미 변환된 데이터인지 확인 (WAV 헤더로 판별)
            # synthesis_service에서 이미 변환된 경우 그대로 저장
            output_data = audio_data
            is_wav_data = audio_data[:4] == b'RIFF'

            # WAV 데이터인데 다른 포맷을 요청한 경우에만 변환
            if is_wav_data and audio_format != AudioFormat.WAV:
                output_data = await self._convert_audio_format(
                    audio_data, audio_format, sample_rate
                )

            # 비동기 파일 쓰기
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: file_path.write_bytes(output_data),
            )

            logger.debug(
                f"[{request_id}] Audio file saved",
                path=str(file_path),
                format=audio_format.value,
                original_size=len(audio_data),
                saved_size=len(output_data),
            )

            return str(file_path)

        except Exception as e:
            logger.error(
                f"[{request_id}] Failed to save audio file",
                error=str(e),
            )
            return None

    async def _convert_audio_format(
        self,
        wav_data: bytes,
        target_format: AudioFormat,
        sample_rate: int,
    ) -> bytes:
        """WAV 데이터를 지정된 포맷으로 변환합니다.

        Args:
            wav_data: WAV 포맷 오디오 데이터
            target_format: 목표 포맷
            sample_rate: 샘플레이트

        Returns:
            변환된 오디오 바이트
        """
        loop = asyncio.get_event_loop()

        def _convert():
            from tts_engine.utils.audio import convert_format

            # WAV 바이트에서 PCM 데이터 추출
            audio_np = self._wav_bytes_to_numpy(wav_data)

            # 목표 포맷으로 변환
            return convert_format(audio_np, sample_rate, target_format)

        return await loop.run_in_executor(None, _convert)

    def _wav_bytes_to_numpy(self, wav_data: bytes) -> np.ndarray:
        """WAV 바이트를 NumPy 배열로 변환합니다.

        Args:
            wav_data: WAV 포맷 바이트

        Returns:
            float32 NumPy 배열 (-1.0 ~ 1.0)
        """
        buffer = io.BytesIO(wav_data)
        with wave.open(buffer, "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            raw_data = wav_file.readframes(n_frames)

        # PCM to numpy
        if sample_width == 2:
            audio = np.frombuffer(raw_data, dtype=np.int16)
            audio = audio.astype(np.float32) / 32767.0
        elif sample_width == 4:
            audio = np.frombuffer(raw_data, dtype=np.int32)
            audio = audio.astype(np.float32) / 2147483647.0
        else:
            audio = np.frombuffer(raw_data, dtype=np.int16)
            audio = audio.astype(np.float32) / 32767.0

        # 스테레오 -> 모노 (첫 채널만)
        if n_channels > 1:
            audio = audio[::n_channels]

        return audio

    async def _cleanup_loop(self) -> None:
        """오래된 파일을 정기적으로 정리합니다."""
        # 1시간마다 정리 실행
        cleanup_interval = 3600

        while self._running:
            try:
                await asyncio.sleep(cleanup_interval)
                await self._cleanup_old_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))

    async def _cleanup_old_files(self) -> None:
        """보관 기간이 지난 파일을 삭제합니다."""
        if not self._config.cleanup_enabled:
            return

        try:
            cutoff_time = time.time() - (self._config.retention_hours * 3600)
            deleted_count = 0
            deleted_dirs = 0

            loop = asyncio.get_event_loop()

            def _do_cleanup():
                nonlocal deleted_count, deleted_dirs

                for item in self._base_path.iterdir():
                    if item.is_dir():
                        # 날짜 디렉토리 처리
                        dir_empty = True
                        for file in item.iterdir():
                            if file.is_file() and file.stat().st_mtime < cutoff_time:
                                file.unlink()
                                deleted_count += 1
                            else:
                                dir_empty = False

                        # 빈 디렉토리 삭제
                        if dir_empty and not any(item.iterdir()):
                            item.rmdir()
                            deleted_dirs += 1

                    elif item.is_file() and item.stat().st_mtime < cutoff_time:
                        item.unlink()
                        deleted_count += 1

            await loop.run_in_executor(None, _do_cleanup)

            if deleted_count > 0 or deleted_dirs > 0:
                logger.info(
                    "Cleanup completed",
                    deleted_files=deleted_count,
                    deleted_dirs=deleted_dirs,
                    retention_hours=self._config.retention_hours,
                )

        except Exception as e:
            logger.error("Cleanup failed", error=str(e))

    def get_stats(self) -> dict:
        """저장소 통계를 반환합니다."""
        if not self._config.enabled:
            return {"enabled": False}

        try:
            total_files = 0
            total_size = 0
            dir_count = 0

            for item in self._base_path.iterdir():
                if item.is_dir():
                    dir_count += 1
                    for file in item.iterdir():
                        if file.is_file():
                            total_files += 1
                            total_size += file.stat().st_size
                elif item.is_file():
                    total_files += 1
                    total_size += item.stat().st_size

            return {
                "enabled": True,
                "path": str(self._base_path),
                "total_files": total_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "date_dirs": dir_count,
                "cleanup_enabled": self._config.cleanup_enabled,
                "retention_hours": self._config.retention_hours,
            }

        except Exception as e:
            return {
                "enabled": True,
                "path": str(self._base_path),
                "error": str(e),
            }
