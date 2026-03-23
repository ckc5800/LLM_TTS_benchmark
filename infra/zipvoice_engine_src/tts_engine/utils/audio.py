# -*- coding: utf-8 -*-
"""Audio Utilities - 오디오 처리 유틸리티.

오디오 변환, 리샘플링, 포맷 변환, 프롬프트 음성 처리 등을 제공합니다.

Features:
- NumPy/Torch 오디오 변환
- 프롬프트 음성 로드 및 전처리 (리샘플링, 침묵 제거, RMS 정규화)
- 청크 교차페이드 연결
- 포맷 변환 (WAV, PCM, MP3, OGG, FLAC)
"""

import io
import wave
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio

from tts_engine.core.constants import AudioFormat, Defaults
from tts_engine.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# 프롬프트 음성 처리 (Prompt Audio Processing)
# ============================================================


def load_prompt_audio(
    path: Union[str, Path],
    sample_rate: int = Defaults.SAMPLE_RATE,
    trim_silence: bool = True,
    trim_top_db: int = 20,
    normalize: bool = True,
    target_rms: float = 0.1,
) -> Tuple[torch.Tensor, float]:
    """프롬프트 음성을 로드하고 전처리합니다.

    Args:
        path: 음성 파일 경로
        sample_rate: 목표 샘플레이트 (기본: 24000)
        trim_silence: 앞뒤 침묵 제거 여부
        trim_top_db: 침묵 감지 임계값 (dB)
        normalize: RMS 정규화 여부
        target_rms: 목표 RMS 값

    Returns:
        (전처리된 오디오 텐서, 원본 RMS)
        - 오디오: shape (1, T) 또는 (C, T)
        - 원본 RMS: 생성 음성 후처리에 사용

    Example:
        >>> audio, orig_rms = load_prompt_audio("prompt.wav")
        >>> # 합성 후 원본 RMS로 정규화
        >>> output = generated_audio * orig_rms / output_rms
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt audio not found: {path}")

    # 1. 로드
    audio, orig_sr = torchaudio.load(str(path))

    # 2. 리샘플링
    if orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=sample_rate,
        )
        audio = resampler(audio)

    # 3. 스테레오 -> 모노 (필요시)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # 4. 앞뒤 침묵 제거 (프롬프트에만 적용)
    if trim_silence:
        audio = trim_silence_edges_torch(audio, top_db=trim_top_db)

    # 5. RMS 정규화
    original_rms = torch.sqrt(torch.mean(torch.square(audio))).item()

    if normalize and original_rms > 0 and original_rms < target_rms:
        audio = audio * (target_rms / original_rms)

    logger.debug(
        "Prompt audio loaded",
        path=str(path),
        sample_rate=sample_rate,
        duration_s=audio.shape[-1] / sample_rate,
        original_rms=f"{original_rms:.4f}",
    )

    return audio, original_rms


def trim_silence_edges_torch(
    audio: torch.Tensor,
    top_db: int = 20,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """앞뒤 침묵만 제거합니다 (중간 침묵 유지).

    librosa.effects.trim과 유사한 torch 구현.

    Args:
        audio: 오디오 텐서 (C, T)
        top_db: 침묵 감지 임계값 (dB below peak)
        frame_length: 프레임 길이
        hop_length: 홉 길이

    Returns:
        침묵이 제거된 오디오 텐서
    """
    if audio.numel() == 0:
        return audio

    # 모노로 분석 (멀티채널이면 평균)
    if audio.dim() == 2 and audio.shape[0] > 1:
        mono = audio.mean(dim=0)
    elif audio.dim() == 2:
        mono = audio.squeeze(0)
    else:
        mono = audio

    # RMS 에너지 계산
    # 간단한 슬라이딩 윈도우 RMS
    n_frames = (len(mono) - frame_length) // hop_length + 1
    if n_frames <= 0:
        return audio

    rms = torch.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = mono[start:end]
        rms[i] = torch.sqrt(torch.mean(frame ** 2))

    # dB 변환
    rms_db = 20 * torch.log10(rms + 1e-10)
    threshold = rms_db.max() - top_db

    # 임계값 이상인 프레임 찾기
    non_silent = rms_db >= threshold
    if not non_silent.any():
        return audio

    # 시작/끝 인덱스 찾기
    indices = torch.where(non_silent)[0]
    start_frame = indices[0].item()
    end_frame = indices[-1].item()

    # 샘플 인덱스로 변환
    start_sample = max(0, start_frame * hop_length)
    end_sample = min(len(mono), (end_frame + 1) * hop_length + frame_length)

    # 슬라이싱
    if audio.dim() == 2:
        return audio[:, start_sample:end_sample]
    return audio[start_sample:end_sample]


def rms_normalize_torch(
    audio: torch.Tensor,
    target_rms: float = 0.1,
) -> Tuple[torch.Tensor, float]:
    """RMS 기반 음량 정규화 (Torch 버전).

    Args:
        audio: 오디오 텐서
        target_rms: 목표 RMS 값

    Returns:
        (정규화된 오디오, 원본 RMS)
    """
    original_rms = torch.sqrt(torch.mean(torch.square(audio))).item()

    if original_rms > 0 and original_rms < target_rms:
        audio = audio * (target_rms / original_rms)

    return audio, original_rms


# ============================================================
# 청크 연결 (Chunk Concatenation)
# ============================================================


def cross_fade_concat_torch(
    chunks: List[torch.Tensor],
    fade_duration: float = 0.1,
    sample_rate: int = Defaults.SAMPLE_RATE,
) -> torch.Tensor:
    """여러 오디오 청크를 교차페이드로 연결합니다.

    Args:
        chunks: 오디오 텐서 리스트, 각각 shape (C, T)
        fade_duration: 교차페이드 길이 (초)
        sample_rate: 샘플레이트

    Returns:
        연결된 오디오 텐서

    Example:
        >>> chunks = [chunk1, chunk2, chunk3]  # 각각 (1, T)
        >>> output = cross_fade_concat_torch(chunks, fade_duration=0.1)
    """
    if not chunks:
        return torch.tensor([])

    if len(chunks) == 1:
        return chunks[0]

    fade_samples = int(fade_duration * sample_rate)

    if fade_samples <= 0:
        return torch.cat(chunks, dim=-1)

    result = chunks[0]

    for next_chunk in chunks[1:]:
        # 안전한 페이드 길이 계산
        k = min(fade_samples, result.shape[-1], next_chunk.shape[-1])

        if k <= 0:
            result = torch.cat([result, next_chunk], dim=-1)
            continue

        # 페이드 커브: 1 -> 0
        fade = torch.linspace(1, 0, k, device=result.device)
        # 브로드캐스팅을 위해 차원 추가
        if result.dim() == 2:
            fade = fade.unsqueeze(0)

        # 3부분 연결:
        # 1. 이전 오디오의 비중첩 부분
        # 2. 교차페이드 영역
        # 3. 다음 오디오의 비중첩 부분
        crossfade_region = result[..., -k:] * fade + next_chunk[..., :k] * (1 - fade)

        result = torch.cat([
            result[..., :-k],
            crossfade_region,
            next_chunk[..., k:],
        ], dim=-1)

    return result


# ============================================================
# Torch <-> NumPy 변환
# ============================================================


def torch_to_numpy(audio: torch.Tensor) -> np.ndarray:
    """Torch 텐서를 NumPy 배열로 변환합니다.

    Args:
        audio: 오디오 텐서 (C, T) 또는 (T,)

    Returns:
        NumPy 배열 (T,) 또는 (C, T)
    """
    arr = audio.cpu().numpy()
    # (1, T) -> (T,)로 압축
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    return arr


def numpy_to_torch(
    audio: np.ndarray,
    device: Optional[str] = None,
) -> torch.Tensor:
    """NumPy 배열을 Torch 텐서로 변환합니다.

    Args:
        audio: NumPy 배열 (T,) 또는 (C, T)
        device: 디바이스 (None이면 CPU)

    Returns:
        오디오 텐서 (C, T)
    """
    tensor = torch.from_numpy(audio)
    # (T,) -> (1, T)로 확장
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if device:
        tensor = tensor.to(device)
    return tensor


def torch_to_wav_bytes(
    audio: torch.Tensor,
    sample_rate: int = Defaults.SAMPLE_RATE,
) -> bytes:
    """Torch 텐서를 WAV 바이트로 변환합니다.

    Args:
        audio: 오디오 텐서 (C, T), float32 [-1, 1]
        sample_rate: 샘플레이트

    Returns:
        WAV 포맷 바이트
    """
    arr = torch_to_numpy(audio)
    return convert_to_wav(arr, sample_rate)


# ============================================================
# 기존 NumPy 기반 함수들
# ============================================================


def convert_to_wav(
    audio_data: np.ndarray,
    sample_rate: int,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """NumPy 배열을 WAV 바이트로 변환합니다.

    Args:
        audio_data: 오디오 데이터 (float32, -1.0 ~ 1.0)
        sample_rate: 샘플레이트
        channels: 채널 수
        sample_width: 샘플 폭 (바이트, 2=16bit)

    Returns:
        WAV 포맷 바이트
    """
    # float32 -> int16 변환
    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        audio_int16 = (audio_data * 32767).astype(np.int16)
    else:
        audio_int16 = audio_data.astype(np.int16)

    # WAV 파일 생성
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    return buffer.getvalue()


def convert_to_pcm(
    audio_data: np.ndarray,
    sample_width: int = 2,
) -> bytes:
    """NumPy 배열을 PCM 바이트로 변환합니다.

    Args:
        audio_data: 오디오 데이터 (float32, -1.0 ~ 1.0)
        sample_width: 샘플 폭 (바이트)

    Returns:
        PCM 포맷 바이트
    """
    if sample_width == 2:
        # 16-bit PCM
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)
        return audio_int16.tobytes()
    elif sample_width == 4:
        # 32-bit PCM
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            audio_int32 = (audio_data * 2147483647).astype(np.int32)
        else:
            audio_int32 = audio_data.astype(np.int32)
        return audio_int32.tobytes()
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")


def wav_bytes_to_numpy(
    wav_bytes: bytes,
) -> Tuple[np.ndarray, int]:
    """WAV 바이트를 NumPy 배열과 샘플레이트로 변환합니다.

    Args:
        wav_bytes: WAV 포맷 바이트

    Returns:
        (float32 NumPy 배열 (-1.0 ~ 1.0), sample_rate)
    """
    buffer = io.BytesIO(wav_bytes)
    with wave.open(buffer, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        raw_data = wav_file.readframes(n_frames)

    if sample_width == 2:
        audio = np.frombuffer(raw_data, dtype=np.int16)
        audio = audio.astype(np.float32) / 32767.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_data, dtype=np.int32)
        audio = audio.astype(np.float32) / 2147483647.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    return audio, sample_rate


def pcm_to_numpy(
    pcm_data: bytes,
    sample_width: int = 2,
    channels: int = 1,
) -> np.ndarray:
    """PCM 바이트를 NumPy 배열로 변환합니다.

    Args:
        pcm_data: PCM 바이트
        sample_width: 샘플 폭 (바이트)
        channels: 채널 수

    Returns:
        float32 NumPy 배열 (-1.0 ~ 1.0)
    """
    if sample_width == 2:
        audio = np.frombuffer(pcm_data, dtype=np.int16)
        audio = audio.astype(np.float32) / 32767.0
    elif sample_width == 4:
        audio = np.frombuffer(pcm_data, dtype=np.int32)
        audio = audio.astype(np.float32) / 2147483647.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels)

    return audio


def resample(
    audio_data: np.ndarray,
    orig_rate: int,
    target_rate: int,
) -> np.ndarray:
    """오디오를 리샘플링합니다.

    Args:
        audio_data: 오디오 데이터
        orig_rate: 원본 샘플레이트
        target_rate: 목표 샘플레이트

    Returns:
        리샘플링된 오디오 데이터
    """
    if orig_rate == target_rate:
        return audio_data

    # 간단한 선형 보간 리샘플링
    duration = len(audio_data) / orig_rate
    target_length = int(duration * target_rate)

    indices = np.linspace(0, len(audio_data) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)

    return resampled.astype(audio_data.dtype)


def normalize_audio(
    audio_data: np.ndarray,
    target_rms: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """오디오 볼륨을 정규화합니다.

    Args:
        audio_data: 오디오 데이터
        target_rms: 목표 RMS 값

    Returns:
        (정규화된 오디오, 원본 RMS)
    """
    original_rms = np.sqrt(np.mean(np.square(audio_data)))

    if original_rms < target_rms and original_rms > 0:
        normalized = audio_data * (target_rms / original_rms)
    else:
        normalized = audio_data

    return normalized, original_rms


def apply_volume(
    audio_data: np.ndarray,
    volume: float,
) -> np.ndarray:
    """볼륨을 적용합니다.

    Args:
        audio_data: 오디오 데이터
        volume: 볼륨 (0.0 ~ 1.0)

    Returns:
        볼륨 적용된 오디오
    """
    if volume == 1.0:
        return audio_data

    return (audio_data * volume).clip(-1.0, 1.0)


def cross_fade(
    audio1: np.ndarray,
    audio2: np.ndarray,
    fade_samples: int,
) -> np.ndarray:
    """두 오디오를 크로스페이드로 연결합니다.

    Args:
        audio1: 첫 번째 오디오
        audio2: 두 번째 오디오
        fade_samples: 페이드 샘플 수

    Returns:
        연결된 오디오
    """
    if fade_samples <= 0 or len(audio1) < fade_samples or len(audio2) < fade_samples:
        return np.concatenate([audio1, audio2])

    # 페이드 커브 생성
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)

    # 오버랩 영역 크로스페이드
    overlap = audio1[-fade_samples:] * fade_out + audio2[:fade_samples] * fade_in

    # 결과 조합
    result = np.concatenate([
        audio1[:-fade_samples],
        overlap,
        audio2[fade_samples:],
    ])

    return result


def calculate_duration(
    audio_data: np.ndarray,
    sample_rate: int,
) -> float:
    """오디오 길이를 계산합니다.

    Args:
        audio_data: 오디오 데이터
        sample_rate: 샘플레이트

    Returns:
        길이 (초)
    """
    return len(audio_data) / sample_rate


def calculate_duration_from_bytes(
    audio_bytes: bytes,
    sample_rate: int,
    sample_width: int = 2,
    channels: int = 1,
) -> float:
    """바이트 데이터에서 오디오 길이를 계산합니다.

    Args:
        audio_bytes: 오디오 바이트
        sample_rate: 샘플레이트
        sample_width: 샘플 폭 (바이트)
        channels: 채널 수

    Returns:
        길이 (초)
    """
    num_samples = len(audio_bytes) // (sample_width * channels)
    return num_samples / sample_rate


def convert_format(
    audio_data: np.ndarray,
    sample_rate: int,
    target_format: AudioFormat,
    channels: int = 1,
) -> bytes:
    """오디오를 지정된 포맷으로 변환합니다.

    Args:
        audio_data: 오디오 데이터 (float32, -1.0 ~ 1.0)
        sample_rate: 샘플레이트
        target_format: 목표 포맷
        channels: 채널 수

    Returns:
        변환된 오디오 바이트

    Supported formats:
        - WAV: 16-bit PCM WAV
        - PCM: Raw 16-bit PCM
        - MP3: MP3 (torchaudio backend 필요)
        - OGG: OGG Vorbis (torchaudio backend 필요)
        - FLAC: FLAC (torchaudio backend 필요)
    """
    if target_format == AudioFormat.WAV:
        return convert_to_wav(audio_data, sample_rate, channels)
    elif target_format == AudioFormat.PCM:
        return convert_to_pcm(audio_data)
    elif target_format in (AudioFormat.MP3, AudioFormat.OGG, AudioFormat.FLAC):
        return _convert_with_torchaudio(audio_data, sample_rate, target_format)
    else:
        raise ValueError(f"Unsupported audio format: {target_format}")


def _convert_with_torchaudio(
    audio_data: np.ndarray,
    sample_rate: int,
    target_format: AudioFormat,
) -> bytes:
    """torchaudio를 사용하여 오디오 포맷을 변환합니다.

    Args:
        audio_data: 오디오 데이터 (float32, -1.0 ~ 1.0)
        sample_rate: 샘플레이트
        target_format: 목표 포맷 (MP3, OGG, FLAC)

    Returns:
        변환된 오디오 바이트
    """
    import tempfile
    import os

    # numpy -> torch
    tensor = torch.from_numpy(audio_data).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # (T,) -> (1, T)

    # 포맷별 설정 (확장자 포함)
    format_config = {
        AudioFormat.MP3: {"format": "mp3", "ext": ".mp3", "compression": 128.0},
        AudioFormat.OGG: {"format": "ogg", "ext": ".ogg", "compression": 5.0},
        AudioFormat.FLAC: {"format": "flac", "ext": ".flac", "compression": 5},
    }

    config = format_config.get(target_format)
    if not config:
        logger.warning(f"Unknown format {target_format}, falling back to WAV")
        return convert_to_wav(audio_data, sample_rate)

    # 임시 파일 사용 (torchaudio가 확장자로 포맷 인식)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=config["ext"], delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name

        # 임시 파일에 저장
        torchaudio.save(
            tmp_path,
            tensor,
            sample_rate,
            format=config["format"],
        )

        # 파일 읽기
        with open(tmp_path, "rb") as f:
            return f.read()

    except Exception as e:
        logger.warning(
            f"Failed to convert to {target_format.value}: {e}, falling back to WAV"
        )
        return convert_to_wav(audio_data, sample_rate)

    finally:
        # 임시 파일 정리
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def split_into_chunks(
    audio_data: np.ndarray,
    chunk_size: int,
) -> list:
    """오디오를 청크로 분할합니다.

    Args:
        audio_data: 오디오 데이터
        chunk_size: 청크 크기 (샘플)

    Returns:
        청크 리스트
    """
    chunks = []
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        chunks.append(chunk)
    return chunks
