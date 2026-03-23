# -*- coding: utf-8 -*-
"""Proto-Domain 변환 유틸리티."""

from typing import Tuple

from tts_engine.core.constants import AudioFormat as DomainAudioFormat
from tts_engine.core.exceptions import (
    EmptyTextError,
    ModelNotFoundError,
    PoolExhaustedError,
    SynthesisError,
    SynthesisTimeoutError,
    TextTooLongError,
    VoiceNotFoundError,
    VoiceNotReadyError,
)
from tts_engine.domain.synthesis import SynthesisRequest
from tts_engine.domain.voice import VoiceData

from tts_engine.proto import (
    AudioFormat,
    ErrorCode,
    VoiceInfo,
    SynthesizeRequest as ProtoSynthesizeRequest,
)


def convert_to_domain_request(
    proto_request: ProtoSynthesizeRequest,
    request_id: str,
) -> SynthesisRequest:
    """Proto 요청을 도메인 객체로 변환."""
    format_map = {
        AudioFormat.AUDIO_FORMAT_WAV: DomainAudioFormat.WAV,
        AudioFormat.AUDIO_FORMAT_MP3: DomainAudioFormat.MP3,
        AudioFormat.AUDIO_FORMAT_OGG: DomainAudioFormat.OGG,
        AudioFormat.AUDIO_FORMAT_PCM: DomainAudioFormat.PCM,
        AudioFormat.AUDIO_FORMAT_FLAC: DomainAudioFormat.FLAC,
    }

    audio_format = format_map.get(proto_request.format, DomainAudioFormat.WAV)

    return SynthesisRequest.create(
        text=proto_request.text,
        voice_id=proto_request.voice_id,
        request_id=request_id,
        speed=proto_request.speed if proto_request.speed > 0 else 1.0,
        pitch=proto_request.pitch if proto_request.pitch > 0 else 1.0,
        volume=proto_request.volume if proto_request.volume > 0 else 1.0,
        format=audio_format,
        sample_rate=(
            proto_request.sample_rate
            if proto_request.HasField("sample_rate")
            else None
        ),
    )


def convert_to_proto_format(domain_format: DomainAudioFormat) -> AudioFormat:
    """도메인 오디오 포맷을 Proto 포맷으로 변환."""
    format_map = {
        DomainAudioFormat.WAV: AudioFormat.AUDIO_FORMAT_WAV,
        DomainAudioFormat.MP3: AudioFormat.AUDIO_FORMAT_MP3,
        DomainAudioFormat.OGG: AudioFormat.AUDIO_FORMAT_OGG,
        DomainAudioFormat.PCM: AudioFormat.AUDIO_FORMAT_PCM,
        DomainAudioFormat.FLAC: AudioFormat.AUDIO_FORMAT_FLAC,
    }
    return format_map.get(domain_format, AudioFormat.AUDIO_FORMAT_WAV)


def convert_to_proto_voice_info(voice: VoiceData) -> VoiceInfo:
    """VoiceData를 Proto VoiceInfo로 변환.

    Args:
        voice: 도메인 VoiceData 객체

    Returns:
        VoiceInfo protobuf 객체 (ready, supported_formats 포함)
    """
    metadata = {}
    if voice.options:
        for k, v in voice.options.items():
            metadata[str(k)] = str(v) if v is not None else ""

    # ready 상태: 프롬프트가 로드되어야 합성 가능
    is_ready = voice.is_prompt_loaded

    return VoiceInfo(
        voice_id=voice.voice_id,
        name=voice.name,
        language=voice.language,
        gender=voice.gender,
        sample_rate=voice.sample_rate,
        description=voice.description,
        enabled=voice.enabled,
        ready=is_ready,
        model_instance=voice.model_instance,
        supported_formats=[
            AudioFormat.AUDIO_FORMAT_WAV,
            AudioFormat.AUDIO_FORMAT_MP3,
            AudioFormat.AUDIO_FORMAT_OGG,
        ],
        metadata=metadata,
    )


def map_exception_to_error(exception: Exception) -> Tuple[int, str]:
    """예외를 ErrorCode와 메시지로 매핑.

    Returns:
        Tuple[int, str]: (ErrorCode enum value, error message)
    """
    error_mapping = {
        EmptyTextError: ErrorCode.ERROR_EMPTY_TEXT,
        TextTooLongError: ErrorCode.ERROR_TEXT_TOO_LONG,
        VoiceNotFoundError: ErrorCode.ERROR_VOICE_NOT_FOUND,
        VoiceNotReadyError: ErrorCode.ERROR_VOICE_NOT_READY,
        ModelNotFoundError: ErrorCode.ERROR_MODEL_NOT_FOUND,
        PoolExhaustedError: ErrorCode.ERROR_POOL_EXHAUSTED,
        SynthesisTimeoutError: ErrorCode.ERROR_TIMEOUT,
        SynthesisError: ErrorCode.ERROR_SYNTHESIS_FAILED,
    }

    for exc_type, code in error_mapping.items():
        if isinstance(exception, exc_type):
            return code, str(exception)

    return ErrorCode.ERROR_INTERNAL, f"Internal error: {exception}"
