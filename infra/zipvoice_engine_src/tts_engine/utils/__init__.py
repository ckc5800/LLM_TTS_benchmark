# -*- coding: utf-8 -*-
"""Utilities Package - 유틸리티 패키지.

오디오 처리, 텍스트 처리, 메트릭 수집 등의 유틸리티를 제공합니다.
"""

from tts_engine.utils.audio import (
    apply_volume,
    calculate_duration,
    calculate_duration_from_bytes,
    convert_format,
    convert_to_pcm,
    convert_to_wav,
    cross_fade,
    normalize_audio,
    pcm_to_numpy,
    resample,
    split_into_chunks,
)
from tts_engine.utils.text import (
    add_punctuation,
    chunk_by_punctuation,
    chunk_text,
    clean_text,
    contains_english,
    contains_korean,
    detect_language,
    estimate_duration,
    normalize_text,
    split_sentences,
    validate_text,
    PUNCTUATION,
    SENTENCE_END,
)

__all__ = [
    # Audio utilities
    "convert_to_wav",
    "convert_to_pcm",
    "pcm_to_numpy",
    "resample",
    "normalize_audio",
    "apply_volume",
    "cross_fade",
    "calculate_duration",
    "calculate_duration_from_bytes",
    "convert_format",
    "split_into_chunks",
    # Text utilities
    "normalize_text",
    "clean_text",
    "add_punctuation",
    "split_sentences",
    "chunk_text",
    "chunk_by_punctuation",
    "estimate_duration",
    "contains_korean",
    "contains_english",
    "detect_language",
    "validate_text",
    "PUNCTUATION",
    "SENTENCE_END",
]
