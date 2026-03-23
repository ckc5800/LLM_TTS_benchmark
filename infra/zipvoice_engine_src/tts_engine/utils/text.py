# -*- coding: utf-8 -*-
"""Text Utilities - 텍스트 처리 유틸리티.

이 모듈은 하위 모듈들을 통합하여 기존 API와의 호환성을 유지합니다.
"""

import re

# 기본 텍스트 처리
from tts_engine.utils.text_base import (
    PUNCTUATION,
    SENTENCE_END,
    normalize_text,
    clean_text,
    add_punctuation,
    split_sentences,
    chunk_text,
    chunk_by_punctuation,
    estimate_duration,
    contains_korean,
    contains_english,
    detect_language,
    validate_text,
    remove_emojis,
    process_parentheses,
)

# 한국어 숫자/시간 변환
from tts_engine.utils.text_korean import (
    NUM_KOR,
    UNIT_KOR,
    HIGH_UNIT,
    HOUR_KOR,
    ALPHABET_PRONUNCIATION,
    SPECIAL_NUMBERS,
    number_to_korean,
    number_to_korean_digits,
    convert_time_to_korean,
    convert_special_numbers,
    convert_numbers_to_korean,
    convert_comma_numbers,
    convert_mixed_alphabet,
    convert_alphanumeric_mixed,
)

# 단위/전화번호/이메일 변환
from tts_engine.utils.text_converters import (
    EMAIL_RE,
    SCIENCE_UNIT_MAP,
    GENERAL_UNIT_MAP,
    convert_science_units,
    convert_general_units,
    convert_phone_to_korean,
    convert_email_to_korean,
    convert_emails_in_text,
)

from tts_engine.core.logging import get_logger

logger = get_logger(__name__)


def normalize_korean(
    text: str,
    parentheses_enable: bool = True,
    parentheses_max_length: int = 0,
) -> str:
    """텍스트를 한국어 음성 합성에 적합하게 정규화합니다."""
    # 1. 시간 변환
    text = convert_time_to_korean(text)

    # 2. 단위 변환
    text = convert_science_units(text)
    text = convert_general_units(text)

    # 3. 이메일 변환
    text = convert_emails_in_text(text)

    # 4. 이모지 제거
    text = remove_emojis(text)

    # 5. 괄호 처리
    text = process_parentheses(
        text, enable=parentheses_enable, max_length=parentheses_max_length
    )

    # 6. 영문자 혼합 처리
    text = convert_mixed_alphabet(text)
    text = convert_alphanumeric_mixed(text)

    # 7. 콤마 숫자 처리
    text = convert_comma_numbers(text)

    # 8. 전화번호 처리
    text = convert_phone_to_korean(text)

    # 9. 특수 번호 처리
    text = convert_special_numbers(text)

    # 10. 숫자 → 한글 변환
    text = convert_numbers_to_korean(text)

    # 11. 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 하위 호환성을 위해 모든 심볼 노출
__all__ = [
    # 상수
    "PUNCTUATION",
    "SENTENCE_END",
    "NUM_KOR",
    "UNIT_KOR",
    "HIGH_UNIT",
    "HOUR_KOR",
    "ALPHABET_PRONUNCIATION",
    "SPECIAL_NUMBERS",
    "EMAIL_RE",
    "SCIENCE_UNIT_MAP",
    "GENERAL_UNIT_MAP",
    # 기본 텍스트
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
    "remove_emojis",
    "process_parentheses",
    # 한국어 변환
    "number_to_korean",
    "number_to_korean_digits",
    "convert_time_to_korean",
    "convert_special_numbers",
    "convert_numbers_to_korean",
    "convert_comma_numbers",
    "convert_mixed_alphabet",
    "convert_alphanumeric_mixed",
    # 단위/전화번호/이메일
    "convert_science_units",
    "convert_general_units",
    "convert_phone_to_korean",
    "convert_email_to_korean",
    "convert_emails_in_text",
    # 통합 함수
    "normalize_korean",
]
