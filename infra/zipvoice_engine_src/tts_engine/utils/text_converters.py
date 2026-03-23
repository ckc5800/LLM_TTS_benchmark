# -*- coding: utf-8 -*-
"""단위/전화번호/이메일 변환 유틸리티."""

import re
from typing import Dict

from tts_engine.utils.text_korean import number_to_korean_digits, NUM_KOR, ALPHABET_PRONUNCIATION

# 이메일 정규표현식
EMAIL_RE = re.compile(r"[\w\.\-+]+@[\w\.-]+\.\w+")

# 과학 단위 매핑
SCIENCE_UNIT_MAP: Dict[str, str] = {
    r"°C\b": "도씨",
    r"℃\b": "도씨",
    r"(?<=\d)\s*K\b": " 켈빈",
    r"(?<=\d)\s*GHz\b": " 기가헤르츠",
    r"(?<=\d)\s*MHz\b": " 메가헤르츠",
    r"(?<=\d)\s*kHz\b": " 킬로헤르츠",
    r"(?<=\d)\s*Hz\b": " 헤르츠",
    r"(?<=\d)\s*kW\b": " 킬로와트",
    r"(?<=\d)\s*mW\b": " 밀리와트",
    r"(?<=\d)\s*kV\b": " 킬로볼트",
    r"(?<=\d)\s*mV\b": " 밀리볼트",
    r"(?<=\d)\s*mA\b": " 밀리암페어",
    r"(?<=\d)\s*MΩ\b": " 메가옴",
    r"(?<=\d)\s*kΩ\b": " 킬로옴",
    r"(?<=\d)\s*Ω\b": " 옴",
    r"(?<=\d)\s*GPa\b": " 기가파스칼",
    r"(?<=\d)\s*MPa\b": " 메가파스칼",
    r"(?<=\d)\s*kPa\b": " 킬로파스칼",
    r"(?<=\d)\s*hPa\b": " 헥토파스칼",
    r"(?<=\d)\s*Pa\b": " 파스칼",
    r"(?<=\d)\s*nm\b": " 나노미터",
    r"(?<=\d)\s*μm\b": " 마이크로미터",
    r"(?<=\d)\s*mm\b": " 밀리미터",
    r"(?<=\d)\s*cm\b": " 센티미터",
    r"(?<=\d)\s*%": " 퍼센트",
}

# 일반 단위 매핑
GENERAL_UNIT_MAP: Dict[str, str] = {
    r"(?i)(?<=\d)km\b": "킬로미터",
    r"(?i)(?<=\d)m\b": "미터",
    r"(?i)(?<=\d)kg\b": "킬로그램",
    r"(?i)(?<=\d)g\b": "그램",
    r"(?i)(?<=\d)l\b": "리터",
    r"(?i)(?<=\d)ml\b": "밀리리터",
}


def convert_science_units(text: str) -> str:
    """과학 단위를 한국어로 변환합니다."""
    for pattern, replacement in SCIENCE_UNIT_MAP.items():
        text = re.sub(pattern, replacement, text)
    return text


def convert_general_units(text: str) -> str:
    """일반 단위를 한국어로 변환합니다."""
    for pattern, repl in GENERAL_UNIT_MAP.items():
        text = re.sub(pattern, repl, text)
    return text


def convert_phone_to_korean(text: str) -> str:
    """전화번호를 한국어로 변환합니다."""
    patterns = [
        re.compile(r"(\d{2,3})-(\d{3,4})-(\d{4})"),
        re.compile(r"(\d{3})-(\d{4})-(\d{4})"),
        re.compile(r"(\d{4})-(\d{4})"),
    ]

    def replace_phone(match):
        groups = match.groups()
        parts = [number_to_korean_digits(g) for g in groups if g]
        return " ".join(parts)

    for pattern in patterns:
        text = pattern.sub(replace_phone, text)

    return text


def convert_email_to_korean(email: str) -> str:
    """이메일을 한국어 발음으로 변환합니다."""
    result = []
    for char in email:
        if char.isdigit():
            result.append(NUM_KOR[int(char)])
        elif char in ALPHABET_PRONUNCIATION:
            result.append(ALPHABET_PRONUNCIATION[char])
        elif char == "@":
            result.append("골뱅이")
        elif char == ".":
            result.append("점")
        elif char == "-":
            result.append("하이픈")
        elif char == "_":
            result.append("언더바")
        else:
            result.append(char)
    return " ".join(result)


def convert_emails_in_text(text: str) -> str:
    """텍스트에서 이메일을 한국어로 변환합니다."""
    return EMAIL_RE.sub(lambda m: convert_email_to_korean(m.group()), text)
