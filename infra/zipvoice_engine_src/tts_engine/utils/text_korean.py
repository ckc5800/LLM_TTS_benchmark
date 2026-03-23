# -*- coding: utf-8 -*-
"""한국어 숫자/시간 변환 유틸리티."""

import re
from typing import Dict

from tts_engine.core.logging import get_logger

logger = get_logger(__name__)

# 한국어 숫자 상수
NUM_KOR = ["영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
UNIT_KOR = ["", "십", "백", "천"]
HIGH_UNIT = ["", "만", "억", "조"]
HOUR_KOR = ["열두", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉", "열", "열한", "열두"]

# 알파벳 발음 매핑
ALPHABET_PRONUNCIATION: Dict[str, str] = {
    "a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이", "f": "에프", "g": "지",
    "h": "에이치", "i": "아이", "j": "제이", "k": "케이", "l": "엘", "m": "엠",
    "n": "엔", "o": "오", "p": "피", "q": "큐", "r": "알", "s": "에스", "t": "티",
    "u": "유", "v": "브이", "w": "더블유", "x": "엑스", "y": "와이", "z": "제트",
}
ALPHABET_PRONUNCIATION.update({k.upper(): v for k, v in ALPHABET_PRONUNCIATION.items()})

# 특수 번호 매핑
SPECIAL_NUMBERS: Dict[str, str] = {
    "119": "일일구",
    "112": "일일이",
    "911": "구일일",
    "114": "일일사",
    "113": "일일삼",
}


def number_to_korean(num: str | int | float) -> str:
    """숫자를 한국어로 변환합니다."""
    try:
        if not num and num != 0:
            return "영"

        num_str = str(num).replace(",", "").strip()
        if not num_str:
            return "영"

        # 소수점 처리
        if "." in num_str:
            parts = num_str.split(".")
            if len(parts) != 2:
                return "영"
            integer, decimal = parts
            if not integer:
                integer = "0"
            if not decimal:
                return number_to_korean(int(integer))
            return (
                number_to_korean(int(integer))
                + "점"
                + "".join(NUM_KOR[int(d)] for d in decimal)
            )

        num_val = int(num_str)
        if num_val == 0:
            return "영"

        num_str = str(num_val)
        num_str = num_str.zfill(((len(num_str) - 1) // 4 + 1) * 4)
        chunks = [num_str[i : i + 4] for i in range(0, len(num_str), 4)]

        result = ""
        for i, chunk in enumerate(chunks):
            chunk_result = ""
            for j, digit in enumerate(chunk):
                n = int(digit)
                if n != 0:
                    if n == 1 and UNIT_KOR[4 - j - 1] in ["십", "백", "천"]:
                        chunk_result += UNIT_KOR[4 - j - 1]
                    else:
                        chunk_result += NUM_KOR[n] + UNIT_KOR[4 - j - 1]
            if chunk_result:
                if chunk_result == "일" and HIGH_UNIT[len(chunks) - i - 1] == "만":
                    result += "만"
                else:
                    result += chunk_result + HIGH_UNIT[len(chunks) - i - 1]

        return result if result else "영"
    except Exception as e:
        logger.debug(f"number_to_korean 변환 오류: {num} - {e}")
        return str(num)


def number_to_korean_digits(num_str: str) -> str:
    """숫자를 한 자리씩 읽습니다."""
    digit_map = {
        "0": "공", "1": "일", "2": "이", "3": "삼", "4": "사",
        "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구",
    }
    return "".join(digit_map.get(d, d) for d in num_str)


def _time_to_kor_hour(n: int) -> str:
    """1~12시 숫자를 한글 발음으로 변환."""
    n = int(n) % 12
    return HOUR_KOR[n]


def _time_to_kor_minute(n: int) -> str:
    """0~59분 숫자를 한글 발음으로 변환."""
    n = int(n)
    if n == 0:
        return "정각"
    tens = n // 10
    ones = n % 10
    result = ""
    if tens == 1:
        result += "십"
    elif tens > 1:
        result += NUM_KOR[tens] + "십"
    if ones > 0:
        result += NUM_KOR[ones]
    return result + "분"


def convert_time_to_korean(text: str) -> str:
    """시간 표현을 한국어로 변환합니다."""
    # 시간 범위 패턴
    range_pattern = re.compile(
        r"(\d{1,2})\s*:\s*(\d{2})\s*(AM|PM|am|pm)?\s*~\s*"
        r"(\d{1,2})\s*:\s*(\d{2})\s*(AM|PM|am|pm)?"
    )

    def repl_range(m):
        h1, m1, ampm1, h2, m2, ampm2 = m.groups()
        prefix1 = "오전 " if ampm1 and ampm1.lower() == "am" else ("오후 " if ampm1 else "")
        prefix2 = "오전 " if ampm2 and ampm2.lower() == "am" else ("오후 " if ampm2 else "")
        t1_h = _time_to_kor_hour(h1) + "시"
        t2_h = _time_to_kor_hour(h2) + "시"
        t1_m = _time_to_kor_minute(m1)
        t2_m = _time_to_kor_minute(m2)
        return f"{prefix1}{t1_h}{t1_m}부터 {prefix2}{t2_h}{t2_m}까지"

    text = range_pattern.sub(repl_range, text)

    # 단일 시간 패턴
    single_pattern = re.compile(r"(\d{1,2})\s*:\s*(\d{2})\s*(AM|PM|am|pm)?")

    def repl_single(m):
        h, minute, ampm = m.groups()
        prefix = "오전 " if ampm and ampm.lower() == "am" else ("오후 " if ampm and ampm.lower() == "pm" else "")
        t_h = _time_to_kor_hour(h) + "시"
        t_m = _time_to_kor_minute(minute)
        return f"{prefix}{t_h}{t_m}"

    return single_pattern.sub(repl_single, text)


def convert_special_numbers(text: str) -> str:
    """특수 번호 (119, 112 등)를 변환합니다."""
    for num, korean in SPECIAL_NUMBERS.items():
        text = re.sub(r"(?<![0-9])" + num + r"(?![0-9])", korean, text)
    return text


def convert_numbers_to_korean(text: str) -> str:
    """텍스트의 모든 숫자를 한국어로 변환합니다."""
    def repl_num(m):
        return number_to_korean(m.group())
    return re.sub(r"\d+(\.\d+)?", repl_num, text)


def convert_comma_numbers(text: str) -> str:
    """콤마가 포함된 숫자를 변환합니다."""
    comma_pattern = re.compile(r"(\d{1,3}(?:,\d{3})+)")

    def replace_comma(match):
        number_str = match.group(1).replace(",", "")
        return number_to_korean(int(number_str))

    return comma_pattern.sub(replace_comma, text)


def convert_mixed_alphabet(text: str) -> str:
    """한글과 섞인 영문자를 발음으로 변환합니다."""
    english_hangul = re.compile(r"([A-Z]{1,10})([가-힣]+)")
    hangul_english = re.compile(r"([가-힣]+)([A-Z]{1,10})")
    standalone = re.compile(r"\b([A-Z]{2,10})\b")

    def replace_eng_kor(match):
        eng = match.group(1)
        kor = match.group(2)
        pron = " ".join(ALPHABET_PRONUNCIATION.get(c, c) for c in eng)
        return pron + kor

    def replace_kor_eng(match):
        kor = match.group(1)
        eng = match.group(2)
        pron = " ".join(ALPHABET_PRONUNCIATION.get(c, c) for c in eng)
        return kor + pron

    def replace_standalone(match):
        eng = match.group(1)
        return " ".join(ALPHABET_PRONUNCIATION.get(c, c) for c in eng)

    text = standalone.sub(replace_standalone, text)
    text = english_hangul.sub(replace_eng_kor, text)
    text = hangul_english.sub(replace_kor_eng, text)

    return text


def convert_alphanumeric_mixed(text: str) -> str:
    """영문자와 숫자가 혼합된 패턴을 처리합니다."""
    pattern = re.compile(
        r"\b([A-Za-z0-9]*[A-Za-z][0-9]+[A-Za-z0-9]*|"
        r"[A-Za-z0-9]*[0-9][A-Za-z]+[A-Za-z0-9]*)\b"
    )

    def replace_alphanum(match):
        text = match.group(1)
        result = []
        for char in text:
            if char.isdigit():
                result.append(NUM_KOR[int(char)])
            elif char in ALPHABET_PRONUNCIATION:
                result.append(ALPHABET_PRONUNCIATION[char])
            else:
                result.append(char)
        return " ".join(result)

    return pattern.sub(replace_alphanum, text)
