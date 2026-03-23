# -*- coding: utf-8 -*-
"""Text Normalizer - 언어별 텍스트 정규화.

각 언어에 맞는 텍스트 정규화를 제공합니다.
- 한국어: 숫자, 단위, 시간, 전화번호 변환
- 영어: 약어, 숫자 확장
- 중국어: 숫자 → 한자, 구두점 변환
- 일본어: 숫자 → 한자, 구두점 변환
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Optional

from tts_engine.core.logging import get_logger

logger = get_logger(__name__)

# 선택적 의존성
_CN2AN_AVAILABLE = False
_INFLECT_AVAILABLE = False

try:
    import cn2an

    _CN2AN_AVAILABLE = True
except ImportError:
    logger.debug("cn2an 미설치 - 중국어 숫자 변환 제한됨")

try:
    import inflect

    _INFLECT_AVAILABLE = True
except ImportError:
    logger.debug("inflect 미설치 - 영어 숫자 변환 제한됨")


class TextNormalizer(ABC):
    """텍스트 정규화 추상 베이스 클래스."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """텍스트를 정규화합니다.

        Args:
            text: 입력 텍스트

        Returns:
            정규화된 텍스트
        """
        pass

    def map_punctuations(self, text: str) -> str:
        """동아시아 구두점을 ASCII로 변환합니다."""
        mappings = {
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "；": ";",
            "：": ":",
            "、": ",",
            "'": "'",
            """: '"',
            """: '"',
            "'": "'",
            "⋯": "…",
            "···": "…",
            "・・・": "…",
            "...": "…",
        }
        for old, new in mappings.items():
            text = text.replace(old, new)
        return text


class KoreanTextNormalizer(TextNormalizer):
    """한국어 텍스트 정규화.

    숫자, 단위, 시간, 전화번호 등을 한글로 변환합니다.
    utils/text.py의 normalize_korean 함수를 래핑합니다.
    """

    def __init__(
        self,
        parentheses_enable: bool = True,
        parentheses_max_length: int = 0,
    ):
        """KoreanTextNormalizer를 초기화합니다.

        Args:
            parentheses_enable: 괄호 안 내용 읽기 여부
            parentheses_max_length: 괄호 안 내용 최대 길이
        """
        self.parentheses_enable = parentheses_enable
        self.parentheses_max_length = parentheses_max_length

    def normalize(self, text: str) -> str:
        """한국어 텍스트를 정규화합니다."""
        # utils/text.py의 함수 사용
        from tts_engine.utils.text import normalize_korean

        return normalize_korean(
            text,
            parentheses_enable=self.parentheses_enable,
            parentheses_max_length=self.parentheses_max_length,
        )


class EnglishTextNormalizer(TextNormalizer):
    """영어 텍스트 정규화.

    약어 확장, 숫자 → 영어 변환을 수행합니다.
    """

    def __init__(self):
        """EnglishTextNormalizer를 초기화합니다."""
        # 약어 패턴
        self._abbreviations = [
            (re.compile(r"\b%s\b" % x[0], re.IGNORECASE), x[1])
            for x in [
                ("mrs", "misess"),
                ("mr", "mister"),
                ("dr", "doctor"),
                ("st", "saint"),
                ("co", "company"),
                ("jr", "junior"),
                ("maj", "major"),
                ("gen", "general"),
                ("drs", "doctors"),
                ("rev", "reverend"),
                ("lt", "lieutenant"),
                ("hon", "honorable"),
                ("sgt", "sergeant"),
                ("capt", "captain"),
                ("esq", "esquire"),
                ("ltd", "limited"),
                ("col", "colonel"),
                ("ft", "fort"),
                ("etc", "et cetera"),
                ("btw", "by the way"),
            ]
        ]

        # inflect 엔진
        self._inflect = inflect.engine() if _INFLECT_AVAILABLE else None

        # 숫자 패턴
        self._comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
        self._decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
        self._percent_number_re = re.compile(r"([0-9\.\,]*[0-9]+%)")
        self._pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
        self._dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
        self._fraction_re = re.compile(r"([0-9]+)/([0-9]+)")
        self._ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
        self._number_re = re.compile(r"[0-9]+")

    def normalize(self, text: str) -> str:
        """영어 텍스트를 정규화합니다."""
        text = self.expand_abbreviations(text)
        if self._inflect:
            text = self.normalize_numbers(text)
        return text

    def expand_abbreviations(self, text: str) -> str:
        """약어를 확장합니다."""
        for regex, replacement in self._abbreviations:
            text = re.sub(regex, replacement, text)
        return text

    def normalize_numbers(self, text: str) -> str:
        """숫자를 영어로 변환합니다."""
        if not self._inflect:
            return text

        text = re.sub(self._comma_number_re, self._remove_commas, text)
        text = re.sub(self._pounds_re, r"\1 pounds", text)
        text = re.sub(self._dollars_re, self._expand_dollars, text)
        text = re.sub(self._fraction_re, self._expand_fraction, text)
        text = re.sub(self._decimal_number_re, self._expand_decimal, text)
        text = re.sub(self._percent_number_re, self._expand_percent, text)
        text = re.sub(self._ordinal_re, self._expand_ordinal, text)
        text = re.sub(self._number_re, self._expand_number, text)
        return text

    def _remove_commas(self, m) -> str:
        return m.group(1).replace(",", "")

    def _expand_dollars(self, m) -> str:
        match = m.group(1)
        parts = match.split(".")
        if len(parts) > 2:
            return f" {match} dollars "
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if dollars and cents:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            cent_unit = "cent" if cents == 1 else "cents"
            return f" {dollars} {dollar_unit}, {cents} {cent_unit} "
        elif dollars:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            return f" {dollars} {dollar_unit} "
        elif cents:
            cent_unit = "cent" if cents == 1 else "cents"
            return f" {cents} {cent_unit} "
        return " zero dollars "

    def _expand_fraction(self, m) -> str:
        num = int(m.group(1))
        den = int(m.group(2))
        if num == 1 and den == 2:
            return " one half "
        if num == 1 and den == 4:
            return " one quarter "
        num_word = self._inflect.number_to_words(num)
        den_word = self._inflect.ordinal(self._inflect.number_to_words(den))
        return f" {num_word} {den_word} "

    def _expand_decimal(self, m) -> str:
        return m.group(1).replace(".", " point ")

    def _expand_percent(self, m) -> str:
        return m.group(1).replace("%", " percent ")

    def _expand_ordinal(self, m) -> str:
        return f" {self._inflect.number_to_words(m.group(0))} "

    def _expand_number(self, m) -> str:
        num = int(m.group(0))
        if 1000 < num < 3000:
            if num == 2000:
                return " two thousand "
            elif 2000 < num < 2010:
                return f" two thousand {self._inflect.number_to_words(num % 100)} "
            elif num % 100 == 0:
                return f" {self._inflect.number_to_words(num // 100)} hundred "
            else:
                words = self._inflect.number_to_words(
                    num, andword="", zero="oh", group=2
                )
                return f" {words.replace(', ', ' ')} "
        return f" {self._inflect.number_to_words(num, andword='')} "


class ChineseTextNormalizer(TextNormalizer):
    """중국어 텍스트 정규화.

    숫자를 한자로 변환하고 구두점을 정규화합니다.
    """

    def normalize(self, text: str) -> str:
        """중국어 텍스트를 정규화합니다."""
        # 구두점 변환
        text = self.map_punctuations(text)

        # 숫자 → 한자
        if _CN2AN_AVAILABLE:
            try:
                text = cn2an.transform(text, "an2cn")
            except Exception as e:
                logger.debug(f"cn2an 변환 실패: {e}")

        return text


class JapaneseTextNormalizer(TextNormalizer):
    """일본어 텍스트 정규화.

    숫자를 일본어로 변환하고 구두점을 정규화합니다.
    """

    # 일본어 숫자 매핑
    JP_DIGITS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    JP_UNITS = ["", "十", "百", "千"]
    JP_HIGH_UNITS = ["", "万", "億", "兆"]

    def normalize(self, text: str) -> str:
        """일본어 텍스트를 정규화합니다."""
        # 구두점 변환
        text = self.map_punctuations(text)

        # 숫자 → 일본어
        text = self._convert_numbers(text)

        return text

    def _convert_numbers(self, text: str) -> str:
        """숫자를 일본어로 변환합니다."""

        def repl(m):
            num_str = m.group(0)
            try:
                return self._number_to_japanese(int(num_str))
            except ValueError:
                return num_str

        return re.sub(r"\d+", repl, text)

    def _number_to_japanese(self, num: int) -> str:
        """정수를 일본어로 변환합니다."""
        if num == 0:
            return "零"

        if num < 0:
            return "マイナス" + self._number_to_japanese(-num)

        result = ""
        num_str = str(num)
        # 4자리씩 그룹화
        num_str = num_str.zfill(((len(num_str) - 1) // 4 + 1) * 4)
        chunks = [num_str[i : i + 4] for i in range(0, len(num_str), 4)]

        for i, chunk in enumerate(chunks):
            chunk_result = ""
            for j, digit in enumerate(chunk):
                n = int(digit)
                unit_idx = 4 - j - 1
                if n != 0:
                    # 十, 百, 千 앞의 一 생략 (일부 경우)
                    if n == 1 and unit_idx > 0:
                        chunk_result += self.JP_UNITS[unit_idx]
                    else:
                        chunk_result += self.JP_DIGITS[n] + self.JP_UNITS[unit_idx]

            if chunk_result:
                high_unit_idx = len(chunks) - i - 1
                result += chunk_result + self.JP_HIGH_UNITS[high_unit_idx]

        return result if result else "零"


# 언어별 정규화기 매핑
NORMALIZERS: Dict[str, type] = {
    "ko": KoreanTextNormalizer,
    "kor": KoreanTextNormalizer,
    "korean": KoreanTextNormalizer,
    "en": EnglishTextNormalizer,
    "en-us": EnglishTextNormalizer,
    "eng": EnglishTextNormalizer,
    "english": EnglishTextNormalizer,
    "zh": ChineseTextNormalizer,
    "cmn": ChineseTextNormalizer,
    "chinese": ChineseTextNormalizer,
    "ja": JapaneseTextNormalizer,
    "jp": JapaneseTextNormalizer,
    "jpn": JapaneseTextNormalizer,
    "japanese": JapaneseTextNormalizer,
}


def get_normalizer(lang: str) -> Optional[TextNormalizer]:
    """언어에 맞는 정규화기를 반환합니다.

    Args:
        lang: 언어 코드

    Returns:
        TextNormalizer 인스턴스 또는 None
    """
    lang_lower = lang.lower().strip()
    normalizer_class = NORMALIZERS.get(lang_lower)
    if normalizer_class:
        return normalizer_class()
    return None
