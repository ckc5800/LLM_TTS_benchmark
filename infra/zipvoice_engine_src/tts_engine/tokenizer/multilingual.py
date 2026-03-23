# -*- coding: utf-8 -*-
"""Multilingual Tokenizer - 다국어 토크나이저.

여러 언어가 혼합된 텍스트를 처리하는 토크나이저입니다.
언어를 자동 감지하고 적절한 정규화 및 G2P를 적용합니다.

Supported Languages:
- Korean (ko)
- English (en)
- Chinese (zh)
- Japanese (ja)
"""

import re
from functools import reduce
from typing import Dict, List, Optional, Tuple

from tts_engine.core.logging import get_logger
from tts_engine.tokenizer.base import Tokenizer, TokenizerConfig
from tts_engine.tokenizer.espeak import (
    _PIPER_AVAILABLE,
    get_espeak_language_code,
    phonemize_espeak,
)
from tts_engine.tokenizer.normalizer import get_normalizer

logger = get_logger(__name__)

# 선택적 의존성
_PYPINYIN_AVAILABLE = False
_JIEBA_AVAILABLE = False

try:
    from pypinyin import Style, lazy_pinyin
    from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

    _PYPINYIN_AVAILABLE = True
except ImportError:
    logger.debug("pypinyin 미설치 - 중국어 병음 변환 제한됨")

try:
    import jieba

    _JIEBA_AVAILABLE = True
    # jieba 로그 레벨 조정
    import logging as std_logging

    jieba.default_logger.setLevel(std_logging.INFO)
except ImportError:
    logger.debug("jieba 미설치 - 중국어 분절 제한됨")


# Unicode 범위 정의
UNICODE_RANGES = {
    "korean": (0xAC00, 0xD7A3),  # 한글 음절
    "korean_jamo": (0x1100, 0x11FF),  # 한글 자모
    "chinese": (0x4E00, 0x9FFF),  # CJK 통합 한자
    "japanese_hiragana": (0x3040, 0x309F),  # 히라가나
    "japanese_katakana": (0x30A0, 0x30FF),  # 가타카나
    "japanese_katakana_ext": (0x31F0, 0x31FF),  # 가타카나 확장
}


def is_korean(char: str) -> bool:
    """한글인지 확인합니다."""
    code = ord(char)
    return (
        UNICODE_RANGES["korean"][0] <= code <= UNICODE_RANGES["korean"][1]
        or UNICODE_RANGES["korean_jamo"][0]
        <= code
        <= UNICODE_RANGES["korean_jamo"][1]
    )


def is_chinese(char: str) -> bool:
    """중국어(한자)인지 확인합니다."""
    code = ord(char)
    return UNICODE_RANGES["chinese"][0] <= code <= UNICODE_RANGES["chinese"][1]


def is_japanese(char: str) -> bool:
    """일본어(히라가나/가타카나)인지 확인합니다."""
    code = ord(char)
    return (
        UNICODE_RANGES["japanese_hiragana"][0]
        <= code
        <= UNICODE_RANGES["japanese_hiragana"][1]
        or UNICODE_RANGES["japanese_katakana"][0]
        <= code
        <= UNICODE_RANGES["japanese_katakana"][1]
        or UNICODE_RANGES["japanese_katakana_ext"][0]
        <= code
        <= UNICODE_RANGES["japanese_katakana_ext"][1]
    )


def is_alphabet(char: str) -> bool:
    """알파벳인지 확인합니다."""
    return (
        "\u0041" <= char <= "\u005A" or "\u0061" <= char <= "\u007A"  # A-Z  # a-z
    )


def detect_char_language(char: str) -> str:
    """문자의 언어를 감지합니다.

    Args:
        char: 단일 문자

    Returns:
        언어 코드 (ko, en, zh, ja, other)
    """
    if is_korean(char):
        return "ko"
    elif is_chinese(char):
        return "zh"
    elif is_japanese(char):
        return "ja"
    elif is_alphabet(char):
        return "en"
    return "other"


def segment_by_language(text: str) -> List[Tuple[str, str]]:
    """텍스트를 언어별로 분절합니다.

    Args:
        text: 입력 텍스트

    Returns:
        (텍스트 조각, 언어 코드) 튜플 리스트

    Example:
        >>> segment_by_language("안녕 Hello 你好")
        [('안녕 ', 'ko'), ('Hello ', 'en'), ('你好', 'zh')]
    """
    if not text:
        return []

    segments: List[Tuple[str, str]] = []
    current_text = ""
    current_lang = ""

    for char in text:
        char_lang = detect_char_language(char)

        if not current_text:
            # 첫 문자
            current_text = char
            current_lang = char_lang
        elif current_lang == "other":
            # 이전이 other면 현재 언어로 병합
            current_text += char
            current_lang = char_lang
        elif char_lang == "other" or char_lang == current_lang:
            # 같은 언어거나 other면 계속 추가
            current_text += char
        else:
            # 언어 변경 - 이전 세그먼트 저장
            segments.append((current_text, current_lang))
            current_text = char
            current_lang = char_lang

    # 마지막 세그먼트
    if current_text:
        segments.append((current_text, current_lang))

    return segments


class MultilingualTokenizer(Tokenizer):
    """다국어 토크나이저.

    여러 언어가 혼합된 텍스트를 자동으로 분절하고
    각 언어에 적합한 G2P를 적용합니다.
    """

    def __init__(
        self,
        config: Optional[TokenizerConfig] = None,
        token_file: Optional[str] = None,
        default_lang: str = "ko",
        use_pinyin_for_chinese: bool = True,
    ):
        """MultilingualTokenizer를 초기화합니다.

        Args:
            config: 토크나이저 설정
            token_file: 토큰 파일 경로
            default_lang: 기본 언어 코드
            use_pinyin_for_chinese: 중국어에 병음 사용 여부
        """
        super().__init__(config)

        if not _PIPER_AVAILABLE:
            raise RuntimeError(
                "piper_phonemize가 설치되지 않음. "
                "pip install piper_phonemize 필요"
            )

        self.default_lang = default_lang
        self.use_pinyin_for_chinese = use_pinyin_for_chinese

        # 언어별 정규화기 캐시
        self._normalizers: Dict[str, object] = {}

        # 토큰 파일 로드
        if token_file:
            self.load_tokens(token_file)

    def get_normalizer(self, lang: str):
        """언어에 맞는 정규화기를 반환합니다."""
        if lang not in self._normalizers:
            self._normalizers[lang] = get_normalizer(lang)
        return self._normalizers.get(lang)

    def normalize_text(self, text: str, lang: str) -> str:
        """텍스트를 정규화합니다.

        Args:
            text: 입력 텍스트
            lang: 언어 코드

        Returns:
            정규화된 텍스트
        """
        normalizer = self.get_normalizer(lang)
        if normalizer:
            return normalizer.normalize(text)
        return text

    def tokenize_korean(self, text: str) -> List[str]:
        """한국어 텍스트를 토큰화합니다."""
        # 정규화
        text = self.normalize_text(text, "ko")

        # espeak으로 phoneme 변환
        try:
            tokens = phonemize_espeak(text, "ko")
            return reduce(lambda x, y: x + y, tokens) if tokens else []
        except Exception as e:
            logger.warning(f"한국어 토큰화 실패: {e}")
            return []

    def tokenize_english(self, text: str) -> List[str]:
        """영어 텍스트를 토큰화합니다."""
        # 정규화
        text = self.normalize_text(text, "en")

        # espeak으로 phoneme 변환
        try:
            tokens = phonemize_espeak(text, "en-us")
            return reduce(lambda x, y: x + y, tokens) if tokens else []
        except Exception as e:
            logger.warning(f"영어 토큰화 실패: {e}")
            return []

    def tokenize_chinese(self, text: str) -> List[str]:
        """중국어 텍스트를 토큰화합니다."""
        # 정규화
        text = self.normalize_text(text, "zh")

        if self.use_pinyin_for_chinese and _PYPINYIN_AVAILABLE and _JIEBA_AVAILABLE:
            # 병음 기반 토큰화
            return self._tokenize_chinese_pinyin(text)
        else:
            # espeak 기반
            try:
                tokens = phonemize_espeak(text, "cmn")
                return reduce(lambda x, y: x + y, tokens) if tokens else []
            except Exception as e:
                logger.warning(f"중국어 토큰화 실패: {e}")
                return []

    def _tokenize_chinese_pinyin(self, text: str) -> List[str]:
        """병음 기반 중국어 토큰화."""
        try:
            # jieba로 분절
            segs = list(jieba.cut(text))

            # 병음 변환
            pinyins = lazy_pinyin(
                segs,
                style=Style.TONE3,
                tone_sandhi=True,
                neutral_tone_with_five=True,
            )

            phones = []
            for pinyin in pinyins:
                # 유효한 병음인지 확인 (알파벳 + 성조 숫자)
                if pinyin and pinyin[:-1].isalpha() and pinyin[-1] in "12345":
                    phones.extend(self._separate_pinyin(pinyin))
                else:
                    phones.append(pinyin)

            return phones
        except Exception as e:
            logger.warning(f"병음 토큰화 실패: {e}")
            return []

    def _separate_pinyin(self, pinyin: str) -> List[str]:
        """병음을 성모/운모로 분리합니다."""
        phones = []
        initial = to_initials(pinyin, strict=False)
        final = to_finals_tone3(
            pinyin,
            strict=False,
            neutral_tone_with_five=True,
        )

        # 성모 (espeak 토큰과 구분을 위해 '0' 추가)
        if initial:
            phones.append(initial + "0")
        if final:
            phones.append(final)

        return phones

    def tokenize_japanese(self, text: str) -> List[str]:
        """일본어 텍스트를 토큰화합니다."""
        # 정규화
        text = self.normalize_text(text, "ja")

        # espeak으로 phoneme 변환
        try:
            tokens = phonemize_espeak(text, "ja")
            return reduce(lambda x, y: x + y, tokens) if tokens else []
        except Exception as e:
            logger.warning(f"일본어 토큰화 실패: {e}")
            return []

    def tokenize_segment(self, text: str, lang: str) -> List[str]:
        """언어에 맞게 텍스트를 토큰화합니다.

        Args:
            text: 텍스트 조각
            lang: 언어 코드

        Returns:
            토큰 리스트
        """
        if lang == "ko":
            return self.tokenize_korean(text)
        elif lang == "en":
            return self.tokenize_english(text)
        elif lang == "zh":
            return self.tokenize_chinese(text)
        elif lang == "ja":
            return self.tokenize_japanese(text)
        else:
            # 기본 언어로 처리
            espeak_lang = get_espeak_language_code(self.default_lang)
            try:
                tokens = phonemize_espeak(text, espeak_lang)
                return reduce(lambda x, y: x + y, tokens) if tokens else []
            except Exception:
                return []

    def texts_to_tokens(
        self,
        texts: List[str],
        lang: Optional[str] = None,
    ) -> List[List[str]]:
        """텍스트 리스트를 토큰 시퀀스로 변환합니다.

        Args:
            texts: 텍스트 리스트
            lang: 언어 코드 (None이면 자동 감지)

        Returns:
            토큰 리스트의 리스트
        """
        result = []

        for text in texts:
            if not text or not text.strip():
                result.append([])
                continue

            if lang:
                # 단일 언어 모드
                tokens = self.tokenize_segment(text, lang)
            else:
                # 다국어 자동 감지 모드
                segments = segment_by_language(text)
                tokens = []
                for seg_text, seg_lang in segments:
                    seg_tokens = self.tokenize_segment(seg_text, seg_lang)
                    tokens.extend(seg_tokens)

            result.append(tokens)

        return result

    def tokenize(
        self,
        text: str,
        lang: Optional[str] = None,
    ) -> List[str]:
        """단일 텍스트를 토큰화합니다.

        Args:
            text: 입력 텍스트
            lang: 언어 코드 (None이면 자동 감지)

        Returns:
            토큰 리스트
        """
        return self.texts_to_tokens([text], lang)[0]

    def __repr__(self) -> str:
        return (
            f"MultilingualTokenizer("
            f"default_lang={self.default_lang}, "
            f"vocab_size={self.vocab_size}, "
            f"use_pinyin={self.use_pinyin_for_chinese})"
        )


def create_tokenizer(
    tokenizer_type: str = "multilingual",
    token_file: Optional[str] = None,
    lang: str = "ko",
    **kwargs,
) -> Tokenizer:
    """토크나이저를 생성합니다.

    Args:
        tokenizer_type: 토크나이저 타입 (espeak, multilingual)
        token_file: 토큰 파일 경로
        lang: 기본 언어
        **kwargs: 추가 인자

    Returns:
        Tokenizer 인스턴스
    """
    from tts_engine.tokenizer.espeak import EspeakTokenizer

    if tokenizer_type == "espeak":
        return EspeakTokenizer(token_file=token_file, lang=lang)
    elif tokenizer_type == "multilingual":
        return MultilingualTokenizer(
            token_file=token_file,
            default_lang=lang,
            **kwargs,
        )
    else:
        raise ValueError(f"지원하지 않는 토크나이저 타입: {tokenizer_type}")
