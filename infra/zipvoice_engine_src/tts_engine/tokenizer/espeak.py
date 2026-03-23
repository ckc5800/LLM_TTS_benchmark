# -*- coding: utf-8 -*-
"""Espeak Tokenizer - espeak-ng 기반 G2P 토크나이저.

espeak-ng를 사용하여 텍스트를 IPA phoneme으로 변환합니다.

Supported Languages:
- ko: 한국어
- en-us: 영어 (미국)
- cmn: 중국어 (만다린)
- ja: 일본어

Requirements:
- piper_phonemize: pip install piper_phonemize
- espeak-ng: apt install espeak-ng
"""

from functools import reduce
from typing import List, Optional

from tts_engine.core.logging import get_logger
from tts_engine.tokenizer.base import Tokenizer, TokenizerConfig

logger = get_logger(__name__)

# piper_phonemize 임포트 시도
_PIPER_AVAILABLE = False
phonemize_espeak = None

try:
    from piper_phonemize import phonemize_espeak as _phonemize_espeak

    phonemize_espeak = _phonemize_espeak
    _PIPER_AVAILABLE = True
except ImportError:
    logger.warning(
        "piper_phonemize를 찾을 수 없음. "
        "설치: pip install piper_phonemize -f "
        "https://k2-fsa.github.io/icefall/piper_phonemize.html"
    )


# espeak-ng 언어 코드 매핑
ESPEAK_LANGUAGE_CODES = {
    # 한국어
    "ko": "ko",
    "kor": "ko",
    "korean": "ko",
    # 영어
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
    "eng": "en-us",
    "english": "en-us",
    # 중국어
    "zh": "cmn",
    "zh-cn": "cmn",
    "cmn": "cmn",
    "chinese": "cmn",
    "mandarin": "cmn",
    # 일본어
    "ja": "ja",
    "jp": "ja",
    "jpn": "ja",
    "japanese": "ja",
    # 기타
    "de": "de",
    "german": "de",
    "fr": "fr",
    "french": "fr",
    "es": "es",
    "spanish": "es",
}


def get_espeak_language_code(lang: str) -> str:
    """언어 코드를 espeak-ng 코드로 변환합니다.

    Args:
        lang: 언어 코드 (ko, en, zh, ja 등)

    Returns:
        espeak-ng 언어 코드
    """
    lang_lower = lang.lower().strip()
    return ESPEAK_LANGUAGE_CODES.get(lang_lower, lang_lower)


class EspeakTokenizer(Tokenizer):
    """espeak-ng 기반 G2P 토크나이저.

    espeak-ng를 사용하여 텍스트를 IPA phoneme 시퀀스로 변환합니다.
    """

    def __init__(
        self,
        config: Optional[TokenizerConfig] = None,
        token_file: Optional[str] = None,
        lang: str = "ko",
    ):
        """EspeakTokenizer를 초기화합니다.

        Args:
            config: 토크나이저 설정
            token_file: 토큰 파일 경로
            lang: 기본 언어 코드

        Raises:
            RuntimeError: piper_phonemize가 설치되지 않은 경우
        """
        super().__init__(config)

        if not _PIPER_AVAILABLE:
            raise RuntimeError(
                "piper_phonemize가 설치되지 않음. "
                "설치: pip install piper_phonemize -f "
                "https://k2-fsa.github.io/icefall/piper_phonemize.html"
            )

        self.lang = get_espeak_language_code(lang)

        # 토큰 파일 로드
        if token_file:
            self.load_tokens(token_file)

    def g2p(self, text: str, lang: Optional[str] = None) -> List[str]:
        """텍스트를 phoneme 시퀀스로 변환합니다.

        Args:
            text: 입력 텍스트
            lang: 언어 코드 (None이면 기본 언어 사용)

        Returns:
            phoneme 토큰 리스트
        """
        if not text or not text.strip():
            return []

        language = get_espeak_language_code(lang) if lang else self.lang

        try:
            # piper_phonemize는 리스트의 리스트를 반환
            # [[phoneme1, phoneme2, ...], [phoneme3, ...], ...]
            tokens = phonemize_espeak(text, language)
            # 평탄화
            tokens = reduce(lambda x, y: x + y, tokens) if tokens else []
            return tokens
        except Exception as e:
            logger.warning(f"G2P 변환 실패 (lang={language}): {e}")
            return []

    def texts_to_tokens(
        self,
        texts: List[str],
        lang: Optional[str] = None,
    ) -> List[List[str]]:
        """텍스트 리스트를 토큰 시퀀스로 변환합니다.

        Args:
            texts: 텍스트 리스트
            lang: 언어 코드

        Returns:
            토큰 리스트의 리스트
        """
        return [self.g2p(text, lang) for text in texts]

    def set_language(self, lang: str) -> None:
        """기본 언어를 변경합니다.

        Args:
            lang: 언어 코드
        """
        self.lang = get_espeak_language_code(lang)
        logger.debug(f"언어 변경: {self.lang}")

    @property
    def supported_languages(self) -> List[str]:
        """지원되는 언어 목록을 반환합니다."""
        return list(ESPEAK_LANGUAGE_CODES.keys())

    def __repr__(self) -> str:
        return (
            f"EspeakTokenizer("
            f"lang={self.lang}, "
            f"vocab_size={self.vocab_size})"
        )


def check_espeak_available() -> bool:
    """espeak-ng가 사용 가능한지 확인합니다.

    Returns:
        사용 가능 여부
    """
    if not _PIPER_AVAILABLE:
        return False

    try:
        # 간단한 테스트
        result = phonemize_espeak("test", "en-us")
        return len(result) > 0
    except Exception:
        return False


def check_language_support(lang: str) -> bool:
    """특정 언어가 지원되는지 확인합니다.

    Args:
        lang: 언어 코드

    Returns:
        지원 여부
    """
    if not _PIPER_AVAILABLE:
        return False

    espeak_lang = get_espeak_language_code(lang)
    try:
        result = phonemize_espeak("test", espeak_lang)
        return len(result) > 0
    except Exception:
        return False
