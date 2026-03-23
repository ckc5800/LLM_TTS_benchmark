# -*- coding: utf-8 -*-
"""Base Tokenizer - 토크나이저 기본 인터페이스.

모든 토크나이저의 추상 베이스 클래스와 공통 설정을 정의합니다.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from tts_engine.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TokenizerConfig:
    """토크나이저 설정."""

    # 토큰 파일 경로
    token_file: Optional[str] = None

    # 기본 언어
    default_language: str = "ko"

    # 지원 언어 목록
    supported_languages: List[str] = field(
        default_factory=lambda: ["ko", "en", "zh", "ja"]
    )

    # 언어별 espeak 코드 매핑
    language_codes: Dict[str, str] = field(
        default_factory=lambda: {
            "ko": "ko",  # 한국어
            "en": "en-us",  # 영어 (미국)
            "zh": "cmn",  # 중국어 (만다린)
            "ja": "ja",  # 일본어
        }
    )

    # 텍스트 정규화 옵션
    normalize_text: bool = True

    # 중국어 병음 사용
    use_pinyin_for_chinese: bool = True


class Tokenizer(ABC):
    """토크나이저 추상 베이스 클래스.

    모든 토크나이저는 이 클래스를 상속해야 합니다.
    텍스트를 음소(phoneme) 토큰으로 변환합니다.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        """토크나이저를 초기화합니다.

        Args:
            config: 토크나이저 설정
        """
        self.config = config or TokenizerConfig()
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.pad_id: int = 0
        self._initialized = False

    def load_tokens(self, token_file: Optional[str] = None) -> bool:
        """토큰 파일을 로드합니다.

        Args:
            token_file: 토큰 파일 경로 ('{token}\\t{id}' 형식)

        Returns:
            성공 여부
        """
        token_path = token_file or self.config.token_file
        if not token_path:
            logger.warning("토큰 파일이 지정되지 않음")
            return False

        path = Path(token_path)
        if not path.exists():
            logger.error(f"토큰 파일을 찾을 수 없음: {token_path}")
            return False

        try:
            self.token2id = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) != 2:
                        logger.warning(f"잘못된 토큰 형식: {line}")
                        continue
                    token, token_id = parts[0], int(parts[1])
                    if token in self.token2id:
                        logger.warning(f"중복 토큰: {token}")
                        continue
                    self.token2id[token] = token_id

            self.id2token = {v: k for k, v in self.token2id.items()}
            self.vocab_size = len(self.token2id)

            # 패딩 ID 설정
            if "_" in self.token2id:
                self.pad_id = self.token2id["_"]
            elif "<pad>" in self.token2id:
                self.pad_id = self.token2id["<pad>"]
            else:
                self.pad_id = 0

            self._initialized = True
            logger.info(f"토큰 로드 완료: {self.vocab_size}개 토큰")
            return True

        except Exception as e:
            logger.error(f"토큰 파일 로드 실패: {e}")
            return False

    @property
    def has_tokens(self) -> bool:
        """토큰이 로드되었는지 확인합니다."""
        return self._initialized and len(self.token2id) > 0

    @abstractmethod
    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        """텍스트를 토큰 시퀀스로 변환합니다.

        Args:
            texts: 텍스트 리스트

        Returns:
            토큰 리스트의 리스트
        """
        pass

    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        """텍스트를 토큰 ID 시퀀스로 변환합니다.

        Args:
            texts: 텍스트 리스트

        Returns:
            토큰 ID 리스트의 리스트
        """
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def tokens_to_token_ids(
        self, tokens_list: List[List[str]]
    ) -> List[List[int]]:
        """토큰을 토큰 ID로 변환합니다.

        Args:
            tokens_list: 토큰 리스트의 리스트

        Returns:
            토큰 ID 리스트의 리스트
        """
        if not self.has_tokens:
            raise RuntimeError("토큰이 로드되지 않음. load_tokens()를 먼저 호출하세요.")

        token_ids_list = []
        for tokens in tokens_list:
            token_ids = []
            for t in tokens:
                if t not in self.token2id:
                    logger.debug(f"OOV 토큰 스킵: {t}")
                    continue
                token_ids.append(self.token2id[t])
            token_ids_list.append(token_ids)

        return token_ids_list

    def token_ids_to_tokens(
        self, token_ids_list: List[List[int]]
    ) -> List[List[str]]:
        """토큰 ID를 토큰으로 변환합니다.

        Args:
            token_ids_list: 토큰 ID 리스트의 리스트

        Returns:
            토큰 리스트의 리스트
        """
        if not self.has_tokens:
            raise RuntimeError("토큰이 로드되지 않음")

        tokens_list = []
        for token_ids in token_ids_list:
            tokens = []
            for tid in token_ids:
                if tid in self.id2token:
                    tokens.append(self.id2token[tid])
            tokens_list.append(tokens)

        return tokens_list

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vocab_size={self.vocab_size}, "
            f"default_lang={self.config.default_language})"
        )
