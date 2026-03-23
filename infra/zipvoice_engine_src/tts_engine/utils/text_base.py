# -*- coding: utf-8 -*-
"""기본 텍스트 처리 유틸리티."""

import re
import unicodedata
from typing import List, Set

# 구두점 정의
PUNCTUATION: Set[str] = {
    ";", ":", ",", ".", "!", "?",
    "；", "：", "，", "。", "！", "？",
    "、", "…", "·",
}

# 문장 종결 구두점
SENTENCE_END: Set[str] = {".", "!", "?", "。", "！", "？", "…"}


def normalize_text(text: str) -> str:
    """텍스트를 정규화합니다."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """텍스트에서 특수문자를 정리합니다."""
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\t ")
    return normalize_text(text)


def add_punctuation(text: str) -> str:
    """문장 끝에 구두점이 없으면 추가합니다."""
    text = text.strip()
    if not text:
        return text
    if text[-1] not in PUNCTUATION:
        text += "."
    return text


def split_sentences(text: str) -> List[str]:
    """텍스트를 문장 단위로 분할합니다."""
    pattern = r"([.!?。！？…]+)"
    parts = re.split(pattern, text)

    sentences = []
    current = ""

    for part in parts:
        if not part:
            continue
        if any(p in part for p in SENTENCE_END):
            current += part
            if current.strip():
                sentences.append(current.strip())
            current = ""
        else:
            current += part

    if current.strip():
        sentences.append(current.strip())

    return sentences


def chunk_text(text: str, max_length: int = 200, overlap: int = 0) -> List[str]:
    """텍스트를 청크로 분할합니다."""
    if len(text) <= max_length:
        return [text]

    sentences = split_sentences(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            for i in range(0, len(sentence), max_length - overlap):
                chunk = sentence[i:i + max_length]
                chunks.append(chunk.strip())
        elif len(current_chunk) + len(sentence) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " "
            current_chunk += sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_by_punctuation(tokens: List[str], max_tokens: int = 100) -> List[List[str]]:
    """토큰 리스트를 구두점 기준으로 청킹합니다."""
    sentences = []
    current_sentence = []

    for token in tokens:
        if (
            len(current_sentence) == 0
            and len(sentences) != 0
            and (token in PUNCTUATION or token == " ")
        ):
            sentences[-1].append(token)
        else:
            current_sentence.append(token)
            if token in PUNCTUATION:
                sentences.append(current_sentence)
                current_sentence = []

    if current_sentence:
        sentences.append(current_sentence)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk.extend(sentence)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def estimate_duration(text: str, chars_per_second: float = 5.0) -> float:
    """텍스트의 예상 발화 시간을 계산합니다."""
    char_count = len(text.replace(" ", ""))
    return char_count / chars_per_second


def contains_korean(text: str) -> bool:
    """한글 포함 여부를 확인합니다."""
    for char in text:
        if "\uac00" <= char <= "\ud7a3":
            return True
        if "\u1100" <= char <= "\u11ff":
            return True
    return False


def contains_english(text: str) -> bool:
    """영어 포함 여부를 확인합니다."""
    return bool(re.search(r"[a-zA-Z]", text))


def detect_language(text: str) -> str:
    """텍스트의 언어를 감지합니다."""
    has_korean = contains_korean(text)
    has_english = contains_english(text)

    if has_korean and has_english:
        return "mixed"
    elif has_korean:
        return "ko"
    elif has_english:
        return "en"
    else:
        return "unknown"


def validate_text(text: str, max_length: int = 2000, min_length: int = 1) -> tuple:
    """텍스트를 검증합니다."""
    if not text:
        return False, "Text is empty"

    text = text.strip()

    if len(text) < min_length:
        return False, f"Text is too short (min: {min_length})"

    if len(text) > max_length:
        return False, f"Text is too long (max: {max_length})"

    return True, None


def remove_emojis(text: str) -> str:
    """이모지를 제거합니다."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U00002600-\U000026FF"
        "\U00002702-\U000027B0"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def process_parentheses(text: str, enable: bool = True, max_length: int = 0) -> str:
    """괄호 안 내용을 처리합니다."""
    patterns = [
        r"\(([^)]*)\)",
        r"\[([^\]]*)\]",
        r"\{([^}]*)\}",
    ]

    for pattern in patterns:
        def replace_paren(match):
            content = match.group(1)
            if not enable:
                return ""
            content_len = len(content.strip())
            if content_len <= max_length:
                return content
            return ""

        text = re.sub(pattern, replace_paren, text)

    return text
