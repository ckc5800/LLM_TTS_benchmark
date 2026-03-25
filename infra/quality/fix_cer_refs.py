"""
CER 참조음성 수정 스크립트
ZH/JA/EN 모델 중 iu_long 참조로 오염된 CER을
언어 매칭 참조음성(zh_male, ja_male 등)으로 재측정.
"""
import json
import os
import re
import sys
import time

INFRA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(INFRA_DIR)
sys.path.insert(0, INFRA_DIR)
from benchmark.core import TEST_TEXTS

WAV_BASE = os.path.join(ROOT_DIR, "results", "results_multilingual", "wav")
CER_PATH = os.path.join(INFRA_DIR, "quality", "cer_multilingual.json")

# ─── 정규화 ────────────────────────────────────────────────────────────────────

def normalize_cjk(text: str) -> str:
    text = re.sub(r"[\s\u3000]+", "", text)
    text = re.sub(r'[。、！？「」『』【】〔〕…・,!?"\'()（）\[\]{}]', "", text)
    return text.strip()

def normalize_en(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()

def normalize_ko(text: str) -> str:
    text = re.sub(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ]", "", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()

def _edit_distance(seq1, seq2) -> int:
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if seq1[i-1] == seq2[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]

def compute_cer(ref: str, hyp: str, lang: str) -> float:
    if lang == "en":
        r = normalize_en(ref).split()
        h = normalize_en(hyp).split()
    elif lang == "ko":
        r = list(normalize_ko(ref).replace(" ", ""))
        h = list(normalize_ko(hyp).replace(" ", ""))
    else:  # ja, zh
        r = list(normalize_cjk(ref))
        h = list(normalize_cjk(hyp))
    if not r:
        return 0.0
    return _edit_distance(r, h) / len(r)

# ─── Whisper ───────────────────────────────────────────────────────────────────

_whisper_model = None

def get_whisper(device="cuda"):
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print(f"  Whisper large-v3 로딩 ({device})...", flush=True)
        _whisper_model = WhisperModel(
            "large-v3", device=device,
            compute_type="float16" if device == "cuda" else "int8"
        )
    return _whisper_model

def transcribe(wav_path: str, lang: str, model) -> str:
    lang_map = {"ko": "ko", "en": "en", "ja": "ja", "zh": "zh"}
    try:
        segs, _ = model.transcribe(wav_path, language=lang_map[lang], beam_size=5, vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()
    except Exception as e:
        print(f"  [ERR] {os.path.basename(wav_path)}: {e}")
        return ""

# ─── WAV 경로 찾기 ──────────────────────────────────────────────────────────────

def find_wav(lang: str, model: str, ref_key: str, text_key: str) -> str:
    """
    새 형식: {model}__{ref_key}__{lang}_{text_key}_0.wav
    구 형식 (zh_female 전용): {model}__{lang}_{text_key}_0.wav
    """
    wav_dir = os.path.join(WAV_BASE, lang, "A")

    # 새 형식: {model}__{ref_key}__{text_key}_0.wav  (text_key 이미 lang 접두사 포함)
    new = os.path.join(wav_dir, f"{model}__{ref_key}__{text_key}_0.wav")
    if os.path.exists(new):
        return new

    # 구 형식 (zh_female 등 old-style): {model}__{text_key}_0.wav
    old = os.path.join(wav_dir, f"{model}__{text_key}_0.wav")
    if os.path.exists(old):
        return old

    return None

# ─── 측정 대상 정의 ────────────────────────────────────────────────────────────

# (lang, model_key, ref_key, text_keys_list)
TARGETS = [
    # ZH — iu_long 오염 모델, zh_male 재측정
    ("zh", "f5tts",       "zh_male",    ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    ("zh", "f5tts",       "zh_female",  ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    ("zh", "fish_speech", "zh_male",    ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    ("zh", "fish_speech", "zh_female",  ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    ("zh", "kani",        "zh_male",    ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    ("zh", "kani",        "zh_female",  ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    ("zh", "spark_tts",   "zh_male",    ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    ("zh", "spark_tts",   "zh_female",  ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    # JA — 누락 모델 측정
    ("ja", "fish_speech", "ja_male",    ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "fish_speech", "ja_female",  ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "index_tts2",  "ja_male",    ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "kokoro",      "ja_male",    ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "kokoro",      "ja_female",  ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "mio_tts_1.7b","ja_male",    ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "mio_tts_1.7b","ja_female",  ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "mio_tts_2.6b","ja_male",    ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "mio_tts_2.6b","ja_female",  ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "outetss",     "default",    ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "voicecraftx", "ja_female",  ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "voicecraftx", "ja_male",    ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    # EN — voicecraftx 재측정
    ("en", "voicecraftx", "en_female",  ["en_short", "en_medium", "en_long", "en_numbers", "en_technical",
                                          "en_conversational", "en_formal", "en_punctuation", "en_names", "en_emotional"]),
    ("en", "voicecraftx", "en_male",    ["en_short", "en_medium", "en_long", "en_numbers", "en_technical",
                                          "en_conversational", "en_formal", "en_punctuation", "en_names", "en_emotional"]),
    # ZH — qwen3 재측정 (zh_male 우선, zh_female는 생성 완료 후)
    ("zh", "qwen3_tts_0.6b", "zh_male",   ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    ("zh", "qwen3_tts_0.6b", "zh_female", ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    ("zh", "qwen3_tts_1.7b", "zh_male",   ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    ("zh", "qwen3_tts_1.7b", "zh_female", ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    # ZH — kokoro zh_male 신규 측정
    ("zh", "kokoro",          "zh_male",  ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    ("zh", "kokoro",          "iu_long",  ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    # JA — qwen3 ja_male 신규 측정 (ja_female는 이미 캐시됨)
    ("ja", "qwen3_tts_0.6b",  "ja_male",  ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "qwen3_tts_1.7b",  "ja_male",  ["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    ("ja", "qwen3_tts_1.7b",  "ja_female",["ja_short", "ja_conversational", "ja_medium", "ja_formal", "ja_technical"]),
    # EN — fish_speech en_female/en_male 신규 측정
    ("en", "fish_speech",     "en_female",["en_short", "en_medium", "en_long", "en_numbers", "en_technical",
                                            "en_conversational", "en_formal", "en_punctuation", "en_names", "en_emotional"]),
    ("en", "fish_speech",     "en_male",  ["en_short", "en_medium", "en_long", "en_numbers", "en_technical",
                                            "en_conversational", "en_formal", "en_punctuation", "en_names", "en_emotional"]),
    # ZH — cosyvoice2/3 zh_male 신규 측정 (zh_female는 번체자 불일치로 제외)
    ("zh", "cosyvoice2",      "zh_male",  ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
    ("zh", "cosyvoice3",      "zh_male",  ["zh_short", "zh_conversational", "zh_medium", "zh_formal", "zh_technical"]),
]

# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 기존 결과 로드
    existing = {}
    if os.path.exists(CER_PATH):
        with open(CER_PATH, encoding="utf-8") as f:
            existing = json.load(f)
    scores = dict(existing)

    whisper = get_whisper(device)

    for lang, model, ref_key, text_keys in TARGETS:
        model_ref_key = f"{model}__{ref_key}" if ref_key != "default" else model
        model_scores = {}

        print(f"\n[{lang.upper()}] {model} / {ref_key}")

        for tk in text_keys:
            item_key = f"{lang}|{model_ref_key}|{tk}"

            # 이미 측정된 경우 스킵
            if item_key in scores:
                val = scores[item_key]
                cached_score = val.get("score") if isinstance(val, dict) else val
                model_scores[tk] = cached_score
                print(f"  {tk}: {cached_score:.3f} [cached]")
                continue

            wav_path = find_wav(lang, model, ref_key, tk)
            if not wav_path:
                print(f"  {tk}: MISSING WAV ({ref_key})")
                continue

            ref_text = TEST_TEXTS.get(tk, "")
            if not ref_text:
                print(f"  {tk}: no reference text")
                continue

            t0 = time.perf_counter()
            hyp = transcribe(wav_path, lang, whisper)
            score = compute_cer(ref_text, hyp, lang)
            elapsed = time.perf_counter() - t0

            model_scores[tk] = score
            scores[item_key] = {"score": round(score, 6), "hyp": hyp[:120]}

            # 중간 저장
            with open(CER_PATH, "w", encoding="utf-8") as f:
                json.dump(scores, f, ensure_ascii=False, indent=2)

            try:
                print(f"  {tk}: {score:.3f} [{elapsed:.1f}s] {hyp[:50]}")
            except UnicodeEncodeError:
                print(f"  {tk}: {score:.3f} [{elapsed:.1f}s]")

        # AVG 계산
        if model_scores:
            avg = sum(model_scores.values()) / len(model_scores)
            avg_key = f"{lang}|{model_ref_key}|AVG"
            scores[avg_key] = round(avg, 6)
            with open(CER_PATH, "w", encoding="utf-8") as f:
                json.dump(scores, f, ensure_ascii=False, indent=2)
            print(f"  → AVG: {avg:.4f}")

    print(f"\n완료: {CER_PATH}")


if __name__ == "__main__":
    main()
