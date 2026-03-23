"""
멀티링구얼 벤치마크 CER/WER 평가 스크립트
results_multilingual/wav/{lang}/A/{model}__{lang}_{text_key}_0.wav 패턴 처리

사용법:
  python quality/run_multilingual_cer.py --results-dir results_multilingual
  python quality/run_multilingual_cer.py --results-dir results_multilingual --langs ko,en
  python quality/run_multilingual_cer.py --results-dir results_multilingual --models fish_speech,gpt_sovits
"""
import argparse
import glob
import json
import os
import re
import sys
import time

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH_DIR)
from benchmark.core import TEST_TEXTS

# 언어별 평가 설정
LANG_CONFIG = {
    "ko": {"whisper_lang": "ko", "metric": "CER"},
    "en": {"whisper_lang": "en", "metric": "WER"},
    "ja": {"whisper_lang": "ja", "metric": "CER"},
    "zh": {"whisper_lang": "zh", "metric": "CER"},
}


# ─── 텍스트 정규화 ─────────────────────────────────────────────────────────────

def normalize_ko(text: str) -> str:
    text = re.sub(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ]", "", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def normalize_en(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def normalize_cjk(text: str) -> str:
    """일본어/중국어: 구두점 제거, 공백 정리"""
    text = re.sub(r"[\s\u3000]+", "", text)  # 공백 제거 (글자 단위 비교)
    text = re.sub(r"[。、！？「」『』【】〔〕…・,!?\"'()（）\[\]{}]", "", text)
    return text.strip()


# ─── 거리 계산 ─────────────────────────────────────────────────────────────────

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
    elif lang in ("ko",):
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

def get_whisper(model_size: str = "large-v3", device: str = "cuda"):
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print(f"  Whisper {model_size} 로딩 중... ({device})", flush=True)
        _whisper_model = WhisperModel(
            model_size, device=device,
            compute_type="float16" if device == "cuda" else "int8"
        )
        print("  Whisper 로드 완료\n", flush=True)
    return _whisper_model


def transcribe(wav_path: str, whisper_lang: str, model) -> str:
    try:
        segments, _ = model.transcribe(wav_path, language=whisper_lang, beam_size=5, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception as e:
        print(f"  [ERR] {os.path.basename(wav_path)}: {e}", flush=True)
        return ""


# ─── WAV 탐색 ──────────────────────────────────────────────────────────────────

def find_wav_files(results_dir: str, langs: list, method: str = "A") -> dict:
    """Returns { lang: { text_key: { model_key: wav_path } } }"""
    out = {}
    for lang in langs:
        wav_dir = os.path.join(results_dir, "wav", lang, method)
        if not os.path.isdir(wav_dir):
            continue
        for wav_path in sorted(glob.glob(os.path.join(wav_dir, "*_0.wav"))):
            fname = os.path.basename(wav_path)  # e.g. fish_speech__en_short_0.wav
            # 파싱: {model_key}__{lang}_{text_key}_0.wav
            m = re.match(r"^(.+?)__(" + lang + r"_\w+)_0\.wav$", fname)
            if not m:
                continue
            mk, tk = m.group(1), m.group(2)
            if tk not in TEST_TEXTS:
                continue
            out.setdefault(lang, {}).setdefault(tk, {})[mk] = wav_path
    return out


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="멀티링구얼 CER/WER 평가")
    parser.add_argument("--results-dir", default="results_multilingual")
    parser.add_argument("--langs", default="ko,en,ja,zh", help="평가 언어 (쉼표 구분)")
    parser.add_argument("--models", default=None, help="특정 모델만 평가 (쉼표 구분, 없으면 전체)")
    parser.add_argument("--output", default=None, help="결과 JSON 파일 (기본: quality/cer_multilingual.json)")
    parser.add_argument("--model-size", default="large-v3")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(BENCH_DIR, results_dir)

    target_langs = [l.strip() for l in args.langs.split(",") if l.strip() in LANG_CONFIG]
    target_models = set(m.strip() for m in args.models.split(",")) if args.models else None

    output_path = args.output or os.path.join(BENCH_DIR, "quality", "cer_multilingual.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 기존 결과 로드 (resume)
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  기존 결과 {len(existing)}개 로드 (resume)\n", flush=True)

    wav_map = find_wav_files(results_dir, target_langs)
    if not wav_map:
        print(f"WAV 파일 없음: {results_dir}")
        return

    total_files = sum(len(mk_map) for tk_map in wav_map.values() for mk_map in tk_map.values())
    print(f"=== 멀티링구얼 CER/WER 평가 (Whisper {args.model_size}, {device}) ===")
    print(f"  언어: {target_langs}  |  총 WAV: {total_files}개\n", flush=True)

    whisper = get_whisper(args.model_size, device)
    scores = dict(existing)

    for lang in target_langs:
        if lang not in wav_map:
            print(f"\n[{lang.upper()}] WAV 파일 없음, 스킵\n")
            continue
        cfg = LANG_CONFIG[lang]
        metric = cfg["metric"]
        wlang  = cfg["whisper_lang"]
        tk_map = wav_map[lang]
        text_keys = sorted(tk_map.keys())
        # 해당 언어의 모델 목록
        all_models = set()
        for mk_map in tk_map.values():
            all_models.update(mk_map.keys())
        if target_models:
            all_models = all_models & target_models

        print(f"\n[{lang.upper()}] {metric}  |  {len(all_models)}개 모델  |  {len(text_keys)}개 텍스트")
        print("-" * 60)

        for mk in sorted(all_models):
            key = f"{lang}|{mk}"
            model_scores = {}
            for tk in text_keys:
                wav_path = tk_map.get(tk, {}).get(mk)
                if not wav_path or not os.path.exists(wav_path):
                    continue
                ref_text = TEST_TEXTS[tk]
                # resume: 이미 평가된 경우 스킵
                item_key = f"{lang}|{mk}|{tk}"
                if item_key in existing:
                    model_scores[tk] = existing[item_key]["score"]
                    continue
                t0 = time.perf_counter()
                hyp = transcribe(wav_path, wlang, whisper)
                score = compute_cer(ref_text, hyp, lang)
                elapsed = time.perf_counter() - t0
                model_scores[tk] = score
                scores[item_key] = {"score": round(score, 6), "hyp": hyp[:120]}
                # 중간 저장
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=2)
                try:
                    print(f"  {mk:22s} [{tk:22s}]  {metric}={score:.3f}  [{elapsed:.1f}s]  {hyp[:50]}", flush=True)
                except UnicodeEncodeError:
                    print(f"  {mk:22s} [{tk:22s}]  {metric}={score:.3f}  [{elapsed:.1f}s]", flush=True)

            if model_scores:
                avg = sum(model_scores.values()) / len(model_scores)
                scores[f"{lang}|{mk}|AVG"] = round(avg, 6)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=2)

        # 언어별 요약
        print(f"\n  [{lang.upper()}] 평균 {metric} 요약:")
        lang_avgs = [(mk, scores[f"{lang}|{mk}|AVG"]) for mk in sorted(all_models) if f"{lang}|{mk}|AVG" in scores]
        for mk, avg in sorted(lang_avgs, key=lambda x: x[1]):
            bar = "█" * max(0, int((1 - avg) * 20))
            print(f"    {mk:25s}  {avg:.4f}  {bar}")

    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
