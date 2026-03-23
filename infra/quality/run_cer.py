"""
CER/WER 음성 인식 정확도 평가 스크립트
faster-whisper (STT) 로 WAV 파일을 전사 후 원문과 비교합니다.

- 한국어: CER (Character Error Rate) — 어절 경계가 없으므로 글자 단위
- 영어: WER (Word Error Rate) — 단어 단위

사용법:
  python run_cer.py --results-dir results_ko_r2 --text ko_medium --output cer_ko_r2.json
  python run_cer.py --results-dir results_en_r2 --text en_medium --output cer_en_r2.json
  python run_cer.py --wav some_file.wav --ref "reference text here"
"""
import argparse
import json
import os
import sys
import glob
import re
import time

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH_DIR)

from benchmark.core import TEST_TEXTS


# ─── 텍스트 정규화 ────────────────────────────────────────────────────────────

def normalize_ko(text: str) -> str:
    """한국어 CER 용 정규화: 구두점 제거, 소문자화, 공백 통일"""
    # 구두점 / 특수문자 제거
    text = re.sub(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ]", "", text)
    # 영숫자 소문자화
    text = text.lower()
    # 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_en(text: str) -> str:
    """영어 WER 용 정규화: 구두점 제거, 소문자화, 공백 통일"""
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── CER / WER 계산 ───────────────────────────────────────────────────────────

def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate: edit_distance(ref_chars, hyp_chars) / len(ref_chars)"""
    ref = normalize_ko(reference)
    hyp = normalize_ko(hypothesis)
    # 공백 포함 글자 단위 비교 (공백은 제거하여 순수 글자만)
    ref_chars = ref.replace(" ", "")
    hyp_chars = hyp.replace(" ", "")
    if len(ref_chars) == 0:
        return 0.0
    d = _edit_distance(ref_chars, hyp_chars)
    return d / len(ref_chars)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: edit_distance(ref_words, hyp_words) / len(ref_words)"""
    ref_words = normalize_en(reference).split()
    hyp_words = normalize_en(hypothesis).split()
    if len(ref_words) == 0:
        return 0.0
    d = _edit_distance(ref_words, hyp_words)
    return d / len(ref_words)


def _edit_distance(seq1, seq2) -> int:
    """Levenshtein distance between two sequences"""
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


# ─── Whisper STT ──────────────────────────────────────────────────────────────

def load_whisper(model_size: str = "large-v3", device: str = "cuda"):
    from faster_whisper import WhisperModel
    print(f"  Whisper {model_size} 로딩 중... (device={device})", flush=True)
    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
    print("  Whisper 로드 완료", flush=True)
    return model


def transcribe(model, wav_path: str, language: str = None) -> str:
    """WAV 파일 전사 (language=None → 자동 감지)"""
    try:
        segments, info = model.transcribe(
            wav_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()
    except Exception as e:
        print(f"  [ERROR] transcribe {os.path.basename(wav_path)}: {e}", flush=True)
        return ""


# ─── WAV 파일 탐색 ────────────────────────────────────────────────────────────

def find_wav_files(results_dir: str) -> dict:
    """results_dir/wav/{model_key}/{ModelName}_0.wav 패턴으로 WAV 파일 탐색"""
    wav_dir = os.path.join(results_dir, "wav")
    if not os.path.isdir(wav_dir):
        print(f"WAV 디렉토리 없음: {wav_dir}")
        return {}

    found = {}
    for model_key in os.listdir(wav_dir):
        model_wav_dir = os.path.join(wav_dir, model_key)
        if not os.path.isdir(model_wav_dir):
            continue
        wavs = sorted(glob.glob(os.path.join(model_wav_dir, "*_0.wav")))
        if not wavs:
            wavs = sorted(glob.glob(os.path.join(model_wav_dir, "*.wav")))
        if wavs:
            found[model_key] = wavs[0]
    return found


# ─── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CER/WER 음성 인식 정확도 평가")
    parser.add_argument("--results-dir", default=None,
                        help="벤치마크 결과 디렉토리 (results_ko_r2 등)")
    parser.add_argument("--text", choices=list(TEST_TEXTS.keys()), default="ko_medium",
                        help="참조 텍스트 키 (TEST_TEXTS 에서)")
    parser.add_argument("--ref", default=None,
                        help="직접 입력할 참조 텍스트 (--text 대신 사용)")
    parser.add_argument("--wav", default=None,
                        help="단일 WAV 파일 평가")
    parser.add_argument("--output", default=None,
                        help="결과 JSON 파일 경로")
    parser.add_argument("--model-size", default="large-v3",
                        help="Whisper 모델 크기 (large-v3, medium, small, ...)")
    parser.add_argument("--language", default=None,
                        help="전사 언어 코드 (ko, en, ... | None=자동감지)")
    parser.add_argument("--cpu", action="store_true", help="CPU 모드")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    # 참조 텍스트 결정
    ref_text = args.ref if args.ref else TEST_TEXTS[args.text]
    # 언어 판별
    ko_ratio = sum(1 for c in ref_text if "\uac00" <= c <= "\ud7a3") / max(len(ref_text), 1)
    is_korean = ko_ratio > 0.2 or (args.language and args.language.startswith("ko"))
    lang_code = args.language or ("ko" if is_korean else "en")
    metric_name = "CER" if is_korean else "WER"
    compute_metric = compute_cer if is_korean else compute_wer

    print(f"\n=== CER/WER 정확도 평가 (Whisper {args.model_size}, device={device}) ===")
    print(f"  언어: {lang_code} | 지표: {metric_name}")
    print(f"  참조 텍스트: {ref_text[:80]}...\n", flush=True)

    # Whisper 모델 로드
    t0 = time.perf_counter()
    model = load_whisper(args.model_size, device)
    print(f"  로드 시간: {time.perf_counter()-t0:.1f}s\n", flush=True)

    # 단일 파일 모드
    if args.wav:
        hyp = transcribe(model, args.wav, lang_code)
        score = compute_metric(ref_text, hyp)
        print(f"  전사: {hyp}")
        print(f"  {metric_name}: {score:.4f} ({score*100:.1f}%)")
        return

    if not args.results_dir:
        parser.print_help()
        return

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(BENCH_DIR, results_dir)

    wav_files = find_wav_files(results_dir)
    if not wav_files:
        print(f"WAV 파일 없음: {results_dir}")
        return

    print(f"  평가 대상: {len(wav_files)}개 모델\n", flush=True)

    # JSON 경로 사전 결정 (크래시 대비 조기 설정)
    output_path = args.output
    if not output_path:
        lang_tag = "ko" if is_korean else "en"
        output_path = os.path.join(BENCH_DIR, "quality", f"cer_{lang_tag}_r2.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def _safe_print(s: str):
        try:
            print(s, flush=True)
        except UnicodeEncodeError:
            print(s.encode("utf-8", errors="replace").decode("ascii", errors="replace"), flush=True)

    scores = {}
    for model_key, wav_path in sorted(wav_files.items()):
        t1 = time.perf_counter()
        hyp = transcribe(model, wav_path, lang_code)
        score = compute_metric(ref_text, hyp)
        elapsed = time.perf_counter() - t1
        _safe_print(f"  {model_key:22s}  {metric_name}={score:.4f} ({score*100:.1f}%)  [{elapsed:.1f}s]")
        _safe_print(f"    -> {hyp[:80]}")
        scores[model_key] = round(score, 6)
        # 중간 저장 (크래시 대비)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)

    print(f"\n  JSON 저장: {output_path}", flush=True)
    print(f"\n{'='*55}", flush=True)
    print(f"  {metric_name} 결과 (낮을수록 좋음, 0=완벽):", flush=True)
    for k, v in sorted(scores.items(), key=lambda x: x[1]):
        bar = "█" * max(0, int((1 - v) * 20))
        print(f"  {k:22s}  {v:.4f}  {bar}", flush=True)


if __name__ == "__main__":
    main()
