"""
기존 JSONL에서 cer/wer가 None인 항목을 찾아 평가 후 업데이트.
Usage:
  PYTHONUTF8=1 python scripts/fill_missing_cer.py \
    --results-dir results/results_multilingual \
    --model-filter index_tts2 \
    --eval-model large-v3
"""
import argparse, json, os, sys

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BENCH_DIR, "infra"))

def resolve_wav(wav_path: str) -> str:
    if os.path.isabs(wav_path):
        return wav_path
    # 상대 경로: results_multilingual/wav/... 형태
    # BENCH_DIR/results/results_multilingual/wav/... 또는 BENCH_DIR/results_multilingual/... 시도
    candidates = [
        os.path.join(BENCH_DIR, wav_path),
        os.path.join(BENCH_DIR, "results", wav_path),
        os.path.normpath(wav_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]  # fallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/results_multilingual")
    parser.add_argument("--model-filter", default=None, help="모델 키 필터 (예: index_tts2)")
    parser.add_argument("--eval-model", default="large-v3")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(BENCH_DIR, "results", results_dir)
        if not os.path.exists(results_dir):
            results_dir = os.path.join(BENCH_DIR, results_dir)

    jsonl_path = os.path.join(results_dir, "detailed_metrics.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"JSONL 없음: {jsonl_path}")
        sys.exit(1)

    # evaluator 로드
    print(f"Whisper {args.eval_model} 로드 중...")
    try:
        from benchmark.evaluator import TTSQualityEvaluator
        evaluator = TTSQualityEvaluator(model_size=args.eval_model)
        print("평가기 로드 완료")
    except ImportError:
        print("faster-whisper 없음. quality venv 사용 필요")
        sys.exit(1)

    # JSONL 읽기
    entries = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # 업데이트 대상: cer/wer가 None이고, run_index==0인 항목
    updated = 0
    for entry in entries:
        if args.model_filter and entry.get("model_key") != args.model_filter:
            continue
        if not entry.get("success", False):
            continue
        if entry.get("run_index", 0) != 0:
            continue
        if entry.get("cer") is not None:
            continue  # 이미 있음

        wav_path = entry.get("wav_path", "")
        abs_wav = resolve_wav(wav_path)

        if not os.path.exists(abs_wav):
            print(f"  [스킵] WAV 없음: {abs_wav}")
            continue

        lang = entry.get("lang", "en")
        text = entry.get("text", "")
        model_key = entry.get("model_key", "")
        text_key = entry.get("text_key", "")

        print(f"  평가 중: {model_key} / {lang} / {text_key}")
        try:
            stt_text = evaluator.transcribe(abs_wav, language=lang)
            metrics = evaluator.calculate_metrics(text, stt_text)
            entry["cer"] = metrics["cer"]
            entry["wer"] = metrics["wer"]
            entry["stt_text"] = stt_text
            print(f"    CER={entry['cer']:.4f}  WER={entry['wer']:.4f}  STT: {stt_text[:60]}")
            updated += 1
        except Exception as e:
            print(f"    [오류] {e}")

    if updated == 0:
        print("업데이트할 항목 없음.")
        return

    # JSONL 덮어쓰기
    backup = jsonl_path + ".bak"
    import shutil
    shutil.copy2(jsonl_path, backup)
    print(f"\nBackup: {backup}")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"완료! {updated}개 항목 업데이트됨.")

if __name__ == "__main__":
    main()
