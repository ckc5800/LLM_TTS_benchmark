"""
특정 모델의 JA/ZH (또는 지정 언어) CER을 평가해서 cer_multilingual.json에 추가.
Usage:
  cd d:/tts-benchmark
  PYTHONUTF8=1 infra/quality/.venv/Scripts/python scripts/add_cer_for_model.py \
    --model index_tts2 --langs ja,zh --eval-model large-v3
"""
import argparse, json, os, sys
from statistics import mean

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BENCH_DIR, "infra"))

WAV_BASE = os.path.join(BENCH_DIR, "results", "results_multilingual", "wav")
JSONL_PATH = os.path.join(BENCH_DIR, "results", "results_multilingual", "detailed_metrics.jsonl")
CER_JSON_PATH = os.path.join(BENCH_DIR, "infra", "quality", "cer_multilingual.json")

from benchmark.core import TEST_TEXTS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model_key (예: index_tts2)")
    parser.add_argument("--langs", default="ja,zh", help="평가할 언어들 (comma separated)")
    parser.add_argument("--eval-model", default="large-v3")
    args = parser.parse_args()

    langs = [l.strip() for l in args.langs.split(",")]

    print(f"Whisper {args.eval_model} 로드 중...")
    from benchmark.evaluator import TTSQualityEvaluator
    evaluator = TTSQualityEvaluator(model_size=args.eval_model)
    print("로드 완료")

    # JSONL에서 해당 모델 run_index=0 entries 수집
    entries_by_lang = {lang: [] for lang in langs}
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            if d.get("model_key") != args.model:
                continue
            # accept any run_index (last-run dedup may have kept non-0)
            pass
            if not d.get("success", False):
                continue
            if d.get("lang") not in langs:
                continue
            entries_by_lang[d["lang"]].append(d)

    # cer_multilingual.json 로드
    if os.path.exists(CER_JSON_PATH):
        with open(CER_JSON_PATH, encoding="utf-8") as f:
            cer_data = json.load(f)
    else:
        cer_data = {}

    for lang in langs:
        entries = entries_by_lang[lang]
        if not entries:
            print(f"[{lang}] 항목 없음 - 스킵")
            continue

        print(f"\n=== {lang.upper()} ({len(entries)} entries) ===")
        lang_scores = []

        for e in entries:
            text_key = e["text_key"]
            text = TEST_TEXTS.get(text_key, e.get("text", ""))
            wav_path = e.get("wav_path", "")

            # 절대 경로 resolve
            if not os.path.isabs(wav_path):
                candidates = [
                    os.path.join(BENCH_DIR, wav_path),
                    os.path.join(BENCH_DIR, "results", wav_path),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        wav_path = c
                        break

            if not os.path.exists(wav_path):
                # WAV_BASE에서 직접 찾기
                ref_key = e.get("ref_key", "")
                safe_model = args.model.replace("/", "_")
                wav_name = f"{safe_model}__{ref_key}__{text_key}_0.wav"
                wav_path = os.path.join(WAV_BASE, lang, "A", wav_name)

            if not os.path.exists(wav_path):
                print(f"  [스킵] WAV 없음: {wav_path}")
                continue

            ref_key = e.get("ref_key", "")
            model_subkey = f"{args.model}__{ref_key}" if ref_key else args.model
            cer_key = f"{lang}|{model_subkey}|{text_key}"

            if cer_key in cer_data:
                score = cer_data[cer_key]["score"]
                print(f"  [캐시] {text_key} ({ref_key}): CER={score:.4f}")
            else:
                try:
                    stt = evaluator.transcribe(wav_path, language=lang)
                    metrics = evaluator.calculate_metrics(text, stt)
                    score = metrics["cer"]
                    cer_data[cer_key] = {"score": score, "hyp": stt[:150]}
                    print(f"  {text_key} ({ref_key}): CER={score:.4f}  STT: {stt[:60]}")
                except Exception as ex:
                    print(f"  [오류] {text_key}: {ex}")
                    continue

            lang_scores.append(score)

        if lang_scores:
            avg = mean(lang_scores)
            # AVG key (ref_key 없이 모델키만, 다중 ref_key면 그냥 평균)
            avg_key = f"{lang}|{args.model}|AVG"
            cer_data[avg_key] = avg
            print(f"  → {lang} AVG CER = {avg:.6f}")

    # 저장
    with open(CER_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(cer_data, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {CER_JSON_PATH}")

if __name__ == "__main__":
    main()
