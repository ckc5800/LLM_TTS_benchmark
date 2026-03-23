"""Find entries where success=True but wav file doesn't exist on disk."""
import json, os

jsonl_path = 'd:/tts-benchmark/results/results_multilingual/detailed_metrics.jsonl'
bench_root = 'd:/tts-benchmark/results'

missing = {}  # (model, lang, method) -> count

with open(jsonl_path, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if not d.get('success'):
            continue
        if d.get('run_index', 0) != 0:
            continue
        wp = d.get('wav_path', '')
        if not wp:
            continue
        if os.path.isabs(wp):
            abs_wav = wp
        else:
            abs_wav = os.path.normpath(os.path.join(bench_root, wp))
        abs_wav = abs_wav.replace('\\', '/')
        if not os.path.exists(abs_wav):
            key = (d.get('model_key'), d.get('lang'), d.get('method', 'A'))
            missing[key] = missing.get(key, 0) + 1

if missing:
    print("재생 불가 (success=True but file missing):")
    for (model, lang, method), cnt in sorted(missing.items()):
        print(f"  {model:25s} {lang}/{method}: {cnt}개 텍스트")
else:
    print("모든 파일 존재 확인!")
