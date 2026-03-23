import json, os

jsonl_path = 'd:/tts-benchmark/results/results_multilingual/detailed_metrics.jsonl'
bench_root = 'd:/tts-benchmark'

broken = []
with open(jsonl_path, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        wp = d.get('wav_path', '')
        if not wp:
            continue
        if os.path.isabs(wp):
            abs_path = wp
        else:
            abs_path = os.path.join(bench_root, wp)
        abs_path = abs_path.replace('\\', '/')
        if not os.path.exists(abs_path):
            broken.append((d.get('model_key'), d.get('lang'), d.get('method'), abs_path[:90]))

seen = set()
for model, lang, method, path in broken:
    key = (model, lang, method)
    if key not in seen:
        seen.add(key)
        print(f'{model} {lang}/{method}: {path}')
print(f'\nTotal broken entries: {len(broken)}')
