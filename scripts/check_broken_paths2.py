import json, os

jsonl_path = 'd:/tts-benchmark/results/results_multilingual/detailed_metrics.jsonl'
results_dir = 'd:/tts-benchmark/results/results_multilingual'
output_path = 'd:/tts-benchmark/html/multilingual_report.html'

abs_results = os.path.abspath(results_dir)
bench_root = os.path.dirname(abs_results)  # = d:/tts-benchmark/results
out_dir = os.path.dirname(os.path.abspath(output_path))

print(f'bench_root: {bench_root}')
print(f'out_dir: {out_dir}')

broken = []
with open(jsonl_path, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        wp = d.get('wav_path', '')
        if not wp:
            continue
        if os.path.isabs(wp):
            abs_wav = wp
        else:
            abs_wav = os.path.normpath(os.path.join(bench_root, wp))
        abs_wav = abs_wav.replace('\\', '/')
        if not os.path.exists(abs_wav):
            broken.append((d.get('model_key'), d.get('lang'), d.get('method'), abs_wav[:90]))

seen = set()
for model, lang, method, path in broken:
    key = (model, lang, method)
    if key not in seen:
        seen.add(key)
        print(f'{model} {lang}/{method}: {path}')
print(f'\nTotal broken entries: {len(broken)}')
