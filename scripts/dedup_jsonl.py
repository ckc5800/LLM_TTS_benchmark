"""Remove duplicate wav_path entries from detailed_metrics.jsonl (keep first occurrence)."""
import json, shutil

jsonl_path = 'd:/tts-benchmark/results/results_multilingual/detailed_metrics.jsonl'

with open(jsonl_path, encoding='utf-8') as f:
    lines = f.readlines()

seen_wav = set()
lines_out = []
removed = 0

for line in lines:
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    wp = d.get('wav_path', '')
    if wp and d.get('success'):
        if wp in seen_wav:
            removed += 1
            continue
        seen_wav.add(wp)
    lines_out.append(json.dumps(d, ensure_ascii=False))

shutil.copy(jsonl_path, jsonl_path + '.bak7')
with open(jsonl_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines_out) + '\n')

print(f'제거: {removed}개 / 남음: {len(lines_out)}개 / 백업: .bak7')
