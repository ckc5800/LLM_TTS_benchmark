import json, os, shutil

jsonl_path = 'd:/tts-benchmark/results/results_multilingual/detailed_metrics.jsonl'
old_prefix = 'D:\\tts-benchmark\\results_multilingual\\'
new_prefix = 'D:\\tts-benchmark\\results\\results_multilingual\\'

lines_out = []
count_fixed = 0

with open(jsonl_path, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        wp = d.get('wav_path', '')
        if wp.startswith(old_prefix):
            d['wav_path'] = new_prefix + wp[len(old_prefix):]
            count_fixed += 1
        lines_out.append(json.dumps(d, ensure_ascii=False))

# Backup original
shutil.copy(jsonl_path, jsonl_path + '.bak2')

with open(jsonl_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines_out) + '\n')

print(f'Fixed {count_fixed} entries. Backup: {jsonl_path}.bak2')
