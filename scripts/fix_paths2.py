"""Fix wav_path issues in detailed_metrics.jsonl:
1. index_tts2: drop old entries where wav_path has no ref_key (superseded by newer entries)
2. Method B: paths with extra 'results/' prefix -> strip it
"""
import json, os, shutil

jsonl_path = 'd:/tts-benchmark/results/results_multilingual/detailed_metrics.jsonl'

lines_out = []
dropped_old = 0
fixed_method_b = 0

with open(jsonl_path, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        wp = d.get('wav_path', '')
        model = d.get('model_key', '')
        method = d.get('method', '')

        # Drop index_tts2 old entries: relative paths without ref_key in filename
        # Pattern: results_multilingual\wav\en\A\index_tts2__en_short_0.wav (no double __)
        if model == 'index_tts2' and not os.path.isabs(wp) and wp:
            # Old paths: "results_multilingual\wav\en\A\index_tts2__en_short_0.wav"
            # New paths are absolute. Drop old relative ones.
            dropped_old += 1
            continue

        # Fix Method B extra 'results/' prefix
        # e.g. 'results/results_multilingual\wav\en\B\...' -> 'results_multilingual/wav/en/B/...'
        if method == 'B' and wp and not os.path.isabs(wp):
            norm = wp.replace('\\', '/')
            if norm.startswith('results/results_multilingual/'):
                d['wav_path'] = norm[len('results/'):]
                fixed_method_b += 1

        lines_out.append(json.dumps(d, ensure_ascii=False))

# Backup and write
shutil.copy(jsonl_path, jsonl_path + '.bak4')
with open(jsonl_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines_out) + '\n')

print(f'index_tts2 old entries dropped: {dropped_old}')
print(f'Method B paths fixed: {fixed_method_b}')
print(f'Backup: {jsonl_path}.bak4')
