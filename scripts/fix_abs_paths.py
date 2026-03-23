"""Normalize absolute wav_paths to relative paths in detailed_metrics.jsonl."""
import json, os, shutil

jsonl_path = 'd:/tts-benchmark/results/results_multilingual/detailed_metrics.jsonl'
bench_root = os.path.normpath('d:/tts-benchmark/results')

lines_out = []
fixed = 0

with open(jsonl_path, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        wp = d.get('wav_path', '')
        if wp and os.path.isabs(wp):
            abs_wp = os.path.normpath(wp)
            try:
                rel = os.path.relpath(abs_wp, bench_root).replace('\\', '/')
                d['wav_path'] = rel
                fixed += 1
            except ValueError:
                pass  # different drive, leave as is
        lines_out.append(json.dumps(d, ensure_ascii=False))

shutil.copy(jsonl_path, jsonl_path + '.bak5')
with open(jsonl_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines_out) + '\n')

print(f'Converted {fixed} absolute paths to relative. Backup: .bak5')
