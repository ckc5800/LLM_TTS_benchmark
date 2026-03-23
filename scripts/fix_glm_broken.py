"""Mark glm_tts entries where success=True but wav file missing as success=False."""
import json, os, shutil

jsonl_path = 'd:/tts-benchmark/results/results_multilingual/detailed_metrics.jsonl'
bench_root = 'd:/tts-benchmark/results'

lines_out = []
fixed = 0

with open(jsonl_path, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d.get('model_key') == 'glm_tts' and d.get('success'):
            wp = d.get('wav_path', '')
            if wp:
                abs_wav = os.path.normpath(os.path.join(bench_root, wp))
                if not os.path.exists(abs_wav):
                    d['success'] = False
                    fixed += 1
        lines_out.append(json.dumps(d, ensure_ascii=False))

shutil.copy(jsonl_path, jsonl_path + '.bak6')
with open(jsonl_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines_out) + '\n')

print(f'Marked {fixed} glm_tts entries as success=False (file missing). Backup: .bak6')
