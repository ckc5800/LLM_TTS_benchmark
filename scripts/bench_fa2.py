"""Qwen3-TTS FlashAttention2 vs SDPA 벤치마크 (Docker 내부 실행용)"""
import time, json, os
import numpy as np
import soundfile as sf
import torch

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/Qwen3-TTS-0.6B")
REF_AUDIO = os.environ.get("REF_AUDIO", "/refs/kor_female_calm.wav")
REF_TEXT  = os.environ.get("REF_TEXT", "")
OUT_DIR   = os.environ.get("OUT_DIR", "/out")
RUNS      = int(os.environ.get("RUNS", "3"))
TEXT      = os.environ.get("TEXT", "안녕하세요, 오늘 날씨가 정말 좋네요.")

os.makedirs(OUT_DIR, exist_ok=True)

def get_vram():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0

def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_peak():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0

def detect_lang(text):
    ko = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
    zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    total = ko + zh + en
    if total == 0: return "Korean"
    if ko > total * 0.2: return "Korean"
    if zh > total * 0.2: return "Chinese"
    return "English"

def run_benchmark(attn_impl):
    from qwen_tts import Qwen3TTSModel

    print(f"\n{'='*50}")
    print(f"Testing: {attn_impl}")
    print(f"{'='*50}")

    ref_audio_np, ref_sr = sf.read(REF_AUDIO)
    ref_audio = (ref_audio_np, ref_sr)
    ref_text = REF_TEXT or open(REF_AUDIO.replace(".wav", ".txt")).read().strip()
    text_lang = detect_lang(TEXT)

    vram_before = get_vram()
    t_load = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram()
    print(f"Load time: {load_time:.2f}s, VRAM: {vram_after_load:.0f}MB")

    rtfs = []
    for run_idx in range(-1, RUNS):
        is_warmup = run_idx == -1
        reset_peak()
        t0 = time.perf_counter()
        wavs, sr = model.generate_voice_clone(
            text=TEXT, language=text_lang,
            ref_audio=ref_audio, ref_text=ref_text,
        )
        elapsed = time.perf_counter() - t0
        peak_vram = get_peak()

        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if hasattr(audio, "cpu"): audio = audio.cpu().numpy()
        audio = np.array(audio, dtype=np.float32).squeeze()
        audio_dur = len(audio) / sr

        if not is_warmup:
            rtf = elapsed / audio_dur
            rtfs.append(rtf)
            print(f"  Run {run_idx}: RTF={rtf:.3f}, dur={audio_dur:.2f}s, VRAM_peak={peak_vram:.0f}MB")
            out_path = os.path.join(OUT_DIR, f"{attn_impl}_run{run_idx}.wav")
            sf.write(out_path, audio, sr, subtype="PCM_16")

    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()

    avg_rtf = sum(rtfs) / len(rtfs)
    print(f"  → AVG RTF: {avg_rtf:.3f}")
    return {
        "attn_impl": attn_impl,
        "avg_rtf": avg_rtf,
        "rtfs": rtfs,
        "load_time": load_time,
        "vram_after_load_mb": vram_after_load,
        "vram_peak_mb": peak_vram,
    }


results = {}
for impl in ["sdpa", "flash_attention_2"]:
    try:
        results[impl] = run_benchmark(impl)
    except Exception as e:
        print(f"ERROR [{impl}]: {e}")
        results[impl] = {"error": str(e)}

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for impl, r in results.items():
    if "error" in r:
        print(f"{impl}: ERROR - {r['error']}")
    else:
        print(f"{impl}: RTF={r['avg_rtf']:.3f}, VRAM={r['vram_peak_mb']:.0f}MB")

if "sdpa" in results and "flash_attention_2" in results:
    s, f = results["sdpa"], results["flash_attention_2"]
    if "avg_rtf" in s and "avg_rtf" in f:
        diff = (f["avg_rtf"] - s["avg_rtf"]) / s["avg_rtf"] * 100
        print(f"\nFlashAttention2 vs SDPA: RTF {diff:+.1f}%")

out_json = os.path.join(OUT_DIR, "fa2_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n결과 저장: {out_json}")
