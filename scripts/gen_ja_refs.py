"""
깨끗한 일본어 참조 음성 생성 (MioTTS jp_female/jp_male preset)
출력: infra/references/ja_female.wav, ja_male.wav (+ .txt)
"""
import os, sys, re, time, torch, soundfile as sf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIO_REPO  = os.path.join(ROOT_DIR, "engines", "mio_tts")
MIO_CODEC = "Aratako/MioCodec-25Hz-44.1kHz-v2"
MODEL_DIR = os.path.join(ROOT_DIR, "models", "MioTTS-1.7B")
REF_DIR   = os.path.join(ROOT_DIR, "infra", "references")
sys.path.insert(0, MIO_REPO)

from transformers import AutoTokenizer, AutoModelForCausalLM
from miocodec import MioCodecModel

TARGETS = [
    {
        "preset":   os.path.join(MIO_REPO, "presets", "jp_female.pt"),
        "out_wav":  os.path.join(REF_DIR, "ja_female.wav"),
        "out_txt":  os.path.join(REF_DIR, "ja_female.txt"),
        "text":     "人工知能による音声合成技術は急速に発展しており、自然で流暢な音声を生成できるようになっています。",
    },
    {
        "preset":   os.path.join(MIO_REPO, "presets", "jp_male.pt"),
        "out_wav":  os.path.join(REF_DIR, "ja_male.wav"),
        "out_txt":  os.path.join(REF_DIR, "ja_male.txt"),
        "text":     "興味をそそられる村を三十分ほど散策するのは価値があります。この地域には歴史ある建物が多く残されています。",
    },
]

TOKEN_PATTERN = re.compile(r"<\|s_(\d+)\|>")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Loading MioTTS 1.7B model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto"
).eval()
codec = MioCodecModel.from_pretrained(MIO_CODEC).eval().to(device)
sample_rate = int(codec.config.sample_rate)
print(f"Model loaded. Sample rate: {sample_rate}")

for t in TARGETS:
    print(f"\n[{os.path.basename(t['out_wav'])}] text: {t['text']}")
    global_embedding = torch.load(t["preset"], map_location=device, weights_only=True)

    messages = [{"role": "user", "content": t["text"]}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=1.0,
            do_sample=True,
            repetition_penalty=1.1,
        )
    elapsed = time.perf_counter() - t0

    generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    token_ids = [int(m) for m in TOKEN_PATTERN.findall(generated_text)]
    if not token_ids:
        print(f"  ERROR: no audio tokens generated!")
        continue

    tokens_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        wav = codec.decode(
            global_embedding=global_embedding.to(device),
            content_token_indices=tokens_tensor,
        )
    wav_np = wav.squeeze().float().cpu().numpy()
    audio_len = len(wav_np) / sample_rate

    sf.write(t["out_wav"], wav_np, sample_rate)
    with open(t["out_txt"], "w", encoding="utf-8") as f:
        f.write(t["text"])

    print(f"  Saved: {t['out_wav']}")
    print(f"  Audio: {audio_len:.1f}s, inference: {elapsed:.1f}s")

print("\nDone.")
