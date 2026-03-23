import os
import json
import argparse
import torch
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(BENCH_DIR, "infra", "references")

def compute_sim(audio1_path, audio2_path, model, feature_extractor, device, ref_emb=None):
    try:
        # Load Reference Embedding if not provided
        if ref_emb is None:
            wav2, sr2 = sf.read(audio2_path)
            wav2 = torch.from_numpy(wav2).float().to(device)
            if len(wav2.shape) > 1: wav2 = wav2.mean(dim=-1)
            if sr2 != 16000:
                import torchaudio
                wav2 = torchaudio.functional.resample(wav2, sr2, 16000)
            wav2 = wav2.squeeze().cpu().numpy()
            inputs2 = feature_extractor(wav2, return_tensors="pt", sampling_rate=16000).to(device)
            with torch.no_grad():
                ref_emb = model(**inputs2).embeddings
            ref_emb = torch.nn.functional.normalize(ref_emb, dim=-1)
            # Cleanup
            del wav2, inputs2

        # Load and Resample Target Audio
        wav1, sr1 = sf.read(audio1_path)
        if sr1 != 16000:
            import torchaudio
            tmp_wav = torch.from_numpy(wav1).float().to(device)
            if len(tmp_wav.shape) > 1: tmp_wav = tmp_wav.mean(dim=-1)
            tmp_wav = torchaudio.functional.resample(tmp_wav, sr1, 16000)
            wav1 = tmp_wav.squeeze().cpu().numpy()
            del tmp_wav
        else:
            if len(wav1.shape) > 1: wav1 = wav1.mean(axis=-1)
            wav1 = wav1.squeeze()

        # Sliding Window / Chunking for Fairness and Stability
        chunk_size_s = 15  # 15 seconds per chunk
        chunk_size_samples = chunk_size_s * 16000
        
        if len(wav1) > chunk_size_samples:
            # Overlap chunks by 5s to be more robust, or just divide clearly
            scores = []
            for start in range(0, len(wav1), chunk_size_samples):
                end = min(start + chunk_size_samples, len(wav1))
                chunk = wav1[start:end]
                if len(chunk) < 16000: continue # Skip fragments < 1s
                
                inputs1 = feature_extractor(chunk, return_tensors="pt", sampling_rate=16000).to(device)
                with torch.no_grad():
                    emb1 = model(**inputs1).embeddings
                emb1 = torch.nn.functional.normalize(emb1, dim=-1)
                
                sim = torch.cosine_similarity(emb1, ref_emb).item()
                scores.append(sim)
                del emb1, inputs1
                
            final_score = sum(scores) / len(scores) if scores else 0.0
            return final_score, ref_emb
        else:
            # Single chunk processing
            inputs1 = feature_extractor(wav1, return_tensors="pt", sampling_rate=16000).to(device)
            with torch.no_grad():
                emb1 = model(**inputs1).embeddings
            emb1 = torch.nn.functional.normalize(emb1, dim=-1)
            
            final_score = torch.cosine_similarity(emb1, ref_emb).item()
            # Cleanup
            del emb1, inputs1
            return final_score, ref_emb

    except Exception as e:
        print(f"Error computing SIM for {audio1_path} and {audio2_path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Specific model key to calculate for")
    parser.add_argument("--langs", type=str, help="Specific languages to calculate for, comma separated")
    parser.add_argument("--results-dir", type=str, default=os.path.join(BENCH_DIR, "results", "results_multilingual"))
    parser.add_argument("--output", type=str, default=os.path.join(BENCH_DIR, "infra", "quality", "sim_multilingual.json"))
    args = parser.parse_args()

    results_wav_dir = os.path.join(args.results_dir, "wav")
    sim_json_path = args.output
    quality_dir = os.path.dirname(sim_json_path)
    os.makedirs(quality_dir, exist_ok=True)
    sim_data = {}
    if os.path.exists(sim_json_path):
        try:
            with open(sim_json_path, "r", encoding="utf-8") as f:
                sim_data = json.load(f)
        except:
            print("Warning: Failed to load existing SIM data. Starting fresh.")
            sim_data = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading WavLM SV model on {device}...")
    model_name = "microsoft/wavlm-base-plus-sv"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = WavLMForXVector.from_pretrained(model_name).to(device)
    model.eval()

    # 캐시 참조 임베딩
    ref_emb_cache = {}

    target_langs = args.langs.split(",") if args.langs else ["ko", "en", "ja", "zh"]
    target_models = args.model.split(",") if args.model else []

    jsonl_path = os.path.join(args.results_dir, "detailed_metrics.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found.")
        return

    # Load all entries from JSONL
    entries = []
    all_scores = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    count = 0
    save_interval = 50
    vram_interval = 20 # Every 20 samples clear cache
    
    for entry in tqdm(entries, desc="Computing SIM"):
        if not entry.get("success"): continue
        if not entry.get("is_llm_based"): continue # Skip non-LLM to save time
        
        lang = entry.get("lang")
        model_key = entry.get("model_key")
        ref_key = entry.get("ref_key")
        wav_path_rel = entry.get("wav_path") # e.g. "results_multilingual\wav\ko\A\outetss__ko_short_0.wav"
        
        if not lang or not model_key or not ref_key or not wav_path_rel:
            continue
            
        if target_langs and lang not in target_langs:
            continue
        if target_models and model_key not in target_models:
            continue
            
        # Determine full path to WAV
        if wav_path_rel.startswith("results_multilingual"):
            full_wav_path = os.path.join(args.results_dir, "..", wav_path_rel)
        else:
            full_wav_path = os.path.join(BENCH_DIR, "results", wav_path_rel)
        
        full_wav_path = os.path.abspath(full_wav_path)
        
        if not os.path.exists(full_wav_path):
            continue

        ref_audio_path = os.path.join(REF_DIR, f"{ref_key}.wav")
        if not os.path.exists(ref_audio_path):
            continue

        sim_key = f"{lang}|{model_key}|{ref_key}|{os.path.basename(full_wav_path)}"
        
        score = None
        if sim_key in sim_data:
            score = sim_data[sim_key]
        else:
            # Use cache for reference embedding
            ref_emb = ref_emb_cache.get(ref_key)
            score, ref_emb = compute_sim(full_wav_path, ref_audio_path, model, feature_extractor, device, ref_emb=ref_emb)
            
            if ref_emb is not None:
                ref_emb_cache[ref_key] = ref_emb
                
            if score is not None:
                sim_data[sim_key] = score
                count += 1
                
                # Incremental save
                if count % save_interval == 0:
                    with open(sim_json_path, "w", encoding="utf-8") as f:
                        json.dump(sim_data, f, indent=4, ensure_ascii=False)
                
                # Periodical VRAM clear
                if count % vram_interval == 0:
                    torch.cuda.empty_cache()
        
        if score is not None:
            avg_k = f"{lang}|{model_key}"
            all_scores.setdefault(avg_k, []).append(score)

    # 평균 계산
    for avg_k, scores in all_scores.items():
        if scores:
            sim_data[f"{avg_k}|AVG"] = sum(scores) / len(scores)

    with open(sim_json_path, "w", encoding="utf-8") as f:
        json.dump(sim_data, f, indent=4, ensure_ascii=False)
        
    print(f"Saved SIM scores to {sim_json_path}")

if __name__ == "__main__":
    main()
