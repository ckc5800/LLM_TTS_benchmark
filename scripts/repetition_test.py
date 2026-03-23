import requests
import time
import statistics
import argparse
import os
import json
import csv
from datetime import datetime

def parse_markdown_table(file_path):
    """
    Parses a markdown table and extracts rows as lists of values.
    Assumes first row is header, second is separator.
    """
    sentences = []
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    # Skip separators (e.g., |---|---|)
                    if parts and not all(c == '-' or c == ':' for c in parts[0]):
                        # Heuristic: the longest part is likely the sentence
                        sentence = max(parts, key=len)
                        if len(sentence) > 5:
                            sentences.append(sentence)
        
        # Remove header if it looks like one (usually the first one)
        if sentences and ("text" in sentences[0].lower() or "문장" in sentences[0]):
            sentences = sentences[1:]
            
    except Exception as e:
        print(f"Error parsing markdown: {e}")
        
    return sentences

def run_repetition_test(input_source, voice_id, count, api_base, results_dir):
    url = f"{api_base}/api/v2/tts-engine/synthesize/audio"
    
    # 1. Load Sentences
    sentences = []
    if os.path.isfile(input_source):
        if input_source.endswith('.md'):
            sentences = parse_markdown_table(input_source)
        else:
            with open(input_source, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
    else:
        sentences = [input_source]

    if not sentences:
        print("No sentences found to test.")
        return

    os.makedirs(results_dir, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"TTS Repetition Test  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print(f"Voice:  {voice_id}")
    print(f"Count:  {count} iterations per sentence")
    print(f"Total Sentences: {len(sentences)}")
    print(f"{'='*60}\n")
    
    all_results = []
    global_latencies = []
    
    for s_idx, sentence in enumerate(sentences, 1):
        print(f"\n[{s_idx}/{len(sentences)}] Testing: \"{sentence[:50]}{'...' if len(sentence)>50 else ''}\"")
        sentence_latencies = []
        
        for i in range(1, count + 1):
            payload = {
                "text": sentence,
                "voice_id": voice_id,
                "format": "mp3"
            }
            
            start_time = time.time()
            try:
                response = requests.post(url, json=payload, timeout=60)
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    sentence_latencies.append(latency)
                    global_latencies.append(latency)
                    print(f"  ({i:02d}/{count}) SUCCESS | {latency:8.2f} ms")
                    
                    all_results.append({
                        "sentence_index": s_idx,
                        "iteration": i,
                        "text": sentence,
                        "voice_id": voice_id,
                        "latency_ms": latency,
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    print(f"  ({i:02d}/{count}) FAILED  | Status: {response.status_code} | {response.text[:100]}")
                    all_results.append({
                        "sentence_index": s_idx,
                        "iteration": i,
                        "text": sentence,
                        "voice_id": voice_id,
                        "status": "error",
                        "error_code": response.status_code,
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                print(f"  ({i:02d}/{count}) ERROR   | {str(e)}")
                all_results.append({
                    "sentence_index": s_idx,
                    "iteration": i,
                    "text": sentence,
                    "voice_id": voice_id,
                    "status": "exception",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        if sentence_latencies:
            avg = statistics.mean(sentence_latencies)
            print(f"  >> Average: {avg:8.2f} ms")

    # 2. Save Results
    if all_results:
        json_path = os.path.join(results_dir, f"repetition_{session_id}.json")
        csv_path = os.path.join(results_dir, f"repetition_{session_id}.csv")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
            
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            if all_results:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
        
        print("\n" + "="*60)
        print("GLOBAL SUMMARY")
        print("-" * 60)
        print(f"Total Requests: {len(all_results)}")
        successes = [r for r in all_results if r["status"] == "success"]
        print(f"Success Rate:   {len(successes)}/{len(all_results)} ({(len(successes)/len(all_results))*100:.1f}%)")
        
        if global_latencies:
            print(f"Avg Latency:    {statistics.mean(global_latencies):.2f} ms")
            print(f"Min Latency:    {min(global_latencies):.2f} ms")
            print(f"Max Latency:    {max(global_latencies):.2f} ms")
            if len(global_latencies) > 1:
                print(f"Std Dev:        {statistics.stdev(global_latencies):.2f} ms")
        
        print(f"\nFiles saved to: {results_dir}")
        print(f"  - {os.path.basename(json_path)}")
        print(f"  - {os.path.basename(csv_path)}")
        print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Inference Repetition Test")
    parser.add_argument("--input", type=str, required=True, help="Text to synthesize or path to .md/.txt file")
    parser.add_argument("--voice", type=str, default="qwen_kor_female_01", help="Voice ID")
    parser.add_argument("--count", type=int, default=5, help="Repetitions per sentence")
    parser.add_argument("--api", type=str, default="http://localhost:9102", help="API Base URL")
    parser.add_argument("--results-dir", type=str, default=r"D:\tts-benchmark\results", help="Directory to save results")
    
    args = parser.parse_args()
    
    run_repetition_test(args.input, args.voice, args.count, args.api, args.results_dir)
