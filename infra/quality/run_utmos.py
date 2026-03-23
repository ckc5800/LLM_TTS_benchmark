"""
UTMOS 음성 품질 평가 스크립트
wvmos 패키지를 사용해 WAV 파일들의 MOS 스코어를 측정합니다.

사용법:
  python run_utmos.py --results-dir results_en_r2 --output mos_en_r2.json
  python run_utmos.py --results-dir results_ko_r2 --output mos_ko_r2.json
  python run_utmos.py --wav some_file.wav  (단일 파일 테스트)

MOS 스코어: 1.0 (최저) ~ 5.0 (최고)  일반적으로:
  < 2.5: 매우 나쁨 (robotic / artifacts)
  2.5-3.5: 보통 (acceptable but unnatural)
  3.5-4.0: 좋음 (natural sounding)
  > 4.0: 매우 좋음 (near human quality)
"""
import argparse
import json
import os
import sys
import glob
import time

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_wvmos_model(cuda=True):
    """UTMOS (wvmos) 모델 로드"""
    from wvmos import get_wvmos
    print("  UTMOS 모델 로딩 중... (첫 실행 시 ~400MB 다운로드)", flush=True)
    model = get_wvmos(cuda=cuda)
    print("  UTMOS 로드 완료", flush=True)
    return model


def score_wav(model, wav_path: str) -> float:
    """단일 WAV 파일 MOS 스코어 계산"""
    try:
        score = model.calculate_one(wav_path)
        return float(score)
    except Exception as e:
        print(f"  [ERROR] {os.path.basename(wav_path)}: {e}", flush=True)
        return -1.0


def find_wav_files(results_dir: str) -> dict:
    """results_dir/wav/{model_key}/{ModelName}_0.wav 패턴으로 WAV 파일 탐색"""
    wav_dir = os.path.join(results_dir, "wav")
    if not os.path.isdir(wav_dir):
        print(f"WAV 디렉토리 없음: {wav_dir}")
        return {}

    found = {}
    for model_key in os.listdir(wav_dir):
        model_wav_dir = os.path.join(wav_dir, model_key)
        if not os.path.isdir(model_wav_dir):
            continue
        # _0.wav 파일 우선
        wavs = sorted(glob.glob(os.path.join(model_wav_dir, "*_0.wav")))
        if not wavs:
            wavs = sorted(glob.glob(os.path.join(model_wav_dir, "*.wav")))
        if wavs:
            found[model_key] = wavs[0]  # 첫 번째 run 파일 사용

    return found


def main():
    parser = argparse.ArgumentParser(description="UTMOS MOS 품질 평가")
    parser.add_argument("--results-dir", default=None,
                        help="벤치마크 결과 디렉토리 (results_ko_r2 등)")
    parser.add_argument("--wav", default=None,
                        help="단일 WAV 파일 평가")
    parser.add_argument("--output", default=None,
                        help="결과 JSON 파일 경로")
    parser.add_argument("--cpu", action="store_true",
                        help="CPU 모드 (느림)")
    args = parser.parse_args()

    import torch
    cuda = torch.cuda.is_available() and not args.cpu
    print(f"\n=== UTMOS 품질 평가 (device={'cuda' if cuda else 'cpu'}) ===\n", flush=True)

    # 모델 로드
    t0 = time.perf_counter()
    model = load_wvmos_model(cuda=cuda)
    print(f"  로드 시간: {time.perf_counter()-t0:.1f}s\n", flush=True)

    if args.wav:
        # 단일 파일 평가
        score = score_wav(model, args.wav)
        print(f"  MOS score: {score:.3f}  ({os.path.basename(args.wav)})")
        return

    if not args.results_dir:
        parser.print_help()
        return

    # results_dir의 모든 WAV 평가
    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(BENCH_DIR, results_dir)

    wav_files = find_wav_files(results_dir)
    if not wav_files:
        print(f"WAV 파일 없음: {results_dir}")
        return

    print(f"  평가 대상: {len(wav_files)}개 모델\n", flush=True)

    scores = {}
    for model_key, wav_path in sorted(wav_files.items()):
        t1 = time.perf_counter()
        score = score_wav(model, wav_path)
        elapsed = time.perf_counter() - t1
        stars = "★" * int(score) + "☆" * (5 - int(score)) if score > 0 else "N/A"
        print(f"  {model_key:22s}  MOS={score:5.3f}  {stars}  ({elapsed:.1f}s)", flush=True)
        scores[model_key] = score

    # JSON 저장
    output_path = args.output
    if not output_path:
        lang = "ko" if "ko" in results_dir else "en"
        output_path = os.path.join(BENCH_DIR, "quality", f"mos_{lang}_r2.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print(f"\n  JSON 저장: {output_path}", flush=True)
    print(f"\n{'='*50}", flush=True)
    print("  MOS 결과 (높을수록 좋음, 최대 5.0):", flush=True)
    for k, v in sorted(scores.items(), key=lambda x: -x[1] if x[1] > 0 else 999):
        bar = "#" * int(v * 4) if v > 0 else ""
        print(f"  {k:22s}  {v:.3f}  {bar}", flush=True)
    print(json.dumps({"status": "ok", "scores": scores}, ensure_ascii=False))


if __name__ == "__main__":
    main()
