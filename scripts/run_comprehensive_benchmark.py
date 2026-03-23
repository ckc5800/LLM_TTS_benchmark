
import os
import json
import time
import argparse
from datetime import datetime
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(SCRIPTS_DIR)
INFRA_DIR = os.path.join(BENCH_DIR, "infra")
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, INFRA_DIR)

from benchmark.core import TEST_TEXTS, TEXT_SUITES, BenchmarkLogger, BenchmarkResult
from benchmark.evaluator import TTSQualityEvaluator
from run_benchmark import MODEL_REGISTRY, run_model_subprocess


def load_done_set(results_dir: str) -> set:
    """이미 완료된 (model_key, ref_key, text_key) 조합을 JSONL에서 읽어 반환"""
    done = set()
    detail_path = os.path.join(results_dir, "detailed_metrics.jsonl")
    if not os.path.exists(detail_path):
        return done
    with open(detail_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = (entry.get("model_name"), entry.get("ref_key"), entry.get("text_key"))
                if all(key):
                    done.add(key)
            except Exception:
                pass
    return done


def run_comprehensive():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--results-dir", default=os.path.join(ROOT_DIR, "results_comprehensive"))
    parser.add_argument("--eval-model", default="base", help="Whisper model size for evaluation")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    wav_dir = os.path.join(args.results_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    # 이미 완료된 조합 로드 (resume)
    done_set = load_done_set(args.results_dir)
    if done_set:
        print(f"[Resume] 이미 완료된 조합 {len(done_set)}개 스킵합니다.")

    # 평가기 로드
    evaluator = TTSQualityEvaluator(model_size=args.eval_model)

    # 모델 선택
    if args.models == "ready":
        model_keys = [k for k, v in MODEL_REGISTRY.items() if v.get("status") == "ready"]
    else:
        model_keys = args.models.split(",")

    # 참조 음성 리스트 (총 7종 공인 세트)
    ref_keys = ["iu_long", "iu_short", "kbs_short", "kor_female_calm", "kor_male_deep", "kor_male_warm", "male_docu"]

    # 테스트할 언어 스위트 (한국어 참조 음성이므로 한국어만 테스트)
    languages = ["ko"]

    logger = BenchmarkLogger(args.results_dir)

    total = len(model_keys) * len(ref_keys) * sum(
        len(TEXT_SUITES.get(lang, [])[:2]) for lang in languages
    )
    remaining = total - len(done_set)
    print(f"\n{'='*60}")
    print(f"전수 벤치마크: {len(model_keys)} 모델 x {len(ref_keys)} 참조음성")
    print(f"전체 조합: {total}개 | 남은 조합: {remaining}개")
    print(f"{'='*60}\n")

    completed = 0
    skipped = 0

    for model_key in model_keys:
        if model_key not in MODEL_REGISTRY:
            print(f"[경고] 모델 '{model_key}'이 MODEL_REGISTRY에 없습니다. 스킵.")
            continue
        info = MODEL_REGISTRY[model_key]
        print(f"\n>>> 모델: {model_key} ({info.get('model_name', model_key)})")

        for ref_key in ref_keys:
            for lang in languages:
                test_keys = TEXT_SUITES.get(lang, [])[:2]  # ko_short, ko_medium 등 앞 2개

                for text_key in test_keys:
                    combo = (model_key, ref_key, text_key)

                    # Resume: 이미 완료된 조합이면 스킵
                    if combo in done_set:
                        skipped += 1
                        print(f"  [스킵] {model_key} / {ref_key} / {text_key}")
                        continue

                    text = TEST_TEXTS[text_key]
                    print(f"  [진행] Ref: {ref_key} | Text: {text_key}")

                    # WAV 파일명: {model_key}_{ref_key}_{text_key}.wav
                    safe_model = model_key.replace("/", "_")
                    wav_filename = f"{safe_model}__{ref_key}__{text_key}.wav"
                    wav_path = os.path.join(wav_dir, wav_filename)

                    try:
                        raw_results = run_model_subprocess(
                            info, text, args.results_dir, runs=1,
                            ref_key=ref_key, timeout_s=400,
                            output_path=wav_path,
                        )

                        if raw_results and raw_results[0].get("success"):
                            res_data = raw_results[0]
                            actual_wav = res_data.get("output_wav", wav_path)

                            # 품질 평가 (STT)
                            stt_text = evaluator.transcribe(actual_wav, language=lang)
                            metrics = evaluator.calculate_metrics(text, stt_text)

                            result = BenchmarkResult(
                                model_name=model_key,
                                model_version=info.get("version", ""),
                                model_size_params=info.get("model_size_params", "-"),
                                korean_support=info.get("korean_support", True),
                                test_text=text,
                                run_index=0,
                                load_time_s=res_data["load_time_s"],
                                ttfa_ms=res_data.get("ttfa_ms", -1),
                                inference_time_s=res_data["inference_time_s"],
                                audio_duration_s=res_data["audio_duration_s"],
                                sample_rate=res_data["sample_rate"],
                                output_wav=actual_wav,
                                vram_used_mb=res_data.get("vram_after_mb", 0) - res_data.get("vram_before_mb", 0),
                                vram_peak_mb=res_data.get("vram_peak_mb", -1),
                                success=True,
                                notes=f"ref={ref_key} text={text_key} cer={metrics['cer']}"
                            )

                            full_res = result.to_dict()
                            full_res.update(metrics)
                            full_res["ref_key"] = ref_key
                            full_res["text_key"] = text_key
                            full_res["lang"] = lang

                            logger.add(result)

                            detail_path = os.path.join(args.results_dir, "detailed_metrics.jsonl")
                            with open(detail_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(full_res, ensure_ascii=False) + "\n")

                            done_set.add(combo)
                            completed += 1
                            print(f"    OK | RTF={res_data['inference_time_s']/res_data['audio_duration_s']:.3f} | CER={metrics['cer']:.3f} | STT: {stt_text[:40]}")

                        else:
                            print(f"  [실패] {model_key}: success=False")

                    except Exception as e:
                        print(f"  [오류] {model_key}/{ref_key}/{text_key}: {e}")

    logger.finalize()
    print(f"\n완료! 새로 실행: {completed}개 | 스킵: {skipped}개")
    print(f"결과: {args.results_dir}/detailed_metrics.jsonl")


if __name__ == "__main__":
    run_comprehensive()
