"""
다국어 TTS 벤치마크 Runner
- Method A: 언어 매칭 참조 음성 (en 텍스트 → en_female ref)
- Method B: 교차 언어 (ko 참조 음성 → 다른 언어 텍스트, cross_lingual 모델만)
- Resume: detailed_metrics.jsonl 기반 스킵
- CER/WER: faster-whisper STT (옵션, --eval-model none으로 비활성화)

실행 예시:
  # 전체 언어, Method A만
  PYTHONUTF8=1 python run_multilingual_benchmark.py --langs all --method A --results-dir results_multilingual

  # 영어만, CER 포함
  PYTHONUTF8=1 python run_multilingual_benchmark.py --langs en --method A --eval-model base

  # 교차 언어 포함 (Method A+B)
  PYTHONUTF8=1 python run_multilingual_benchmark.py --langs en,ja,zh --method both
"""
import os
import sys

# Windows 콘솔 유니코드 출력 대응 (UnicodeEncodeError 방지)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import argparse
from datetime import datetime
from statistics import mean, stdev as _stdev

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(SCRIPTS_DIR)
INFRA_DIR = os.path.join(BENCH_DIR, "infra")
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, INFRA_DIR)

from benchmark.core import TEST_TEXTS, TEXT_SUITES
from run_benchmark import MODEL_REGISTRY, run_model_subprocess


# ─── 언어별 참조 음성 그룹 (Method A - 언어 매칭) ──────────────────────────
LANG_REF_GROUPS = {
    "ko": ["iu_long", "kor_female_calm", "kor_male_deep"],
    "en": ["en_female", "en_male", "iu_long"],  # 3번째는 교차 언어 성능 확인용
    "ja": ["ja_female", "ja_male", "iu_long"],
    "zh": ["zh_female", "zh_male", "iu_long"],
}

# Method B: cross-lingual 참조 음성 그룹 (KO 참조 → 다른 언어)
CROSS_LINGUAL_REF_GROUP = ["iu_long", "kor_female_calm", "kor_male_deep"]

# Cross-lingual 지원 모델 (KO 참조 → 다른 언어 합성 가능)
CROSS_LINGUAL_MODELS = {
    "cosyvoice2",
    "cosyvoice3",
    "qwen3_tts_0.6b",
    "qwen3_tts_1.7b",
    "gpt_sovits",
    "xtts",
    "fish_speech",
    "outetss",
    "voicecraftx",
    "maskgct",
    "hierspeech",
    "chattts",
    "zipvoice",
}

# Method C (Default Preset) 지원 모델: 클로닝이 가능하지만 우수한 기본 음성도 있는 경우
DEFAULT_PRESET_MODELS = {
    "chattts",
    "xtts",
    "openvoice",
    "outetss",
    "qwen3_tts_0.6b",
    "qwen3_tts_1.7b",
    "cosyvoice2",
    "cosyvoice3",
    "glm_tts",
}


# ─── Resume 지원 ──────────────────────────────────────────────────────────────
def load_done_set(results_dir: str) -> set:
    """JSONL에서 이미 완료된 (model_key, lang, ref_key, text_key, method) 조합 로드."""
    done = set()
    path = os.path.join(results_dir, "detailed_metrics.jsonl")
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = (
                    entry.get("model_key"),
                    entry.get("lang"),
                    entry.get("ref_key"),
                    entry.get("text_key"),
                    entry.get("method"),
                )
                if all(k is not None for k in key):
                    done.add(key)
            except Exception:
                pass
    return done


# ─── 평균 계산 ────────────────────────────────────────────────────────────────
def _build_avg_entry(model_key, lang, method, entries, ref_key=None):
    """entries 리스트로 집계 딕셔너리 생성."""
    info = MODEL_REGISTRY.get(model_key, {})
    rtfs      = [e["rtf"]              for e in entries if e.get("rtf", -1) >= 0]
    infers    = [e["inference_time_s"] for e in entries if e.get("inference_time_s", -1) >= 0]
    durations = [e["audio_duration_s"] for e in entries if e.get("audio_duration_s", -1) >= 0]
    vrams     = [e["vram_peak_mb"]     for e in entries if e.get("vram_peak_mb", -1) >= 0]
    ttfas     = [e["ttfa_ms"]          for e in entries if e.get("ttfa_ms", -1) >= 0]
    cers      = [e["cer"]              for e in entries if e.get("cer") is not None]
    # text_key별 최신(첫) WAV 경로 수집
    per_text_wav: dict = {}
    for e in entries:
        tk = e.get("text_key", "")
        if tk and tk not in per_text_wav and e.get("wav_path"):
            per_text_wav[tk] = e["wav_path"]
    # n_texts = 고유 text_key 수
    unique_texts = len({e.get("text_key") for e in entries})
    return {
        "model_key":            model_key,
        "model_name":           info.get("model_name", model_key),
        "model_size_params":    info.get("model_size_params", ""),
        "is_llm_based":         info.get("is_llm_based", False),
        "supported_languages":  info.get("supported_languages", []),
        "lang":                 lang,
        "method":               method,
        "ref_key":              ref_key,   # None이면 전체 평균
        "n_texts":              unique_texts,
        "n_samples":            len(entries),
        "avg_rtf":              round(mean(rtfs), 4)      if rtfs      else -1,
        "std_rtf":              round(_stdev(rtfs), 4)    if len(rtfs) > 1 else 0.0,
        "avg_inference_time_s": round(mean(infers), 3)    if infers    else -1,
        "avg_audio_duration_s": round(mean(durations), 3) if durations else -1,
        "avg_vram_peak_mb":     round(mean(vrams), 1)     if vrams     else -1,
        "avg_ttfa_ms":          round(mean(ttfas), 1)     if ttfas     else -1,
        "avg_cer":              round(mean(cers), 4)      if cers      else None,
        "per_text_wav":         per_text_wav,
    }


def compute_averages(results_dir: str, all_data: list):
    """JSONL 데이터로 모델별·언어별 평균 계산.
    - averages.json         : (model, lang, method) 전체 평균 (기존 호환)
    - speaker_averages.json : (model, lang, method, ref_key) 화자별 평균
    """
    # ── 그룹화 ──────────────────────────────────────────────────────────────
    groups: dict[tuple, list] = {}          # (model, lang, method)
    spk_groups: dict[tuple, list] = {}      # (model, lang, method, ref_key)
    for d in all_data:
        if not d.get("success"):
            continue
        mk, lang, method = d["model_key"], d["lang"], d["method"]
        rk = d.get("ref_key", "")
        groups.setdefault((mk, lang, method), []).append(d)
        spk_groups.setdefault((mk, lang, method, rk), []).append(d)

    # ── averages.json (전체 평균) ────────────────────────────────────────────
    averages = {}
    for (mk, lang, method), entries in sorted(groups.items()):
        averages[f"{mk}|{lang}|{method}"] = _build_avg_entry(mk, lang, method, entries)

    out_path = os.path.join(results_dir, "averages.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(averages, f, ensure_ascii=False, indent=2)

    # ── speaker_averages.json (화자별 평균) ──────────────────────────────────
    spk_averages = {}
    for (mk, lang, method, rk), entries in sorted(spk_groups.items()):
        spk_averages[f"{mk}|{lang}|{method}|{rk}"] = _build_avg_entry(mk, lang, method, entries, ref_key=rk)

    spk_path = os.path.join(results_dir, "speaker_averages.json")
    with open(spk_path, "w", encoding="utf-8") as f:
        json.dump(spk_averages, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  평균 결과 저장: {out_path}")
    print(f"  화자별 평균:   {spk_path}")
    print(f"\n  {'모델':<22} {'언어':<5} {'방식':<4} {'avg_RTF':>8}  {'±std':>6}  {'N':>3}  {'avg_VRAM':>9}  {'avg_CER':>8}")
    print(f"  {'-'*68}")
    for gk, v in sorted(averages.items(), key=lambda x: (x[1]['lang'], x[1]['avg_rtf'] if x[1]['avg_rtf'] > 0 else 999)):
        vram_s = f"{v['avg_vram_peak_mb']:.0f}MB" if v['avg_vram_peak_mb'] > 0 else "-"
        cer_s  = f"{v['avg_cer']:.3f}"             if v['avg_cer'] is not None else "-"
        rtf_s  = f"{v['avg_rtf']:.3f}"             if v['avg_rtf'] > 0 else "-"
        std_s  = f"{v['std_rtf']:.3f}"             if v['avg_rtf'] > 0 else "-"
        print(f"  {v['model_name']:<22} {v['lang']:<5} {v['method']:<4} {rtf_s:>8}  {std_s:>6}  {v['n_texts']:>3}  {vram_s:>9}  {cer_s:>8}")
    print(f"{'='*70}")


# ─── 메인 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="다국어 TTS 벤치마크")
    parser.add_argument(
        "--models", default="ready",
        help="모델 키 (쉼표 구분) 또는 'ready' / 'all'",
    )
    parser.add_argument(
        "--langs", default="all",
        help="언어 (쉼표 구분: ko,en,ja,zh) 또는 'all'",
    )
    parser.add_argument(
        "--method", choices=["A", "B", "C", "both", "all"], default="A",
        help="A=언어매칭 참조음성, B=교차언어(ko ref→다른언어), C=기본음성, both=A+B, all=A+B+C",
    )
    parser.add_argument(
        "--results-dir", default=os.path.join(BENCH_DIR, "results", "results_multilingual"),
    )
    parser.add_argument(
        "--eval-model", default="none",
        help="CER 평가용 Whisper 크기 (base/large-v3/none). 기본=none(스킵)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="이미 완료된 조합도 재실행",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="검증용 모드: 각 조합당 1회 합성, 1개 텍스트만 실행",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="지원 언어 매트릭스 출력 후 종료",
    )
    args = parser.parse_args()

    # ── 모델/언어 목록 선택 ───────────────────────────────────────────────────
    if args.models == "ready":
        model_keys = [k for k, v in MODEL_REGISTRY.items() if v.get("status") == "ready"]
    elif args.models == "all":
        model_keys = list(MODEL_REGISTRY.keys())
    else:
        model_keys = [k.strip() for k in args.models.split(",")]

    # --- LLM Filter (User Request) ---
    try:
        from generate_multilingual_report import MODEL_INFO
        valid_llm = {k for k, v in MODEL_INFO.items() if v.get("arch_type") == "LLM"}
        model_keys = [mk for mk in model_keys if mk in valid_llm]
        print(f"\n[Info] LLM 모델만 필터링됨: {model_keys}\n")
    except ImportError:
        pass

    if args.langs == "all":
        languages = ["ko", "en", "ja", "zh"]
    else:
        languages = [l.strip() for l in args.langs.split(",")]

    # ── --list ───────────────────────────────────────────────────────────────
    if args.list:
        print(f"\n{'모델':<22} {'크기':<8} {'KO':<4} {'EN':<4} {'JA':<4} {'ZH':<4} {'XL':<4} {'상태'}")
        print("-" * 65)
        for k in model_keys:
            info = MODEL_REGISTRY.get(k, {})
            langs_s = info.get("supported_languages", [])
            ko = "✔" if "ko" in langs_s else " "
            en = "✔" if "en" in langs_s else " "
            ja = "✔" if "ja" in langs_s else " "
            zh = "✔" if "zh" in langs_s else " "
            xl = "✔" if k in CROSS_LINGUAL_MODELS else " "
            print(f"{k:<22} {info.get('model_size_params',''):<8} {ko:<4} {en:<4} {ja:<4} {zh:<4} {xl:<4} {info.get('status','')}")
        return

    do_A = args.method in ("A", "both", "all")
    do_B = args.method in ("B", "both", "all")
    do_C = args.method in ("C", "all")

    os.makedirs(args.results_dir, exist_ok=True)
    wav_base = os.path.join(args.results_dir, "wav")
    os.makedirs(wav_base, exist_ok=True)
    jsonl_path = os.path.join(args.results_dir, "detailed_metrics.jsonl")

    # ── Resume ───────────────────────────────────────────────────────────────
    done_set = load_done_set(args.results_dir)
    if done_set:
        print(f"[Resume] 이미 완료된 조합 {len(done_set)}개 스킵합니다.")

    # 기존 JSONL 데이터 로드 (평균 계산용)
    all_data: list[dict] = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_data.append(json.loads(line))
                    except Exception:
                        pass

    # ── CER 평가기 (선택적) ───────────────────────────────────────────────────
    evaluator = None
    if args.eval_model.lower() != "none":
        try:
            from benchmark.evaluator import TTSQualityEvaluator
            evaluator = TTSQualityEvaluator(model_size=args.eval_model)
            print(f"CER 평가기 로드 완료 (whisper {args.eval_model})")
        except Exception as e:
            print(f"[경고] CER 평가기 로드 실패: {e}")
            print("  → faster-whisper 없음. CER 스킵 (quality venv로 별도 실행 가능)")

    # ── 전체 조합 수 미리 계산 ────────────────────────────────────────────────
    total_combos = 0
    for lang in languages:
        lang_texts = TEXT_SUITES.get(lang, [])
        ref_keys_A = LANG_REF_GROUPS.get(lang, ["en_female"])
        for mk in model_keys:
            info = MODEL_REGISTRY.get(mk, {})
            if info.get("status") != "ready":
                continue
            if lang not in info.get("supported_languages", []):
                continue
            if do_A:
                total_combos += len(lang_texts) * len(ref_keys_A)
            if do_B and mk in CROSS_LINGUAL_MODELS and lang != "ko":
                total_combos += len(lang_texts) * len(CROSS_LINGUAL_REF_GROUP)
            if do_C and mk in DEFAULT_PRESET_MODELS:
                total_combos += len(lang_texts)

    print(f"\n{'='*70}")
    print(f"다국어 TTS 벤치마크  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print(f"모델: {len(model_keys)}개  |  언어: {languages}  |  방식: {args.method}")
    print(f"전체 조합: {total_combos}개  |  이미 완료: {len(done_set)}개")
    print(f"결과 디렉토리: {args.results_dir}")
    print(f"{'='*70}\n")

    completed = skipped = failed = 0

    def run_one(model_key: str, lang: str, ref_key: str, text_key: str, method: str):
        nonlocal completed, skipped, failed
        combo = (model_key, lang, ref_key, text_key, method)
        info = MODEL_REGISTRY[model_key]
        text = TEST_TEXTS[text_key]

        # WAV 경로
        safe_model = model_key.replace("/", "_")
        wav_dir_sub = os.path.join(wav_base, lang, method)
        os.makedirs(wav_dir_sub, exist_ok=True)
        wav_path = os.path.join(wav_dir_sub, f"{safe_model}__{ref_key}__{text_key}.wav")

        # Resume 판단: JSONL 트래킹 우선, WAV 존재도 스킵 조건
        if (combo in done_set or os.path.exists(wav_path)) and not args.force:
            skipped += 1
            print(f"  [스킵] {model_key} / {lang} / {text_key} / {method}")
            return

        print(f"\n  [{lang}/{method}] {model_key}  ref={ref_key}  text={text_key}")
        print(f"  텍스트: {text[:60]}{'...' if len(text) > 60 else ''}")

        try:
            # 드라이 런인 경우 runs=1 강제
            actual_runs = 1 if args.dry_run else 3
            raw_results = run_model_subprocess(
                info, text, wav_dir_sub, runs=actual_runs,
                ref_key=ref_key,
                timeout_s=info.get("timeout_s", 1800),
                output_path=wav_path,
            )
        except Exception as e:
            print(f"    [오류] {e}")
            failed += 1
            return

        if not raw_results:
            print(f"    [실패] 결과 없음")
            failed += 1
            return

        # 모든 run 결과를 순회하며 로그 기록
        first_success_wav = None
        for res_idx, res in enumerate(raw_results):
            if not res.get("success"):
                continue
            
            actual_wav = res.get("output_wav", wav_path)
            if first_success_wav is None:
                first_success_wav = actual_wav
            
            # RTF 계산
            infer_s = res.get("inference_time_s", -1)
            dur_s   = res.get("audio_duration_s", 1)
            rtf     = round(infer_s / dur_s, 4) if infer_s > 0 and dur_s > 0 else -1

            # CER/WER 평가 (첫 번째 성공한 run에서만 수행하여 시간 절약)
            cer = wer = stt_text = None
            if evaluator and res_idx == 0 and os.path.exists(actual_wav):
                try:
                    stt_text = evaluator.transcribe(actual_wav, language=lang)
                    metrics  = evaluator.calculate_metrics(text, stt_text)
                    cer      = metrics["cer"]
                    wer      = metrics["wer"]
                except Exception as e:
                    print(f"    [CER 오류] {e}")

            entry = {
                "model_key":        model_key,
                "model_name":       info.get("model_name", model_key),
                "model_size_params": info.get("model_size_params", ""),
                "is_llm_based":     info.get("is_llm_based", False),
                "supported_languages": info.get("supported_languages", []),
                "lang":             lang,
                "method":           method,
                "ref_key":          ref_key,
                "text_key":         text_key,
                "run_index":        res.get("run_index", res_idx),
                "text":             text[:120],
                "wav_path":         actual_wav,
                "success":          True,
                "load_time_s":      res.get("load_time_s", -1),
                "ttfa_ms":          res.get("ttfa_ms", -1),
                "inference_time_s": infer_s,
                "audio_duration_s": dur_s,
                "rtf":              rtf,
                "sample_rate":      res.get("sample_rate", -1),
                "vram_before_mb":   res.get("vram_before_mb", -1),
                "vram_after_mb":    res.get("vram_after_mb", -1),
                "vram_peak_mb":     res.get("vram_peak_mb", -1),
                "cer":              cer,
                "wer":              wer,
                "stt_text":         stt_text,
                "timestamp":        datetime.now().isoformat(),
            }

            # JSONL 저장
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            all_data.append(entry)
            completed += 1

        done_set.add(combo)
        
        # 요약 출력 (첫 번째 결과 기준)
        if raw_results[0].get("success"):
            r0 = raw_results[0]
            rtf0 = r0["inference_time_s"] / r0["audio_duration_s"] if r0["audio_duration_s"] > 0 else -1
            vram_s = f"{r0['vram_peak_mb']:.0f}MB" if r0['vram_peak_mb'] > 0 else "-"
            print(f"    OK | runs={len(raw_results)} | avg_RTF={rtf0:.3f} | VRAM={vram_s} | audio={r0['audio_duration_s']:.1f}s")
        else:
            print(f"    [부분 실패] 일부 run이 실패했습니다.")

    # ── 메인 루프 ─────────────────────────────────────────────────────────────
    for lang in languages:
        lang_texts  = TEXT_SUITES.get(lang, [])
        if args.dry_run and lang_texts:
            lang_texts = lang_texts[:1] # 드라이 런: 첫 번째 텍스트만
            
        ref_keys_A  = LANG_REF_GROUPS.get(lang, ["en_female"])

        print(f"\n{'━'*70}")
        print(f"  언어: {lang.upper()}  |  텍스트 {len(lang_texts)}개  |  참조음성그룹(A): {ref_keys_A}")
        print(f"{'━'*70}")

        for model_key in model_keys:
            info = MODEL_REGISTRY.get(model_key, {})
            if info.get("status") != "ready":
                continue
            if lang not in info.get("supported_languages", []):
                continue

            print(f"\n>>> {info.get('model_name', model_key)}  "
                  f"({info.get('model_size_params', '')}, "
                  f"{'LLM' if info.get('is_llm_based') else 'NAR/Flow'})")

            for text_key in lang_texts:
                # Method A: 언어 매칭 참조음성 (보이스 그룹 순회)
                if do_A:
                    for rk in ref_keys_A:
                        run_one(model_key, lang, rk, text_key, "A")

                # Method B: KO 참조음성 → 다른 언어 (교차 언어 합성)
                if do_B and model_key in CROSS_LINGUAL_MODELS and lang != "ko":
                    for rk in CROSS_LINGUAL_REF_GROUP:
                        run_one(model_key, lang, rk, text_key, "B")
                
                # Method C: 기본 내장 음성 (Default Preset)
                if do_C and model_key in DEFAULT_PRESET_MODELS:
                    run_one(model_key, lang, "default_preset", text_key, "C")

    # ── 평균 계산 저장 ────────────────────────────────────────────────────────
    compute_averages(args.results_dir, all_data)

    print(f"\n{'='*70}")
    print(f"완료!  새로 실행: {completed}  |  스킵: {skipped}  |  실패: {failed}")
    print(f"상세 결과: {jsonl_path}")
    print(f"평균 결과: {os.path.join(args.results_dir, 'averages.json')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
