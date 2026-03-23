"""
각 모델 venv에서 실행되는 인퍼런스 서브프로세스
사용법: python run_model.py --model <name> --text "..." --output-path <wav_path> --runs N

JSON 결과를 stdout으로 출력

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  표준 측정 규칙 v2 (2026-03-20, 전체 적용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. 워밍업: WARMUP_RUNS = 3회 (CUDA 커널 컴파일 + GPU 캐시 안정화)
  2. GPU 동기화: inf_time 측정 전후로 sync_gpu() 호출
     (GPU 비동기 연산 완료를 보장, time.perf_counter 오류 방지)
  3. 참조 음성 로딩: get_reference_data()는 반드시 루프 밖, 타이밍 전에 호출
     (파일 I/O 시간이 RTF에 포함되지 않도록)
  4. VRAM 측정: reset_vram_peak()는 루프 안, t0 직전에 호출
  5. 타이밍 범위: 오직 모델 추론(synthesize/generate)만 포함
     (모델 로딩, 파일 저장, 후처리는 포함하지 않음)
  6. 비교 그룹: Zero-shot ↔ Fine-tuned 직접 비교 금지 (보고서에서 분리 표시)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import json
import sys
import os
import time

# 벤치마크 루트 추가
BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # d:\tts-benchmark\infra
ROOT_DIR = os.path.dirname(BENCH_DIR)  # d:\tts-benchmark
sys.path.insert(0, BENCH_DIR)

GLM_TTS_REPO = os.path.join(ROOT_DIR, "engines", "glm_tts")

# ─── 표준 참조 데이터 레지스트리 ──────────────────────────────────────────────
# 오디오와 텍스트는 d:/tts-benchmark/references/ 폴더 내에 짝을 맞춰 저장함
REF_DIR = os.path.join(BENCH_DIR, "references")
REFERENCE_REGISTRY = {
    "iu_long": {
        "wav": os.path.join(REF_DIR, "iu_long.wav"),
        "txt_file": os.path.join(REF_DIR, "iu_long.txt")
    },
    "iu_short": {
        "wav": os.path.join(REF_DIR, "iu_long.wav"), # 오디오는 원본 공유
        "txt_file": os.path.join(REF_DIR, "iu_short.txt")
    },
    "kbs_short": {
        "wav": os.path.join(REF_DIR, "kbs_short.wav"),
        "txt_file": os.path.join(REF_DIR, "kbs_short.txt")
    },
    "male_docu": {
        "wav": os.path.join(REF_DIR, "male_docu.wav"),
        "txt_file": os.path.join(REF_DIR, "male_docu.txt")
    },
    "kor_female_calm": {
        "wav": os.path.join(REF_DIR, "kor_female_calm.wav"),
        "txt_file": os.path.join(REF_DIR, "kor_female_calm.txt")
    },
    "kor_male_deep": {
        "wav": os.path.join(REF_DIR, "kor_male_deep.wav"),
        "txt_file": os.path.join(REF_DIR, "kor_male_deep.txt")
    },
    "kor_male_warm": {
        "wav": os.path.join(REF_DIR, "kor_male_warm.wav"),
        "txt_file": os.path.join(REF_DIR, "kor_male_warm.txt")
    },
    # 영어 참조 음성 (FLEURS en_us, ~10s)
    "en_female": {
        "wav": os.path.join(REF_DIR, "en_female.wav"),
        "txt_file": os.path.join(REF_DIR, "en_female.txt")
    },
    "en_male": {
        "wav": os.path.join(REF_DIR, "en_male.wav"),
        "txt_file": os.path.join(REF_DIR, "en_male.txt")
    },
    # 일본어 참조 음성 (JSUT female / FLEURS ja_jp male, ~10s)
    "ja_female": {
        "wav": os.path.join(REF_DIR, "ja_female.wav"),
        "txt_file": os.path.join(REF_DIR, "ja_female.txt")
    },
    "ja_male": {
        "wav": os.path.join(REF_DIR, "ja_male.wav"),
        "txt_file": os.path.join(REF_DIR, "ja_male.txt")
    },
    # 중국어 참조 음성 (FLEURS cmn_hans_cn, ~10s)
    "zh_female": {
        "wav": os.path.join(REF_DIR, "zh_female.wav"),
        "txt_file": os.path.join(REF_DIR, "zh_female.txt")
    },
    "zh_male": {
        "wav": os.path.join(REF_DIR, "zh_male.wav"),
        "txt_file": os.path.join(REF_DIR, "zh_male.txt")
    },
    # 하위 호환성용
    "default": {
        "wav": os.path.join(REF_DIR, "iu_long.wav"),
        "txt_file": os.path.join(REF_DIR, "iu_long.txt")
    }
}

def get_reference_data(args, target_sr: int = None, max_sec: float = None) -> tuple[str, str]:
    """레퍼런스 데이터(WAV 경로, Text)를 반환. 필요시 임시 파일로 크롭/리샘플링 수행.
    args.ref_audio 또는 args.ref_text가 있으면 이를 우선 사용."""
    import soundfile as sf
    import numpy as np
    import math

    ref_key = getattr(args, "ref_key", "default")
    wav_path = getattr(args, "ref_audio", None)
    ref_text = getattr(args, "ref_text", None)

    # 1. 수동 지정이 없는 경우 레지스트리에서 로드
    if not wav_path or not ref_text:
        if ref_key not in REFERENCE_REGISTRY:
            ref_key = "default"
        
        ref_info = REFERENCE_REGISTRY[ref_key]
        if not wav_path: wav_path = ref_info["wav"]
        
        if not ref_text:
            txt_file = ref_info["txt_file"]
            if os.path.exists(txt_file):
                with open(txt_file, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()
            else:
                # 최후의 Fallback
                ref_text = "참조 데이터가 설정되지 않았습니다."

    # 2. 경로 확인 및 Fallback
    if not os.path.exists(wav_path):
        alt_path = wav_path.replace(r"D:\aicess-tts-integrated\shared", os.path.join(BENCH_DIR, "shared"))
        if os.path.exists(alt_path):
            wav_path = alt_path

    # 3. 크롭이나 리샘플링이 필요한 경우 임시 파일 생성
    if target_sr or max_sec:
        data, sr = sf.read(wav_path)
        if data.ndim > 1: data = data.mean(axis=1) # mono

        # 크롭
        if max_sec:
            data = data[:int(max_sec * sr)]
            # 특정 파일 명이나 키워드가 있을 때 자동으로 단축 대본 스위칭
            if "iu_long" in str(wav_path) and max_sec <= 6.0 and not getattr(args, "ref_text", None):
                short_txt_file = os.path.join(BENCH_DIR, "references", "iu_short.txt")
                if os.path.exists(short_txt_file):
                    with open(short_txt_file, "r", encoding="utf-8") as f:
                        ref_text = f.read().strip()

        # 리샘플링
        if target_sr and sr != target_sr:
            from scipy.signal import resample_poly
            gcd = math.gcd(int(sr), int(target_sr))
            data = resample_poly(data, target_sr // gcd, sr // gcd)
            sr = target_sr
        
        # 임시 파일 경로 (해시 대신 파일명 활용)
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        tmp_path = os.path.join(BENCH_DIR, f"_ref_tmp_{base_name}_{target_sr or sr}.wav")
        sf.write(tmp_path, data, sr, subtype='PCM_16')
        return tmp_path, ref_text

    return wav_path, ref_text


def get_vram_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return -1.0


def get_vram_peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return -1.0


def reset_vram_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def sync_gpu():
    """GPU 비동기 연산 완료 대기 (표준 측정 규칙 v2: inf_time 측정 전후 호출).
    CPU-only 환경에서는 no-op."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# 표준 측정 규칙 v2: 워밍업 횟수 (CUDA 커널 컴파일 + GPU 캐시 안정화)
WARMUP_RUNS = 3


def run_cosyvoice2(args):
    import torch, gc

    # onnxruntime CUDAExecutionProvider를 위해 PyTorch CUDA DLL 경로를 PATH에 추가
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')

    COSYVOICE_REPO = os.path.join(ROOT_DIR, 'engines', 'cosyvoice')
    sys.path.insert(0, COSYVOICE_REPO)
    sys.path.insert(0, os.path.join(COSYVOICE_REPO, 'third_party', 'Matcha-TTS'))

    # 표준 참조 데이터 적용 (CosyVoice는 24kHz 사용)
    REF_WAV, REF_TEXT = get_reference_data(args, target_sr=24000, max_sec=10.0)

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    from cosyvoice.cli.cosyvoice import AutoModel
    model = AutoModel(model_dir=args.model_dir, load_jit=False, load_trt=False)
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    results = []
    # 한국어 reference zero_shot (태그 없이 텍스트 직접)
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        chunks, ttfa_ms = [], -1.0
        reset_vram_peak()
        # Dual Voice Support: Cloning vs Preset
        if getattr(args, "ref_key", "") == "default_preset":
            # Using built-in speaker ('中文女' - Chinese Female or similar)
            # Find a ready speaker id from the model
            spk_id = '中文女' if 'zh' in args.text else 'English Female'
            # Note: inference_sft is for SFT/Preset voices
            gen_func = lambda: model.inference_sft(args.text, spk_id, stream=False)
        else:
            # Zero-shot cloning
            gen_func = lambda: model.inference_zero_shot(args.text, REF_TEXT, REF_WAV, stream=False)

        for i, result in enumerate(gen_func()):
            if i == 0:
                ttfa_ms = (time.perf_counter() - t0) * 1000
            chunks.append(result['tts_speech'])
        sync_gpu()
        inf_time = time.perf_counter() - t0
        

        if not is_warmup:
            import torch, numpy as np, soundfile as sf
            audio = torch.cat(chunks, dim=-1)
            data = audio.cpu().float().numpy()
            if data.ndim == 2 and data.shape[0] == 1:
                data = data[0]
            max_val = np.abs(data).max()
            if max_val > 1.0:
                data /= max_val
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, data, model.sample_rate, subtype='PCM_16')
            duration = data.shape[0] / model.sample_rate
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": model.sample_rate,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_cosyvoice3(args):
    import torch, gc

    # onnxruntime CUDAExecutionProvider를 위해 PyTorch CUDA DLL 경로를 PATH에 추가
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')

    COSYVOICE_REPO = os.path.join(ROOT_DIR, 'engines', 'cosyvoice')
    sys.path.insert(0, COSYVOICE_REPO)
    sys.path.insert(0, os.path.join(COSYVOICE_REPO, 'third_party', 'Matcha-TTS'))

    # 표준 참조 데이터 적용 (CosyVoice 3.0도 이제 아이유 등 표준 샘플 사용)
    REF_WAV, REF_TEXT = get_reference_data(args, target_sr=24000, max_sec=10.0)

    # CosyVoice3 requires <|endofprompt|> in prompt_text (ref text)
    # zero_shot mode: no language tags needed in either text
    if '<|endofprompt|>' not in REF_TEXT:
        REF_TEXT = f"You are a helpful assistant.<|endofprompt|>{REF_TEXT}"

    tts_text = args.text

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    from cosyvoice.cli.cosyvoice import AutoModel
    model = AutoModel(model_dir=args.model_dir, load_trt=False)
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        chunks, ttfa_ms = [], -1.0
        reset_vram_peak()
        # Dual Voice Support: Cloning vs Preset
        if getattr(args, "ref_key", "") == "default_preset":
            spk_id = '中文女' if 'zh' in args.text else 'English Female'
            gen_func = lambda: model.inference_sft(tts_text, spk_id, stream=False)
        else:
            gen_func = lambda: model.inference_zero_shot(tts_text, REF_TEXT, REF_WAV, stream=False)

        # 스트리밍 결과 반환 루프
        for i, result in enumerate(gen_func()):
            if i == 0:
                ttfa_ms = (time.perf_counter() - t0) * 1000
            chunks.append(result['tts_speech'])
        sync_gpu()
        inf_time = time.perf_counter() - t0

        if not is_warmup:
            import numpy as np, soundfile as sf
            audio = torch.cat(chunks, dim=-1)
            data = audio.cpu().float().numpy()
            if data.ndim == 2 and data.shape[0] == 1:
                data = data[0]
            max_val = np.abs(data).max()
            if max_val > 1.0:
                data /= max_val
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, data, model.sample_rate, subtype='PCM_16')
            duration = data.shape[0] / model.sample_rate
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": model.sample_rate,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_fish_speech(args):
    """Fish Speech (openaudio-s1-mini) 인퍼런스"""
    import torch, gc
    import numpy as np
    import soundfile as sf

    FISH_REPO = os.path.join(ROOT_DIR, 'engines', 'fish_speech')
    sys.path.insert(0, FISH_REPO)
    # output_path를 절대경로로 변환 (os.chdir 후 상대경로 깨짐 방지)
    args.output_path = os.path.abspath(args.output_path)
    os.chdir(FISH_REPO)  # fish_speech needs to be in working dir for hydra configs

    # torchaudio 2.10+ removed list_audio_backends & replaced load() with torchcodec
    # Monkey-patch both to use soundfile instead
    import torchaudio as _ta
    import soundfile as _sf_ta_patch
    import torch as _torch_ta_patch
    import io as _io_ta_patch

    if not hasattr(_ta, 'list_audio_backends'):
        _ta.list_audio_backends = lambda: ["soundfile"]

    if not hasattr(_ta, '_orig_load'):
        _ta._orig_load = _ta.load

        def _soundfile_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                            channels_first=True, format=None, buffer_size=4096, backend=None):
            if hasattr(uri, 'read'):
                data, sr = _sf_ta_patch.read(uri, dtype='float32', always_2d=True)
            elif isinstance(uri, (bytes, bytearray)):
                data, sr = _sf_ta_patch.read(_io_ta_patch.BytesIO(uri), dtype='float32', always_2d=True)
            else:
                data, sr = _sf_ta_patch.read(str(uri), dtype='float32', always_2d=True)
            # data: [time, channels], convert to [channels, time]
            tensor = _torch_ta_patch.from_numpy(data.T.copy() if data.ndim > 1 else data.reshape(1, -1))
            if not channels_first and tensor.ndim > 1:
                tensor = tensor.T
            if frame_offset > 0:
                tensor = tensor[..., frame_offset:]
            if num_frames > 0 and num_frames != -1:
                tensor = tensor[..., :num_frames]
            return tensor, sr

        _ta.load = _soundfile_load

    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.vqgan.inference import load_model as load_decoder_model
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

    precision = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    llama_ckpt = args.model_dir
    # Fish Speech 1.5 codec file name
    codec_ckpt = os.path.join(args.model_dir, 'firefly-gan-vq-fsq-8x1024-21hz-generator.pth')
    if not os.path.exists(codec_ckpt):
        # fallback for other versions
        codec_ckpt = os.path.join(args.model_dir, 'codec.pth')
    decoder_cfg = "firefly_gan_vq"

    vram_before = get_vram_mb()
    t_load = time.perf_counter()

    llama_queue = launch_thread_safe_queue(
        checkpoint_path=llama_ckpt,
        device=device,
        precision=precision,
        compile=False,
    )
    decoder_model = load_decoder_model(
        config_name=decoder_cfg,
        checkpoint_path=codec_ckpt,
        device=device,
    )
    engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        precision=precision,
        compile=False,
    )
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    # 샘플레이트 확인
    if hasattr(decoder_model, 'spec_transform'):
        sample_rate = decoder_model.spec_transform.sample_rate
    else:
        sample_rate = decoder_model.sample_rate

    # 텍스트 언어 감지: 한국어 텍스트일 때만 한국어 레퍼런스 사용
    _ko_chars = sum(1 for c in args.text if '\uac00' <= c <= '\ud7a3')
    _is_korean_text = _ko_chars > len(args.text) * 0.2

    # 표준 참조 데이터 적용
    ref_wav, ref_txt = get_reference_data(args.ref_key)
    with open(ref_wav, 'rb') as _f:
        _ref_audio_bytes = _f.read()
    _fish_refs = [ServeReferenceAudio(audio=_ref_audio_bytes, text=ref_txt)]

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        req = ServeTTSRequest(
            text=args.text,
            streaming=False,
            references=_fish_refs,
            normalize=True,
        )
        ttfa_ms = -1.0
        audio_segments = []
        reset_vram_peak()
        t0 = time.perf_counter()
        for result in engine.inference(req):
            if result.code in ("segment", "final") and result.audio is not None:
                if len(audio_segments) == 0:
                    ttfa_ms = (time.perf_counter() - t0) * 1000
                _, audio_data = result.audio
                audio_segments.append(audio_data.astype(np.float32))
        sync_gpu()
        inf_time = time.perf_counter() - t0

        if audio_segments and not is_warmup:
            audio = np.concatenate(audio_segments, axis=-1)
            if audio.ndim > 1:
                audio = audio.squeeze()
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio /= max_val
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio, sample_rate, subtype='PCM_16')
            duration = audio.shape[0] / sample_rate
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sample_rate,
                "output_wav": out_path,
                "success": True,
            })
        elif not is_warmup:
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "success": False,
                "error": "No audio segments generated",
            })

    del engine, decoder_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_qwen3_tts(args):
    """Qwen3-TTS (Base voice clone) 인퍼런스"""
    import torch, gc
    import numpy as np
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vram_before = get_vram_mb()
    t_load = time.perf_counter()

    model = Qwen3TTSModel.from_pretrained(
        args.model_dir,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    # 텍스트 언어 자동감지 (ko/zh/en)
    def _detect_lang_qwen(text):
        ko = sum(1 for c in text if '\uac00' <= c <= '\ud7a3' or '\u3130' <= c <= '\u318f')
        zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        en = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        total = ko + zh + en
        if total == 0: return "Korean"
        if ko > total * 0.2: return "Korean"
        if zh > total * 0.2: return "Chinese"
        return "English"
    text_lang = _detect_lang_qwen(args.text)

    # 표준 측정 규칙 v2: 참조 음성 루프 밖에서 1회만 로드
    _ref_key = getattr(args, "ref_key", "")
    _m_type = model.model.tts_model_type if hasattr(model, 'model') and hasattr(model.model, 'tts_model_type') else "base"
    if _ref_key != "default_preset" or _m_type not in ("custom_voice", "voice_design"):
        REF_WAV, REF_TXT = get_reference_data(args)
    else:
        REF_WAV = REF_TXT = None

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        # Dual Voice Support: Cloning vs Preset
        if _ref_key == "default_preset":
            if _m_type == "custom_voice":
                spks = model.model.get_supported_speakers()
                spk = spks[0] if spks else "default"
                wavs, sr = model.generate_custom_voice(text=args.text, speaker=spk, language=text_lang)
            elif _m_type == "voice_design":
                wavs, sr = model.generate_voice_design(text=args.text, instruct="", language=text_lang)
            else:
                wavs, sr = model.generate_voice_clone(
                    text=args.text,
                    language=text_lang,
                    ref_audio=REF_WAV,
                    ref_text=REF_TXT,
                )
        else:
            wavs, sr = model.generate_voice_clone(
                text=args.text,
                language=text_lang,
                ref_audio=REF_WAV,
                ref_text=REF_TXT,
            )
        sync_gpu()
        inf_time = time.perf_counter() - t0

        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio /= max_val
        audio_dur = len(audio) / sr

        if not is_warmup:
            out_path = args.output_path.replace(".wav", f"_{run_idx}.wav")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, audio, sr, subtype="PCM_16")
            results.append({
                "load_time_s": load_time,
                "ttfa_ms": inf_time * 1000,   # 비스트리밍 → TTFA = 전체 시간
                "inference_time_s": inf_time,
                "audio_duration_s": audio_dur,
                "rtf": inf_time / audio_dur if audio_dur > 0 else 0,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_qwen3_tts_int8(args):
    """Qwen3-TTS INT8 Quantization 버전 (bitsandbytes load_in_8bit)"""
    import torch, gc
    import numpy as np
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    from transformers import BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        args.model_dir,
        device_map=device,
        quantization_config=bnb_config,
        attn_implementation="sdpa",
    )
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    def _detect_lang(text):
        ko = sum(1 for c in text if '\uac00' <= c <= '\ud7a3' or '\u3130' <= c <= '\u318f')
        zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        en = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        total = ko + zh + en
        if total == 0: return "Korean"
        if ko > total * 0.2: return "Korean"
        if zh > total * 0.2: return "Chinese"
        return "English"
    text_lang = _detect_lang(args.text)

    # 표준 측정 규칙 v2: 참조 음성 루프 밖에서 1회만 로드
    REF_WAV, REF_TXT = get_reference_data(args)

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        wavs, sr = model.generate_voice_clone(
            text=args.text, language=text_lang,
            ref_audio=REF_WAV, ref_text=REF_TXT,
        )
        sync_gpu()
        inf_time = time.perf_counter() - t0

        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        audio = np.array(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio /= max_val
        audio_dur = len(audio) / sr

        if not is_warmup:
            out_path = args.output_path.replace(".wav", f"_{run_idx}.wav")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, audio, sr, subtype="PCM_16")
            results.append({
                "load_time_s": load_time,
                "ttfa_ms": inf_time * 1000,
                "inference_time_s": inf_time,
                "audio_duration_s": audio_dur,
                "rtf": inf_time / audio_dur if audio_dur > 0 else 0,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


SPARK_TTS_REPO = os.path.join(ROOT_DIR, "engines", "spark_tts")


def run_spark_tts(args):
    """Spark-TTS 인퍼런스 (controllable TTS - ref audio 없이)"""
    import torch, gc
    import numpy as np
    import soundfile as sf
    from pathlib import Path

    sys.path.insert(0, SPARK_TTS_REPO)
    from cli.SparkTTS import SparkTTS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vram_before = get_vram_mb()
    t_load = time.perf_counter()

    model = SparkTTS(model_dir=Path(args.model_dir), device=device)
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    sample_rate = model.sample_rate

    # 표준 측정 규칙 v2: 참조 음성 루프 밖에서 1회만 로드
    REF_WAV, _ = get_reference_data(args)

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        wav = model.inference(
            text=args.text,
            prompt_speech_path=Path(REF_WAV),
        )
        sync_gpu()
        inf_time = time.perf_counter() - t0

        audio = wav.cpu().numpy() if hasattr(wav, "cpu") else np.array(wav)
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio /= max_val
        audio_dur = len(audio) / sample_rate

        if not is_warmup:
            out_path = args.output_path.replace(".wav", f"_{run_idx}.wav")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, audio, sample_rate, subtype="PCM_16")
            results.append({
                "load_time_s": load_time,
                "ttfa_ms": inf_time * 1000,
                "inference_time_s": inf_time,
                "audio_duration_s": audio_dur,
                "rtf": inf_time / audio_dur if audio_dur > 0 else 0,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sample_rate,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_glm_tts(args):
    """GLM-TTS 인퍼런스 (중/영 only, 한국어는 품질 저하)"""
    import torch, gc
    import torchaudio
    import soundfile as sf
    from functools import partial

    # GLM-TTS는 repo 내 상대경로 사용 → CWD를 glm_tts로 변경
    os.chdir(GLM_TTS_REPO)
    sys.path.insert(0, GLM_TTS_REPO)

    # torchaudio.load는 torchcodec(FFmpeg DLL 필요)를 사용 → soundfile로 패치
    import torchaudio, soundfile as _sf
    def _sf_torchaudio_load(path, *a, **kw):
        data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr
    torchaudio.load = _sf_torchaudio_load

    from utils.yaml_util import load_flow_model, load_speech_tokenizer
    from utils.tts_model_util import Token2Wav
    from utils.audio import mel_spectrogram
    from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
    from transformers import AutoTokenizer, LlamaForCausalLM
    from llm.glmtts import GLMTTS
    from glmtts_inference import get_special_token_ids, generate_long

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = args.model_dir  # D:\models\GLM-TTS

    vram_before = get_vram_mb()
    t_load = time.perf_counter()

    # Speech tokenizer (Whisper VQ encoder)
    _model, _feature_extractor = load_speech_tokenizer(
        os.path.join(model_dir, "speech_tokenizer")
    )
    speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)

    # GLM tokenizer
    glm_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_dir, "vq32k-phoneme-tokenizer"), trust_remote_code=True
    )
    tokenize_fn = lambda text: glm_tokenizer.encode(text)

    # Mel spectrogram extractor (24kHz)
    feat_extractor = partial(
        mel_spectrogram, sampling_rate=24000, hop_size=480,
        n_fft=1920, num_mels=80, win_size=1920, fmin=0, fmax=8000, center=False
    )

    # TTSFrontEnd (campplus ONNX is in repo frontend/)
    campplus_path = os.path.join(GLM_TTS_REPO, "frontend", "campplus.onnx")
    frontend = TTSFrontEnd(
        tokenize_fn, speech_tokenizer, feat_extractor, campplus_path, device=device
    )
    text_frontend = TextFrontEnd(use_phoneme=False)

    # LLM (LlamaForCausalLM)
    llm = GLMTTS(
        llama_cfg_path=os.path.join(model_dir, "llm", "config.json"), mode="PRETRAIN"
    )
    llm.llama = LlamaForCausalLM.from_pretrained(
        os.path.join(model_dir, "llm"), torch_dtype=torch.float16
    ).to(device)
    llm.llama_embedding = llm.llama.model.embed_tokens
    special_token_ids = get_special_token_ids(tokenize_fn)
    llm.set_runtime_vars(special_token_ids=special_token_ids)

    # Flow model
    flow = load_flow_model(
        os.path.join(model_dir, "flow", "flow.pt"),
        os.path.join(model_dir, "flow", "config.yaml"),
        device,
    )
    token2wav = Token2Wav(flow, sample_rate=24000, device=device)

    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    # 표준 참조 데이터 적용
    ref_wav, ref_text = get_reference_data(args)

    # Dual Voice Support: Cloning vs Preset
    if getattr(args, "ref_key", "") == "default_preset":
        # GLM-TTS does not have explicit SFT speakers in common API, 
        # but we can use a pre-defined prompt audio as "Default".
        prompt_speech = ref_wav
        prompt_text = text_frontend.text_normalize(ref_text)
    else:
        # User defined reference
        prompt_speech = ref_wav
        prompt_text = text_frontend.text_normalize(ref_text)

    prompt_text_token = frontend._extract_text_token(prompt_text + " ")
    prompt_speech_token = frontend._extract_speech_token([prompt_speech])
    speech_feat = frontend._extract_speech_feat(prompt_speech, sample_rate=24000)
    embedding = frontend._extract_spk_embedding(prompt_speech)
    cache_speech_token = [prompt_speech_token.squeeze().tolist()]
    flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(device)

    sample_rate = 24000
    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0

        # 캐시 초기화
        cache = {
            "cache_text": [prompt_text],
            "cache_text_token": [prompt_text_token],
            "cache_speech_token": cache_speech_token,
            "use_cache": True,
        }
        synth_text = text_frontend.text_normalize(args.text)

        reset_vram_peak()
        t0 = time.perf_counter()
        tts_speech, _, _, _ = generate_long(
            frontend=frontend,
            text_frontend=text_frontend,
            llm=llm,
            flow=token2wav,  # Token2Wav wrapper (has token2wav_with_cache method)
            text_info=["utt0", synth_text],
            cache=cache,
            embedding=embedding,
            flow_prompt_token=flow_prompt_token,
            speech_feat=speech_feat,
            device=device,
        )
        sync_gpu()
        inf_time = time.perf_counter() - t0

        audio = tts_speech.squeeze().cpu().numpy()
        audio_dur = len(audio) / sample_rate

        if not is_warmup:
            out_path = args.output_path.replace(".wav", f"_{run_idx}.wav")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, audio, sample_rate, subtype="PCM_16")
            results.append({
                "load_time_s": load_time,
                "ttfa_ms": inf_time * 1000,
                "inference_time_s": inf_time,
                "audio_duration_s": audio_dur,
                "rtf": inf_time / audio_dur if audio_dur > 0 else 0,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sample_rate,
                "output_wav": out_path,
                "success": True,
            })

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


INDEX_TTS_REPO = os.path.join(ROOT_DIR, "engines", "indextts")


def run_index_tts2(args):
    """IndexTTS-2 인퍼런스"""
    import torch, gc
    import soundfile as sf

    # torchaudio.load는 torchcodec 사용 → soundfile로 패치
    import torchaudio, soundfile as _sf
    def _sf_torchaudio_load(path, *a, **kw):
        data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr
    torchaudio.load = _sf_torchaudio_load

    sys.path.insert(0, INDEX_TTS_REPO)
    os.chdir(INDEX_TTS_REPO)

    # transformers 4.50+ 에서 QuantizedCacheConfig가 제거됨 → stub 추가
    import transformers.cache_utils as _tcu
    if not hasattr(_tcu, "QuantizedCacheConfig"):
        class _QCC:
            def __init__(self, **kwargs): pass
        _tcu.QuantizedCacheConfig = _QCC

    from indextts.infer_v2 import IndexTTS2

    model_dir = args.model_dir  # D:\models\IndexTTS-2
    cfg_path = os.path.join(model_dir, "config.yaml")

    ref_wav_path, _ = get_reference_data(args)

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    # use_cuda_kernel=True: MAS(Monotonic Alignment Search)를 CUDA 커널로 실행 → 속도 향상 기대
    try:
        tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=True,
            use_cuda_kernel=True,
        )
    except Exception:
        # CUDA 커널 컴파일 실패 시 fallback
        tts = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=True,
            use_cuda_kernel=False,
        )
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        out_path = args.output_path.replace(".wav", f"_{run_idx}.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        reset_vram_peak()
        t0 = time.perf_counter()
        tts.infer(spk_audio_prompt=ref_wav_path, text=args.text, output_path=out_path)
        sync_gpu()
        inf_time = time.perf_counter() - t0

        if not is_warmup:
            data, sr = sf.read(out_path)
            duration = data.shape[0] / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": -1.0,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_bark(args):
    """Bark TTS 인퍼런스"""
    import torch, gc
    import numpy as np
    import soundfile as sf

    # Bark uses old torch.load format (numpy scalars); patch to allow weights_only=False
    _orig_load = torch.load
    def _patched_load(*a, **kw):
        if "weights_only" not in kw:
            kw["weights_only"] = False
        return _orig_load(*a, **kw)
    torch.load = _patched_load

    os.environ["SUNO_OFFLOAD_CPU"] = "0"
    os.environ["SUNO_USE_SMALL_MODELS"] = "0"
    os.environ["XDG_CACHE_HOME"] = args.model_dir

    from bark import SAMPLE_RATE, generate_audio, preload_models

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    preload_models()
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        audio = generate_audio(args.text)
        sync_gpu()
        inf_time = time.perf_counter() - t0

        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio /= max_val
        audio_dur = len(audio) / SAMPLE_RATE

        if not is_warmup:
            out_path = args.output_path.replace(".wav", f"_{run_idx}.wav")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, audio, SAMPLE_RATE, subtype="PCM_16")
            results.append({
                "load_time_s": load_time,
                "ttfa_ms": inf_time * 1000,
                "inference_time_s": inf_time,
                "audio_duration_s": audio_dur,
                "rtf": inf_time / audio_dur if audio_dur > 0 else 0,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": SAMPLE_RATE,
                "output_wav": out_path,
                "success": True,
            })

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


MIO_TTS_REPO = os.path.join(ROOT_DIR, "engines", "mio_tts")
MIO_TTS_CODEC = "Aratako/MioCodec-25Hz-44.1kHz-v2"
MIO_TTS_PRESET = os.path.join(MIO_TTS_REPO, "presets", "en_female.pt")


def run_mio_tts(args):
    """MioTTS 인퍼런스 (직접 transformers + miocodec)"""
    import re
    import torch, gc
    import soundfile as sf

    sys.path.insert(0, MIO_TTS_REPO)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from miocodec import MioCodecModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vram_before = get_vram_mb()
    t_load = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    codec = MioCodecModel.from_pretrained(MIO_TTS_CODEC).eval().to(device)
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    sample_rate = int(codec.config.sample_rate)

    # preset embedding for voice style
    global_embedding = torch.load(MIO_TTS_PRESET, map_location=device, weights_only=True)

    TOKEN_PATTERN = re.compile(r"<\|s_(\d+)\|>")

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0

        messages = [{"role": "user", "content": args.text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        reset_vram_peak()
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = llm.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.8,
                top_p=1.0,
                do_sample=True,
                repetition_penalty=1.0,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=False,
        )
        speech_token_ids = [int(v) for v in TOKEN_PATTERN.findall(generated)]

        if not speech_token_ids:
            raise RuntimeError(f"No speech tokens in output: {generated[:300]}")

        tokens_tensor = torch.tensor(speech_token_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            audio = codec.decode(
                global_embedding=global_embedding.to(device),
                content_token_indices=tokens_tensor,
            )
        sync_gpu()
        inf_time = time.perf_counter() - t0

        audio_data = audio.squeeze().cpu().float().numpy()
        audio_dur = len(audio_data) / sample_rate

        if not is_warmup:
            out_path = args.output_path.replace(".wav", f"_{run_idx}.wav")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, audio_data, sample_rate, subtype="PCM_16")
            results.append({
                "load_time_s": load_time,
                "ttfa_ms": inf_time * 1000,
                "inference_time_s": inf_time,
                "audio_duration_s": audio_dur,
                "rtf": inf_time / audio_dur if audio_dur > 0 else 0,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sample_rate,
                "output_wav": out_path,
                "success": True,
            })

    del llm, codec
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_gpt_sovits(args):
    """GPT-SoVITS V3 인퍼런스"""
    import torch, gc
    import numpy as np
    import soundfile as sf
    import librosa
    import sys
    from io import BytesIO

    GSV_REPO = os.path.join(ROOT_DIR, "engines", "gptsovits")

    # output_path 절대경로 변환 (chdir 전에 필수)
    args.output_path = os.path.abspath(args.output_path)

    sys.path.insert(0, GSV_REPO)
    sys.path.insert(0, os.path.join(GSV_REPO, "GPT_SoVITS"))
    os.chdir(GSV_REPO)  # chinese2.py가 'GPT_SoVITS/text/G2PWModel' 상대경로 사용

    # config를 임포트하기 전에 환경 변수나 패치를 통해 경로를 설정해야 함
    # 하지만 api.py에 정의된 클래스와 함수들이 필요하므로 필요한 것만 가져오거나 모방함
    
    from feature_extractor import cnhubert
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    from module.models import SynthesizerTrnV3
    from AR.models.t2s_lightning_module import Text2SemanticLightningModule
    from process_ckpt import load_sovits_new
    from BigVGAN import bigvgan
    from text import cleaned_text_to_sequence
    from text.cleaner import clean_text
    from module.mel_processing import spectrogram_torch, mel_spectrogram_torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_half = True if device == "cuda" else False
    dtype = torch.float16 if is_half else torch.float32

    # 모델 경로 설정
    model_dir = args.model_dir
    bert_path = os.path.join(model_dir, "chinese-roberta-wwm-ext-large")
    cnhubert_path = os.path.join(model_dir, "chinese-hubert-base")
    sovits_path = os.path.join(model_dir, "s2Gv3.pth")
    gpt_path = os.path.join(model_dir, "s1v3.ckpt")
    bigvgan_path = os.path.join(model_dir, "models--nvidia--bigvgan_v2_24khz_100band_256x")

    vram_before = get_vram_mb()
    t_load = time.perf_counter()

    # Hubert & BERT
    cnhubert.cnhubert_base_path = cnhubert_path
    ssl_model = cnhubert.get_model().to(device)
    if is_half: ssl_model = ssl_model.half()
    
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path).to(device)
    if is_half: bert_model = bert_model.half()

    # GPT 모델 로드
    dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
    gpt_config = dict_s1["config"]
    max_sec = gpt_config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(gpt_config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half: t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device).eval()

    # SoVITS 모델 로드
    dict_s2 = load_sovits_new(sovits_path)
    hps = dict_s2["config"]
    # Dict to Attr helper (simplified)
    class AttrDict(dict):
        def __getattr__(self, key): return self[key]
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict): v = AttrDict(v)
                self[k] = v
    hps = AttrDict(hps)
    
    vq_model = SynthesizerTrnV3(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **dict(hps.model)
    )
    vq_model.load_state_dict(dict_s2["weight"], strict=False)
    if is_half: vq_model = vq_model.half()
    vq_model = vq_model.to(device).eval()

    # Vocoder (BigVGAN)
    vocoder_model = bigvgan.BigVGAN.from_pretrained(bigvgan_path, use_cuda_kernel=False)
    vocoder_model.remove_weight_norm()
    if is_half: vocoder_model = vocoder_model.half()
    vocoder_model = vocoder_model.to(device).eval()

    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    # 전처리기 함수들
    def get_bert_feature(text, word2ph):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs: inputs[i] = inputs[i].to(device)
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        return torch.cat(phone_level_feature, dim=0).T

    def get_phones_and_bert(text, lang):
        phones, word2ph, norm_text = clean_text(text, lang, "v3")
        phones = cleaned_text_to_sequence(phones, "v3")
        if lang == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros((1024, len(phones)), dtype=dtype).to(device)
        return phones, bert, norm_text

    mel_fn = lambda x: mel_spectrogram_torch(x, n_fft=1024, win_size=1024, hop_size=256, num_mels=100, sampling_rate=24000, fmin=0, fmax=None, center=False)
    
    # 벤치마킹 실행
    results = []
    sr = 24000
    
    # 표준 참조 데이터 적용 (GPT-SoVITS V3는 24kHz 사용, 5초 분량)
    ref_wav_path, prompt_text = get_reference_data(args, target_sr=24000, max_sec=5.0)
    MAX_REF_SECS = 5.0

    phones1, bert1, _ = get_phones_and_bert(prompt_text, "ko")

    with torch.no_grad():
        wav16k, _ = librosa.load(ref_wav_path, sr=16000)
        # 5초로 크롭
        wav16k = wav16k[:int(MAX_REF_SECS * 16000)]
        wav16k = torch.from_numpy(wav16k).to(device)
        if is_half: wav16k = wav16k.half()
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0].unsqueeze(0) # [1, T]

        # V3 reference spec (5초 크롭)
        ref_audio, _sr = sf.read(ref_wav_path)
        if ref_audio.ndim == 2:  # stereo → mono
            ref_audio = ref_audio.mean(axis=1)
        ref_audio = ref_audio[:int(MAX_REF_SECS * _sr)]  # 5초 크롭
        ref_audio = torch.from_numpy(ref_audio).to(device).float().unsqueeze(0)
        if _sr != 24000:
            import torchaudio
            ref_audio = torchaudio.transforms.Resample(_sr, 24000).to(device)(ref_audio)

        mel2 = mel_fn(ref_audio)
        mel2 = (mel2 - (-12)) / (2 - (-12)) * 2 - 1 # norm_spec
        refer = spectrogram_torch(ref_audio, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, center=False).to(dtype)

    # Auto-detect language from text (ko/zh/en)
    def _detect_lang(text):
        ko = sum(1 for c in text if '\uac00' <= c <= '\ud7a3' or '\u3130' <= c <= '\u318f')
        zh = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        en = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        total = ko + zh + en
        if total == 0: return "ko"
        if ko > total * 0.2: return "ko"
        if zh > total * 0.2: return "zh"
        return "en"
    text_lang = _detect_lang(args.text)

    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()

        phones2, bert2, _ = get_phones_and_bert(args.text, text_lang)
        bert = torch.cat([bert1, bert2], 1).unsqueeze(0)
        all_phones = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        all_len = torch.tensor([all_phones.shape[-1]]).to(device)
        
        with torch.no_grad():
            # 1. GPT Inference
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phones, all_len, prompt_semantic, bert,
                top_k=15, top_p=1.0, temperature=1.0, early_stop_num=50 * max_sec
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
            
            # 2. SoVITS CFM Inference (V3 logic)
            phoneme_ids0 = torch.LongTensor(phones1).to(device).unsqueeze(0)
            phoneme_ids1 = torch.LongTensor(phones2).to(device).unsqueeze(0)
            
            fea_ref, ge = vq_model.decode_encp(prompt_semantic.unsqueeze(0), phoneme_ids0, refer)
            fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, 1.0)
            
            T_min = min(mel2.shape[2], fea_ref.shape[2])
            # Clip T_min to T_ref to ensure chunk_len > 0
            Tchunk, Tref = 934, 468
            if T_min > Tref:
                mel2 = mel2[:, :, -Tref:]
                fea_ref = fea_ref[:, :, -Tref:]
                T_min = Tref
            m2 = mel2[:, :, :T_min].to(dtype)
            f_ref = fea_ref[:, :, :T_min]

            chunk_len = Tchunk - T_min
            idx_fea = 0
            cfm_resss = []
            while True:
                fea_chunk = fea_todo[:, :, idx_fea : idx_fea + chunk_len]
                if fea_chunk.shape[-1] == 0: break
                idx_fea += chunk_len
                fea = torch.cat([f_ref, fea_chunk], 2).transpose(2, 1)
                cfm_res = vq_model.cfm.inference(
                    fea, torch.LongTensor([fea.size(1)]).to(device), m2, 32, 0
                )
                cfm_res = cfm_res[:, :, m2.shape[2] :]
                m2 = cfm_res[:, :, -T_min:]
                f_ref = fea_chunk[:, :, -T_min:]
                cfm_resss.append(cfm_res)
            
            cfm_res_full = torch.cat(cfm_resss, 2)
            cfm_res_full = (cfm_res_full + 1) / 2 * (2 - (-12)) + (-12) # denorm_spec
            
            # 3. Vocoder
            wav_gen = vocoder_model(cfm_res_full)
            audio = wav_gen[0][0].cpu().float().numpy()
        
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000 # Non-streaming
        
        if not is_warmup:
            max_val = np.abs(audio).max()
            if max_val > 1.0: audio /= max_val
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio, sr, subtype='PCM_16')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })
            
    # Cleanup
    del t2s_model, vq_model, vocoder_model, ssl_model, bert_model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_f5tts(args):
    """F5-TTS 인퍼런스 (flow matching, zero-shot voice cloning)"""
    import torch, gc, numpy as np, soundfile as sf

    # torchaudio 2.10+: torchcodec 필요 → soundfile로 monkey-patch
    import torchaudio as _ta_f5, soundfile as _sf_f5, io as _io_f5
    if not hasattr(_ta_f5, '_orig_load'):
        _ta_f5._orig_load = _ta_f5.load
        def _sf_load_f5(uri, frame_offset=0, num_frames=-1, normalize=True,
                        channels_first=True, format=None, buffer_size=4096, backend=None):
            if hasattr(uri, 'read'):
                data, sr = _sf_f5.read(uri, dtype='float32', always_2d=True)
            elif isinstance(uri, (bytes, bytearray)):
                data, sr = _sf_f5.read(_io_f5.BytesIO(uri), dtype='float32', always_2d=True)
            else:
                data, sr = _sf_f5.read(str(uri), dtype='float32', always_2d=True)
            t = torch.from_numpy(data.T.copy() if data.ndim > 1 else data.reshape(1, -1))
            if not channels_first and t.ndim > 1:
                t = t.T
            if frame_offset > 0:
                t = t[..., frame_offset:]
            if num_frames > 0 and num_frames != -1:
                t = t[..., :num_frames]
            return t, sr
        _ta_f5.load = _sf_load_f5

    from f5_tts.api import F5TTS

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = F5TTS(device=device)
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    # 텍스트 언어 감지: 한국어 텍스트이면 한국어 ref 사용 (F5-TTS는 영어 학습)
    _ko_chars = sum(1 for c in args.text if '\uac00' <= c <= '\ud7a3')
    _is_korean = _ko_chars > len(args.text) * 0.2

    # 표준 측정 규칙 v2: 참조 음성 루프 밖에서 1회만 로드
    REF_WAV, REF_TXT = get_reference_data(args.ref_key)

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        wav, sr, _ = model.infer(
            ref_file=REF_WAV,
            ref_text=REF_TXT,
            gen_text=args.text,
            remove_silence=False,
        )
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            if isinstance(wav, np.ndarray):
                audio = wav
            else:
                audio = wav.cpu().float().numpy() if hasattr(wav, 'cpu') else np.array(wav)
            if audio.ndim > 1:
                audio = audio.squeeze()
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio /= max_val
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio, sr, subtype='PCM_16')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_xtts(args):
    """XTTS v2 인퍼런스 (17개 언어, 한국어 공식 지원)"""
    import torch, gc, numpy as np, soundfile as sf

    # torchaudio 2.10+: torchcodec (ffmpeg) 필요 → soundfile로 monkey-patch
    import torchaudio as _ta_xtts, soundfile as _sf_xtts, io as _io_xtts
    if not hasattr(_ta_xtts, '_orig_load'):
        _ta_xtts._orig_load = _ta_xtts.load
        def _sf_load_xtts(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, backend=None, encoding_args=None):
            if hasattr(uri, 'read'):
                data, sr = _sf_xtts.read(uri, always_2d=False, dtype='float32')
            elif isinstance(uri, (bytes, bytearray)):
                data, sr = _sf_xtts.read(_io_xtts.BytesIO(uri), always_2d=False, dtype='float32')
            else:
                data, sr = _sf_xtts.read(str(uri), always_2d=False, dtype='float32')
            if data.ndim == 1:
                tensor = torch.from_numpy(data.reshape(1, -1))
            else:
                tensor = torch.from_numpy(data.T.copy())
            return tensor, sr
        _ta_xtts.load = _sf_load_xtts

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    # 텍스트 언어 감지
    _ko_chars = sum(1 for c in args.text if '\uac00' <= c <= '\ud7a3')
    _zh_chars = sum(1 for c in args.text if '\u4e00' <= c <= '\u9fff')
    _total = _ko_chars + _zh_chars + sum(1 for c in args.text if c.isalpha() and ord(c) < 128)
    if _ko_chars > _total * 0.2:
        xtts_lang = "ko"
    elif _zh_chars > _total * 0.2:
        xtts_lang = "zh-cn"
    else:
        xtts_lang = "en"

    # 표준 측정 규칙 v2: 참조 음성 루프 밖에서 1회만 로드
    REF_WAV, _ = get_reference_data(args)

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
        reset_vram_peak()
        t0 = time.perf_counter()
        # Dual Voice Support: Cloning vs Preset
        if getattr(args, "ref_key", "") == "default_preset":
            # XTTS v2 built-in speaker — try 'Claribel', fall back to zero-shot cloning
            try:
                tts.tts_to_file(
                    text=args.text,
                    speaker="Claribel",
                    language=xtts_lang,
                    file_path=out_path,
                )
            except Exception:
                # Claribel .pth not in local cache → fall back to zero-shot cloning
                tts.tts_to_file(
                    text=args.text,
                    speaker_wav=REF_WAV,
                    language=xtts_lang,
                    file_path=out_path,
                )
        else:
            # 표준 참조 데이터 적용 (Cloning)
            tts.tts_to_file(
                text=args.text,
                speaker_wav=REF_WAV,
                language=xtts_lang,
                file_path=out_path,
            )
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            audio, sr = sf.read(out_path, dtype='float32')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_melotts(args):
    """MeloTTS 한국어 인퍼런스 (VITS 기반, CPU 실시간)"""
    import torch, gc, numpy as np, soundfile as sf

    # 텍스트 언어 감지
    _ko_chars = sum(1 for c in args.text if '\uac00' <= c <= '\ud7a3')
    _is_korean = _ko_chars > len(args.text) * 0.2
    if not _is_korean:
        melo_lang = 'EN'
    else:
        melo_lang = 'KR'  # MeloTTS uses 'KR' not 'KO'

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    from melo.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TTS(language=melo_lang, device=device)
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    speaker_ids = model.hps.data.spk2id
    # KO speaker: 'KR', EN speakers: 'EN-Default', 'EN-US', etc.
    speaker_key = 'KR' if melo_lang == 'KR' else 'EN-Default'
    if speaker_key not in speaker_ids:
        speaker_key = list(speaker_ids.keys())[0]
    speaker_id = speaker_ids[speaker_key]

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
        reset_vram_peak()
        t0 = time.perf_counter()
        model.tts_to_file(args.text, speaker_id, out_path, speed=1.0)
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            audio, sr = sf.read(out_path, dtype='float32')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": model.hps.data.sampling_rate,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_chatterbox(args):
    """Chatterbox TTS 인퍼런스 (ResembleAI, 23개 언어 지원)"""
    import torch, gc, numpy as np, soundfile as sf

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    from chatterbox.tts import ChatterboxTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained(device=device)
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    # 표준 측정 규칙 v2: 참조 음성 루프 밖에서 1회만 로드
    REF_WAV, _ = get_reference_data(args)

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        wav = model.generate(
            text=args.text,
            audio_prompt_path=REF_WAV,
        )
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            if hasattr(wav, 'cpu'):
                audio = wav.squeeze().cpu().float().numpy()
            else:
                audio = np.array(wav).squeeze()
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio /= max_val
            sr = model.sr
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio, sr, subtype='PCM_16')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_outetss(args):
    """OuteTTS 1.0 인퍼런스 (LLM-기반 TTS, 9개 언어)"""
    import torch, gc, numpy as np, soundfile as sf

    # torchaudio DLL 경로 설정 (Windows)
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    os.environ['PATH'] = _torch_lib + os.pathsep + os.environ.get('PATH', '')

    # torchaudio 2.10+: torchcodec 필요 → soundfile로 monkey-patch (outetts 내부도 포함)
    import torchaudio as _ta_oute, soundfile as _sf_oute, io as _io_oute
    if not hasattr(_ta_oute, '_orig_load'):
        _ta_oute._orig_load = _ta_oute.load
        def _sf_load_oute(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, backend=None, encoding_args=None):
            import numpy as _np_oute
            if hasattr(uri, 'read'):
                data, sr = _sf_oute.read(uri, always_2d=False, dtype='float32')
            elif isinstance(uri, (bytes, bytearray)):
                data, sr = _sf_oute.read(_io_oute.BytesIO(uri), always_2d=False, dtype='float32')
            else:
                data, sr = _sf_oute.read(str(uri), always_2d=False, dtype='float32')
            if data.ndim == 1:
                tensor = torch.from_numpy(data.reshape(1, -1))
            else:
                tensor = torch.from_numpy(data.T.copy())
            return tensor, sr
        _ta_oute.load = _sf_load_oute

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    import outetts

    # 텍스트 언어 감지
    _ko_chars = sum(1 for c in args.text if '\uac00' <= c <= '\ud7a3')
    _is_korean = _ko_chars > len(args.text) * 0.2

    interface = outetts.Interface(
        config=outetts.ModelConfig(
            model=args.model_dir,
            backend=outetts.Backend.HF,
        )
    )
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    # Dual Voice Support: Cloning vs Preset
    if getattr(args, "ref_key", "") == "default_preset":
        # Using built-in default speaker
        speaker = interface.load_default_speaker("en-female-1-neutral")
    else:
        # Cloning from reference audio
        ref_wav, _ = get_reference_data(args)
        speaker = interface.create_speaker(audio_path=ref_wav)

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        output = interface.generate(
            config=outetts.GenerationConfig(
                text=args.text,
                generation_type=outetts.GenerationType.CHUNKED,
                speaker=speaker,
                sampler_config=outetts.SamplerConfig(
                    temperature=0.4,
                    repetition_penalty=1.1,
                    top_k=40,
                    top_p=0.9,
                ),
            )
        )
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            # torchaudio.save broken in 2.10+ → save via soundfile directly
            _oute_audio = output.audio.detach().cpu().float()
            if _oute_audio.dim() == 3:
                _oute_audio = _oute_audio[0]
            if _oute_audio.dim() == 2 and _oute_audio.shape[0] == 1:
                _oute_audio = _oute_audio[0]
            _oute_np = _oute_audio.numpy()
            sr = output.sr
            out_path_norm = out_path.replace('\\', '/')
            sf.write(out_path_norm, _oute_np, sr, subtype='PCM_16')
            duration = len(_oute_np) / sr
            out_path = out_path_norm
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del interface
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_kokoro(args):
    """Kokoro-TTS 인퍼런스 (82M VITS, EN/ZH/JP/ES/FR/HI/IT/PT 지원, EN 특화)"""
    import torch, gc, numpy as np, soundfile as sf

    # 텍스트 언어 감지 - 한국어는 kokoro 미지원이라 EN 폴백
    _ko_chars = sum(1 for c in args.text if '\uac00' <= c <= '\ud7a3')
    _is_korean = _ko_chars > len(args.text) * 0.2
    if _is_korean:
        # 한국어 텍스트는 EN으로 처리 불가 → 짧은 EN 문장으로 대체 (메타데이터용)
        gen_text = "This text was originally in Korean but Kokoro does not support Korean. This is a placeholder."
        lang_code = 'a'
    else:
        gen_text = args.text
        lang_code = 'a'  # American English

    # KPipeline 생성
    from kokoro import KPipeline

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    pipeline = KPipeline(lang_code=lang_code)
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        # generator → 청크별 오디오 생성
        audio_chunks = []
        generator = pipeline(gen_text, voice='af_heart', speed=1.0)
        for _, _, audio_chunk in generator:
            if audio_chunk is not None:
                audio_chunks.append(audio_chunk)
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            if audio_chunks:
                audio = np.concatenate(audio_chunks, axis=-1)
            else:
                audio = np.zeros(24000, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.squeeze()
            sr = 24000
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio, sr, subtype='PCM_16')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_parler(args):
    """Parler-TTS mini 인퍼런스 (description 기반, EN 전용)"""
    import torch, gc, numpy as np, soundfile as sf

    # 한국어 감지 → EN 폴백
    _ko_chars = sum(1 for c in args.text if '\uac00' <= c <= '\ud7a3')
    _is_korean = _ko_chars > len(args.text) * 0.2
    if _is_korean:
        gen_text = "Artificial intelligence speech synthesis technology has made remarkable progress."
        rtf_note = "KO_FALLBACK"
    else:
        gen_text = args.text
        rtf_note = ""

    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    # 음성 스타일 description
    description = (
        "A female speaker with a slightly low-pitched, expressive voice delivers her words clearly. "
        "The recording has very high quality audio in a quiet environment."
    )

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()

        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(gen_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
            )
        audio_arr = generation.cpu().numpy().squeeze()

        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            sr = model.config.sampling_rate
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio_arr, sr, subtype='PCM_16')
            duration = len(audio_arr) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_styletts2(args):
    """StyleTTS2 인퍼런스 (EN 고품질, phonemizer 없이 kokoro phonemizer 활용)"""
    import torch, gc, numpy as np, soundfile as sf

    # 한국어 감지 → EN 폴백
    _ko_chars = sum(1 for c in args.text if '\uac00' <= c <= '\ud7a3')
    _is_korean = _ko_chars > len(args.text) * 0.2
    if _is_korean:
        gen_text = "Artificial intelligence speech synthesis technology has made remarkable progress. The combination of natural language processing and deep learning makes speech synthesis more natural and fluent."
    else:
        gen_text = args.text

    import styletts2
    from styletts2 import tts

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    _tts = tts.StyleTTS2()
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        wav = _tts.inference(gen_text, diffusion_steps=5, embedding_scale=1)
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            sr = 24000
            if isinstance(wav, np.ndarray):
                audio = wav
            else:
                audio = np.array(wav)
            if audio.ndim > 1:
                audio = audio.squeeze()
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio /= max_val
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio, sr, subtype='PCM_16')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del _tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_dia(args):
    """Dia-1.6B 인퍼런스 (Nari Labs, 대화형 TTS, EN 위주)"""
    import torch, gc, numpy as np, soundfile as sf

    # 한국어 감지 → EN 폴백 (Dia는 EN 전용)
    _ko_chars = sum(1 for c in args.text if '\uac00' <= c <= '\ud7a3')
    _is_korean = _ko_chars > len(args.text) * 0.2
    if _is_korean:
        gen_text = "[S1] Artificial intelligence speech synthesis technology has made remarkable progress. [S2] The combination of natural language processing and deep learning makes speech synthesis more natural and fluent."
    else:
        # Dia uses [S1] [S2] dialogue format
        gen_text = f"[S1] {args.text}"

    from dia.model import Dia

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        output = model.generate(gen_text, use_torch_compile=False, verbose=False)
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            sr = 44100
            if isinstance(output, np.ndarray):
                audio = output
            elif hasattr(output, 'cpu'):
                audio = output.cpu().float().numpy()
            else:
                audio = np.array(output)
            if audio.ndim > 1:
                audio = audio.squeeze()
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio /= max_val
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio, sr, subtype='PCM_16')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_openvoice(args):
    """OpenVoice v2 인퍼런스 (한국어 공식 지원, MyShell AI)"""
    import torch, gc, numpy as np, soundfile as sf, os

    # 언어 감지
    _ko_chars = sum(1 for c in args.text if '\uac00' <= c <= '\ud7a3')
    _is_korean = _ko_chars > len(args.text) * 0.2
    tts_lang = 'KR' if _is_korean else 'EN'

    from melo.api import TTS as MeloTTS
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter

    # model_dir should point to the checkpoints_v2 base directory
    ckpt_base = args.model_dir
    tone_color_converter = ToneColorConverter(
        os.path.join(ckpt_base, "converter", "config.json"),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    tone_color_converter.load_ckpt(os.path.join(ckpt_base, "converter", "checkpoint.pth"))

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MeloTTS(language=tts_lang, device=device)
    speaker_ids = dict(model.hps.data.spk2id)
    spk_id = speaker_ids.get('KR', speaker_ids.get('EN-US', list(speaker_ids.values())[0]))

    # 표준 참조 데이터 적용
    ref_wav, _ = get_reference_data(args)
    target_se, _ = se_extractor.get_se(
        ref_wav,
        tone_color_converter,
        vad=False
    )
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        reset_vram_peak()
        t0 = time.perf_counter()

        # Dual Voice Support: Cloning vs Preset
        if getattr(args, "ref_key", "") == "default_preset":
            # Default: MeloTTS output without tone conversion
            model.tts_to_file(args.text, spk_id, out_path, speed=1.0)
        else:
            # Cloning: MeloTTS -> Tone Color Conversion
            tmp_path = args.output_path.replace('.wav', f'_tmp_{run_idx}.wav')
            model.tts_to_file(args.text, spk_id, tmp_path, speed=1.0)

            # Apply tone color conversion
            src_se, _ = se_extractor.get_se(tmp_path, tone_color_converter, vad=False)
            tone_color_converter.convert(
                audio_src_path=tmp_path,
                src_se=src_se,
                tgt_se=target_se,
                output_path=out_path,
                message="@MyShell",
            )
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            data, sr = sf.read(out_path)
            duration = len(data) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del model, tone_color_converter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_mars5(args):
    """MARS5-TTS (Camb-AI) 영어 Zero-shot 인퍼런스"""
    import torch, gc
    import numpy as np
    import soundfile as sf

    MARS5_REPO = os.path.join(ROOT_DIR, 'engines', 'mars5', 'repo')
    sys.path.insert(0, MARS5_REPO)

    from inference import Mars5TTS, InferenceConfig

    # 영어 reference 오디오: MeloTTS EN 출력 사용 (빠르고 명확한 EN 음성)
    REF_WAV_EN = os.path.join(BENCH_DIR, 'results_en_r2', 'wav', 'melotts', 'MeloTTS-KO_0.wav')
    from benchmark.core import TEST_TEXTS
    REF_TEXT_EN = TEST_TEXTS['en_medium']

    if not os.path.exists(REF_WAV_EN):
        raise FileNotFoundError(f"Reference audio not found: {REF_WAV_EN}. Run melotts EN benchmark first.")

    # reference 오디오 로드 (최대 12초, MARS5 내부 SR=24000으로 리샘플링)
    ref_wav_np, ref_sr = sf.read(REF_WAV_EN, dtype='float32', always_2d=True)
    if ref_wav_np.shape[1] > 1:
        ref_wav_np = ref_wav_np.mean(axis=1)
    else:
        ref_wav_np = ref_wav_np[:, 0]
    MARS5_SR = 24000
    if ref_sr != MARS5_SR:
        import math
        from scipy.signal import resample_poly
        g = math.gcd(ref_sr, MARS5_SR)
        ref_wav_np = resample_poly(ref_wav_np, MARS5_SR // g, ref_sr // g)
    ref_wav_np = ref_wav_np[:MARS5_SR * 12]
    ref_tensor = torch.tensor(ref_wav_np, dtype=torch.float32).unsqueeze(0)

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    mars5 = Mars5TTS.from_pretrained('camb-ai/mars5-tts')
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    results = []
    cfg = InferenceConfig(deep_clone=False)  # shallow clone: faster

    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        ar_codes, wav_out = mars5.tts(args.text, ref_tensor, REF_TEXT_EN, cfg)
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000  # MARS5는 스트리밍 없음, 전체 생성 후 반환

        if not is_warmup:
            audio_np = wav_out.squeeze().cpu().numpy()
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio_np, mars5.sr, subtype='PCM_16')
            duration = audio_np.shape[0] / mars5.sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": mars5.sr,
                "output_wav": out_path,
                "success": True,
            })

    del mars5
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_zipvoice_finetuned(args):
    """ZipVoice Fine-tuned (REST API) - KO 전용
    Docker 컨테이너의 tts-api-simple (port 9102) v2 audio 엔드포인트 사용.
    /api/v2/tts-engine/synthesize/audio → audio/wav bytes 직접 반환"""
    import io
    import soundfile as sf
    import requests

    API_URL = "http://localhost:9102/api/v2/tts-engine/synthesize/audio"
    VOICE_ID = "spk066"  # 한국어 여성 아나운서 (ZipVoice 파인튜닝 음성)

    vram_before = get_vram_mb()

    # 표준 측정 규칙 v2: REST API 모델은 WARMUP_RUNS회 요청으로 서버 캐시 안정화
    t_load = time.perf_counter()
    for _w in range(WARMUP_RUNS):
        warmup_resp = requests.post(
            API_URL,
            json={"text": args.text, "voice_id": VOICE_ID, "format": "wav"},
            timeout=120,
        )
        if warmup_resp.status_code != 200:
            raise RuntimeError(f"Warmup {_w+1} 실패: {warmup_resp.status_code} {warmup_resp.text[:200]}")
    load_time = time.perf_counter() - t_load

    vram_after_load = get_vram_mb()
    results = []

    for run_idx in range(args.runs):
        reset_vram_peak()
        t0 = time.perf_counter()
        resp = requests.post(
            API_URL,
            json={"text": args.text, "voice_id": VOICE_ID, "format": "wav"},
            timeout=120,
        )
        sync_gpu()
        inf_time = time.perf_counter() - t0

        if resp.status_code != 200:
            raise RuntimeError(f"Synthesis 실패: {resp.status_code} {resp.text[:200]}")

        wav_bytes = resp.content
        audio_np, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=-1)
        duration = len(audio_np) / sr

        out_path = args.output_path.replace(".wav", f"_{run_idx}.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, audio_np, sr, subtype="PCM_16")

        results.append({
            "run_index": run_idx,
            "load_time_s": load_time,
            "ttfa_ms": inf_time * 1000,
            "inference_time_s": inf_time,
            "audio_duration_s": duration,
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after_load,
            "vram_peak_mb": get_vram_peak_mb(),
            "sample_rate": sr,
            "output_wav": out_path,
            "success": True,
        })

    return results


def run_zipvoice(args):
    """ZipVoice Official (CLI based) - EN/ZH 지원
    HuggingFace에서 모델을 다운로드하거나 local model_dir을 사용하여 추론.
    코드가 복잡하여 subprocess로 직접 실행."""
    import subprocess
    import soundfile as sf
    import numpy as np

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    load_time = 0 
    vram_after_load = vram_before 
    
    # Get reference data
    REF_WAV, REF_TEXT = get_reference_data(args)

    results = []
    for run_idx in range(args.runs):
        reset_vram_peak()
        out_path = args.output_path.replace(".wav", f"_{run_idx}.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        t0 = time.perf_counter()
        
        # Subprocess env
        env = os.environ.copy()
        env["PYTHONPATH"] = args.model_dir
        env["PYTHONUTF8"] = "1"
        
        # Command construction (Official script)
        cmd = [
            sys.executable, "-m", "zipvoice.bin.infer_zipvoice",
            "--model-name", "zipvoice",
            "--prompt-wav", REF_WAV,
            "--prompt-text", REF_TEXT,
            "--text", args.text,
            "--res-wav-path", out_path
        ]
        
        # Execute in original model dir
        ret = subprocess.run(cmd, cwd=args.model_dir, env=env, capture_output=True, text=True)
        sync_gpu()
        inf_time = time.perf_counter() - t0
        
        if ret.returncode != 0:
            raise RuntimeError(f"ZipVoice CLI 실패: {ret.stderr}")

        # Result verification
        if os.path.exists(out_path):
            audio, sr = sf.read(out_path)
            duration = len(audio) / sr
        else:
            duration = 0
            sr = 24000

        results.append({
            "run_index": run_idx,
            "load_time_s": load_time,
            "ttfa_ms": inf_time * 1000,
            "inference_time_s": inf_time,
            "audio_duration_s": duration,
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after_load,
            "vram_peak_mb": get_vram_peak_mb(), 
            "sample_rate": sr,
            "output_wav": out_path,
            "success": True,
        })

    return results




def run_chattts(args):
    """ChatTTS 인퍼런스 (2noise, 대화형 특화)"""
    import torch, gc, numpy as np, soundfile as sf
    import sys

    CHATTTS_REPO = os.path.join(ROOT_DIR, 'models', 'ChatTTS')
    if not os.path.exists(CHATTTS_REPO):
         CHATTTS_REPO = args.model_dir

    sys.path.insert(0, CHATTTS_REPO)
    import ChatTTS

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    chat = ChatTTS.Chat()
    chat.load_models(source='local', local_path=args.model_dir)
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    # Dual Voice Support: Cloning vs Preset
    if getattr(args, "ref_key", "") == "default_preset":
        # 고정 시드 화자 사용 (Default)
        rand_spk = chat.sample_random_speaker(seed=2222)
    else:
        # 무작위 화자 (또는 추후 오디오 기반 임베딩 추출 로직 추가 가능)
        rand_spk = chat.sample_random_speaker()

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        
        # params_refine_text = ChatTTS.Chat.RefineTextParams(prompt='[oral_2][laugh_0][break_6]')
        wavs = chat.infer([args.text], use_decoder=True, params_infer_code=ChatTTS.Chat.InferCodeParams(spk_emb=rand_spk))
        
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            audio = np.array(wavs[0]).flatten()
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio /= max_val
            sr = 24000
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio, sr, subtype='PCM_16')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del chat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_piper(args):
    """Piper TTS 인퍼런스 (ONNX 기반, 초고속 CPU 실행)"""
    import torch, gc, numpy as np, soundfile as sf
    import subprocess
    import tempfile

    # piper는 보통 전용 venv의 piper 패키지나 바이너리 사용
    # 여기서는 piper-ko 폴더 내의 onnx 모델을 사용함
    onnx_path = os.path.join(args.model_dir, "piper-kss-korean.onnx")
    if not os.path.exists(onnx_path):
        # search for any .onnx in model_dir
        import glob
        onnx_files = glob.glob(os.path.join(args.model_dir, "*.onnx"))
        if onnx_files:
            onnx_path = onnx_files[0]
        else:
            raise FileNotFoundError(f"ONNX model not found in {args.model_dir}")

    vram_before = get_vram_mb() # piper is mostly CPU
    t_load = time.perf_counter()
    # Piper doesn't really have a 'load' in the same way, but we'll count the check
    load_time = time.perf_counter() - t_load

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
        
        t0 = time.perf_counter()
        # piper.exe --model [model] --output_file [out]
        # piper 패키지가 설치되어 있다면 'piper' 명령어로 실행 가능
        # 여기서는 subprocess로 실행하거나 piper 패키지 사용
        # 유닛 테스트 환경에서는 piper 패키지 선호
        try:
            from piper.voice import PiperVoice
            voice = PiperVoice.load(onnx_path)
            with open(out_path, "wb") as f:
                voice.synthesize(args.text, f)
            sr = voice.config.sample_rate
        except ImportError:
            # Fallback to subprocess if package not in venv
            # Assume 'piper' executable is in PATH or near the model
            cmd = ["piper", "--model", onnx_path, "--output_file", out_path]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            process.communicate(input=args.text)
            sr = 22050 # default fallback
        
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            data, sr = sf.read(out_path)
            duration = len(data) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_before,
                "vram_peak_mb": vram_before,
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    return results


def run_voicecraftx(args):
    """VoiceCraft-X 인퍼런스 (11개 언어, Zero-shot voice cloning)"""
    import torch, gc, numpy as np, soundfile as sf, sys

    # Add torch + conda DLL dirs to PATH so torchaudio can find its libs
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    # For conda envs: also add Library\bin and conda base\Library\bin
    conda_prefix = os.path.dirname(os.path.dirname(sys.executable))
    conda_lib_bin = os.path.join(conda_prefix, 'Library', 'bin')
    conda_bin = os.path.join(conda_prefix, 'bin')
    for p in [torch_lib, conda_lib_bin, conda_bin]:
        if p not in os.environ.get('PATH', ''):
            os.environ['PATH'] = p + os.pathsep + os.environ.get('PATH', '')

    import torchaudio

    # torchaudio 2.10+ requires torchcodec → monkey-patch with soundfile
    def _sf_load(uri, frame_offset=0, num_frames=-1, **kw):
        data, sr = sf.read(uri, dtype='float32', always_2d=True)
        data = data.T
        if frame_offset: data = data[:, frame_offset:]
        if num_frames > 0: data = data[:, :num_frames]
        return torch.from_numpy(data), sr
    torchaudio.load = _sf_load

    VCX_REPO = os.path.join(ROOT_DIR, 'engines', 'voicecraftx', 'repo')
    sys.path.insert(0, VCX_REPO)
    sys.path.insert(0, os.path.join(VCX_REPO, 'src'))
    
    from helper import load_tokenizer, load_voicecraftx, load_speaker_model, generate
    from omegaconf import OmegaConf

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    
    # Load config from yaml
    tts_cfg_path = os.path.join(VCX_REPO, 'src', 'config', 'inference', 'tts.yaml')
    config = OmegaConf.load(tts_cfg_path)
    config.pretrained_models = args.model_dir
    config.voicecraftx_path = os.path.join(args.model_dir, "voicecraftx.ckpt")
    config.skip_zh_tn_model = True
    config.model.attn_implementation = "sdpa"  # flash_attention_2 not available on Windows

    text_tok, aud_tok = load_tokenizer(config)
    model = load_voicecraftx(config)
    spk_model = load_speaker_model(config)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    # Move models to device
    aud_tok = aud_tok.to(_device)
    model = model.to(_device)

    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    # Detect language from text characters
    _ko = sum(1 for c in args.text if '\uac00' <= c <= '\ud7a3' or '\u3130' <= c <= '\u318f')
    _zh = sum(1 for c in args.text if '\u4e00' <= c <= '\u9fff')
    _ja = sum(1 for c in args.text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
    _total = _ko + _zh + _ja + sum(1 for c in args.text if c.isalpha() and ord(c) < 128)
    if _ko > _total * 0.2:
        voicecraft_lang = "korean"
    elif _zh > _total * 0.2:
        voicecraft_lang = "mandarin"
    elif _ja > _total * 0.2:
        voicecraft_lang = "japanese"
    else:
        voicecraft_lang = "english"

    # 표준 측정 규칙 v2: 참조 음성 루프 밖에서 1회만 로드 (16kHz mono, max 10s)
    REF_WAV, REF_TXT = get_reference_data(args, target_sr=16000, max_sec=10.0)

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()

        outputs = generate(
            prompt_audio=REF_WAV,
            prompt_text=REF_TXT,
            target_text=args.text,
            config=config,
            model=model,
            speaker_model=spk_model,
            text_tokenizer=text_tok,
            audio_tokenizer=aud_tok,
            language=voicecraft_lang,
            n_samples=1,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            # Decode codec tokens → waveform
            gen_audio = aud_tok.decode(outputs[0])
            audio = gen_audio[0].detach().cpu().float().numpy()
            if audio.ndim > 1:
                audio = audio.squeeze()
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio /= max_val
            sr = 16000
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio, sr, subtype='PCM_16')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del model, spk_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_maskgct(args):
    """MaskGCT 인퍼런스 (Amphion, Fast non-autoregressive TTS)"""
    import torch, gc, numpy as np, soundfile as sf
    import safetensors.torch

    os.environ.setdefault('PHONEMIZER_ESPEAK_LIBRARY', r'C:\Program Files\eSpeak NG\libespeak-ng.dll')

    MASKGCT_ROOT = os.path.join(ROOT_DIR, 'engines', 'maskgct', 'Amphion')
    sys.path.insert(0, MASKGCT_ROOT)
    _orig_cwd = os.getcwd()
    os.chdir(MASKGCT_ROOT)

    try:
        from models.tts.maskgct.maskgct_utils import (
            MaskGCT_Inference_Pipeline,
            build_t2s_model, build_s2a_model,
            build_semantic_model, build_semantic_codec, build_acoustic_codec,
            load_config,
        )
        from huggingface_hub import hf_hub_download

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cfg_path = os.path.join(MASKGCT_ROOT, "models/tts/maskgct/config/maskgct.json")
        cfg = load_config(cfg_path)

        vram_before = get_vram_mb()
        t_load = time.perf_counter()

        # Build models
        semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
        semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
        codec_encoder, codec_decoder = build_acoustic_codec(cfg.model.acoustic_codec, device)
        t2s_model = build_t2s_model(cfg.model.t2s_model, device)
        s2a_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
        s2a_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

        # Load checkpoints (from HF hub, cached locally)
        def _dl(fname): return hf_hub_download("amphion/MaskGCT", filename=fname)
        safetensors.torch.load_model(semantic_codec, _dl("semantic_codec/model.safetensors"))
        safetensors.torch.load_model(codec_encoder, _dl("acoustic_codec/model.safetensors"))
        safetensors.torch.load_model(codec_decoder, _dl("acoustic_codec/model_1.safetensors"))
        safetensors.torch.load_model(t2s_model, _dl("t2s_model/model.safetensors"))
        safetensors.torch.load_model(s2a_1layer, _dl("s2a_model/s2a_model_1layer/model.safetensors"))
        safetensors.torch.load_model(s2a_full, _dl("s2a_model/s2a_model_full/model.safetensors"))

        pipeline = MaskGCT_Inference_Pipeline(
            semantic_model, semantic_codec, codec_encoder, codec_decoder,
            t2s_model, s2a_1layer, s2a_full,
            semantic_mean, semantic_std, device,
        )
        load_time = time.perf_counter() - t_load
        vram_after_load = get_vram_mb()

        # Detect language
        _ko = sum(1 for c in args.text if '\uac00' <= c <= '\ud7a3')
        _zh = sum(1 for c in args.text if '\u4e00' <= c <= '\u9fff')
        n = max(len(args.text), 1)
        tgt_lang = "ko" if _ko/n > 0.2 else ("zh" if _zh/n > 0.2 else "en")

        ref_wav, ref_text = get_reference_data(args)

        # Detect ref text language separately (ref audio may differ from target)
        _ref_ko = sum(1 for c in ref_text if '\uac00' <= c <= '\ud7a3')
        _ref_zh = sum(1 for c in ref_text if '\u4e00' <= c <= '\u9fff')
        _ref_n = max(len(ref_text), 1)
        prompt_lang = "ko" if _ref_ko/_ref_n > 0.2 else ("zh" if _ref_zh/_ref_n > 0.2 else "en")

        results = []
        for run_idx in range(-WARMUP_RUNS, args.runs):
            is_warmup = run_idx < 0
            reset_vram_peak()
            t0 = time.perf_counter()

            recovered_audio = pipeline.maskgct_inference(
                prompt_speech_path=ref_wav,
                prompt_text=ref_text,
                target_text=args.text,
                language=prompt_lang,
                target_language=tgt_lang,
            )

            sync_gpu()
            inf_time = time.perf_counter() - t0

            if not is_warmup:
                import torch as _torch
                if isinstance(recovered_audio, _torch.Tensor):
                    audio = recovered_audio.squeeze().cpu().float().numpy()
                else:
                    audio = np.array(recovered_audio, dtype=np.float32).squeeze()
                sr = 24000
                max_val = np.abs(audio).max()
                if max_val > 1.0:
                    audio /= max_val
                out_path = os.path.abspath(args.output_path.replace('.wav', f'_{run_idx}.wav'))
                sf.write(out_path, audio, sr, subtype='PCM_16')
                duration = len(audio) / sr
                results.append({
                    "run_index": run_idx,
                    "load_time_s": load_time,
                    "ttfa_ms": inf_time * 1000,
                    "inference_time_s": inf_time,
                    "audio_duration_s": duration,
                    "vram_before_mb": vram_before,
                    "vram_after_mb": vram_after_load,
                    "vram_peak_mb": get_vram_peak_mb(),
                    "sample_rate": sr,
                    "output_wav": out_path,
                    "success": True,
                })

        del pipeline, semantic_model, semantic_codec, codec_encoder, codec_decoder
        del t2s_model, s2a_1layer, s2a_full
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results
    finally:
        os.chdir(_orig_cwd)


def run_hierspeech(args):
    """HierSpeech++ 인퍼런스 (Hierarchical TTS, KO/EN zero-shot voice cloning)"""
    import torch, gc, numpy as np, soundfile as sf
    import torchaudio

    # torchaudio 2.10+ requires torchcodec → monkey-patch with soundfile
    def _sf_load(uri, frame_offset=0, num_frames=-1, **kw):
        data, sr = sf.read(uri, dtype='float32', always_2d=True)
        data = data.T  # (channels, samples)
        if frame_offset: data = data[:, frame_offset:]
        if num_frames > 0: data = data[:, :num_frames]
        return torch.from_numpy(data), sr
    torchaudio.load = _sf_load

    os.environ.setdefault('PHONEMIZER_ESPEAK_LIBRARY', r'C:\Program Files\eSpeak NG\libespeak-ng.dll')

    HIER_REPO = os.path.join(ROOT_DIR, 'engines', 'hierspeech', 'repo')
    MODEL_DIR = args.model_dir  # d:\tts-benchmark\models\hierspeechpp
    sys.path.insert(0, HIER_REPO)
    _orig_cwd = os.getcwd()
    os.chdir(HIER_REPO)

    try:
        import utils
        from Mels_preprocess import MelSpectrogramFixed
        from hierspeechpp_speechsynthesizer import SynthesizerTrn
        from ttv_v1.text import text_to_sequence
        from ttv_v1.t2w2v_transformer import SynthesizerTrn as Text2W2V
        from speechsr24k.speechsr import SynthesizerTrn as SpeechSR24
        from denoiser.generator import MPNet
        from denoiser.infer import denoise

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt_main     = os.path.join(MODEL_DIR, 'hierspeechpp_eng_kor', 'hierspeechpp_v1.1_ckpt.pth')
        ckpt_ttv      = os.path.join(MODEL_DIR, 'ttv_libritts_v1', 'ttv_lt960_ckpt.pth')
        ckpt_sr       = os.path.join(MODEL_DIR, 'speechsr24k', 'G_340000.pth')
        ckpt_denoiser = os.path.join(MODEL_DIR, 'denoiser', 'g_best')

        hps          = utils.get_hparams_from_file(os.path.join(MODEL_DIR, 'hierspeechpp_eng_kor', 'config.json'))
        hps_t2w2v    = utils.get_hparams_from_file(os.path.join(MODEL_DIR, 'ttv_libritts_v1', 'config.json'))
        hps_sr       = utils.get_hparams_from_file(os.path.join(MODEL_DIR, 'speechsr24k', 'config.json'))
        hps_denoiser = utils.get_hparams_from_file(os.path.join(MODEL_DIR, 'denoiser', 'config.json'))

        vram_before = get_vram_mb()
        t_load = time.perf_counter()

        mel_fn = MelSpectrogramFixed(
            sample_rate=hps.data.sampling_rate, n_fft=hps.data.filter_length,
            win_length=hps.data.win_length, hop_length=hps.data.hop_length,
            f_min=hps.data.mel_fmin, f_max=hps.data.mel_fmax,
            n_mels=hps.data.n_mel_channels, window_fn=torch.hann_window
        ).to(device)

        net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length, **hps.model).to(device)
        net_g.load_state_dict(torch.load(ckpt_main, map_location=device))
        net_g.eval()

        text2w2v = Text2W2V(hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length, **hps_t2w2v.model).to(device)
        text2w2v.load_state_dict(torch.load(ckpt_ttv, map_location=device))
        text2w2v.eval()

        speechsr = SpeechSR24(hps_sr.data.n_mel_channels,
            hps_sr.train.segment_size // hps_sr.data.hop_length, **hps_sr.model).to(device)
        utils.load_checkpoint(ckpt_sr, speechsr, None)
        speechsr.eval()

        denoiser_model = MPNet(hps_denoiser).to(device)
        _den_ckpt = torch.load(ckpt_denoiser, map_location=device)
        if 'generator' in _den_ckpt:
            denoiser_model.load_state_dict(_den_ckpt['generator'])
        else:
            utils.load_checkpoint(ckpt_denoiser, denoiser_model, None)
        denoiser_model.eval()

        load_time = time.perf_counter() - t_load
        vram_after_load = get_vram_mb()

        # Preprocess text (intersperse with blank token 0)
        text_seq = text_to_sequence(str(args.text), ["english_cleaners2"])
        interleaved = [0] + [x for t in text_seq for x in (t, 0)]
        token = torch.LongTensor(interleaved).unsqueeze(0).to(device)
        token_length = torch.LongTensor([token.size(-1)]).to(device)

        ref_wav, _ = get_reference_data(args, max_sec=10.0)

        # 표준 측정 규칙 v2: 참조 음성 파일 I/O는 루프 밖에서 1회만 수행
        _ref_audio, _ref_sr = torchaudio.load(ref_wav)
        _ref_audio = _ref_audio[:1, :]  # mono
        if _ref_sr != 16000:
            _ref_audio = torchaudio.functional.resample(_ref_audio, _ref_sr, 16000)

        results = []
        for run_idx in range(-WARMUP_RUNS, args.runs):
            is_warmup = run_idx < 0
            reset_vram_peak()
            t0 = time.perf_counter()

            audio = _ref_audio.clone()
            ori_len = audio.shape[-1]
            p = (ori_len // 1600 + 1) * 1600 - ori_len
            audio_padded = torch.nn.functional.pad(audio, (0, p)).to(device)

            with torch.no_grad():
                denoised = denoise(audio_padded.squeeze(0), denoiser_model, hps_denoiser)
            audio_cat = torch.cat([audio_padded, denoised[:, :audio_padded.shape[-1]]], dim=0)
            audio_cat = audio_cat[:, :ori_len]

            src_mel = mel_fn(audio_cat)
            src_length = torch.LongTensor([src_mel.size(2)]).to(device)
            src_length2 = torch.cat([src_length, src_length], dim=0)

            with torch.no_grad():
                w2v_x, pitch = text2w2v.infer_noise_control(
                    token, token_length, src_mel, src_length2,
                    noise_scale=0.333, denoise_ratio=0.8)
                src_length3 = torch.LongTensor([w2v_x.size(2)]).to(device)
                pitch[pitch < torch.log(torch.tensor([55.0]).to(device))] = 0
                converted = net_g.voice_conversion_noise_control(
                    w2v_x, src_length3, src_mel, src_length2, pitch,
                    noise_scale=0.333, denoise_ratio=0.8)
                converted = speechsr(converted)

            sync_gpu()
            inf_time = time.perf_counter() - t0

            if not is_warmup:
                audio_out = converted.squeeze().cpu().float().numpy()
                audio_out = audio_out / max(np.abs(audio_out).max(), 1e-8) * 0.999
                sr = 24000
                out_path = os.path.abspath(args.output_path.replace('.wav', f'_{run_idx}.wav'))
                sf.write(out_path, audio_out, sr, subtype='PCM_16')
                duration = len(audio_out) / sr
                results.append({
                    "run_index": run_idx,
                    "load_time_s": load_time,
                    "ttfa_ms": inf_time * 1000,
                    "inference_time_s": inf_time,
                    "audio_duration_s": duration,
                    "vram_before_mb": vram_before,
                    "vram_after_mb": vram_after_load,
                    "vram_peak_mb": get_vram_peak_mb(),
                    "sample_rate": sr,
                    "output_wav": out_path,
                    "success": True,
                })

        del net_g, text2w2v, speechsr, denoiser_model, mel_fn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results
    finally:
        os.chdir(_orig_cwd)


def run_llmvox(args):
    """LLMVoX 인퍼런스 (30M lightweight autoregressive)"""
    import torch, gc, numpy as np, soundfile as sf
    import sys
    import torch.nn.functional as F

    LLMVOX_REPO = r"D:\LLMVoX"
    sys.path.insert(0, LLMVOX_REPO)
    _orig_cwd = os.getcwd()
    os.chdir(LLMVOX_REPO)

    try:
        from inference.model_handler import ModelHandler
        from configs.inference_config import config as base_config
        
        config = base_config.copy()
        config["wav_config_path"] = os.path.join(LLMVOX_REPO, "WavTokenizer", "configs", "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
        config["wav_model_path"] = os.path.join(args.model_dir, "wavtokenizer_large_speech_320_24k.ckpt")
        config["llmvox_checkpoint_path"] = os.path.join(args.model_dir, "ckpt_english_tiny.pt")

        vram_before = get_vram_mb()
        t_load = time.perf_counter()
        handler = ModelHandler(config, device_id=0 if torch.cuda.is_available() else None)
        load_time = time.perf_counter() - t_load
        vram_after_load = get_vram_mb()

        results = []
        for run_idx in range(-WARMUP_RUNS, args.runs):
            is_warmup = run_idx < 0
            reset_vram_peak()
            t0 = time.perf_counter()
            
            # Simple AR generation loop
            text_tokens = handler.tokenizer(args.text, add_special_tokens=True)["input_ids"]
            # Add EOS token if not present
            if text_tokens[-1] != 385: text_tokens.append(385)
            
            text_tokens_tensor = torch.tensor(text_tokens).unsqueeze(0).to(handler.device)
            text_embeddings = handler.llm_model(text_tokens_tensor)
            
            speech_outputs = []
            current_speech_token = None
            speech_gen_index = 0
            kvcache = None
            eoa_token_id = config["eoa_token_id"]
            
            with torch.no_grad():
                # Process text tokens and generate speech tokens
                for i in range(text_embeddings.shape[1]):
                    if speech_gen_index == 0:
                        speech_embed = torch.zeros((1, 1, 512), device=handler.device)
                    else:
                        speech_token = torch.tensor([[current_speech_token]]).to(handler.device)
                        speech_embed = handler.wavtokenizer.codes_to_features(speech_token).permute(0, 2, 1).to(handler.device)
                    
                    text_embed = text_embeddings[:, i, :].unsqueeze(1)
                    speech_decoder_input = torch.cat([text_embed, speech_embed], dim=2)
                    speech_decoder_input = F.normalize(speech_decoder_input, p=2, dim=2, eps=1e-8)
                    
                    output, _, kvcache = handler.model(speech_decoder_input, kvcache=kvcache)
                    logits = output[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    current_speech_token = probs.argmax(dim=-1).item()
                    speech_outputs.append(current_speech_token)
                    speech_gen_index += 1
                    if current_speech_token == eoa_token_id: break

                # Continue generating until EOA or max length
                max_len = 1000
                while current_speech_token != eoa_token_id and len(speech_outputs) < max_len:
                    pad_token = torch.tensor([config["pad_token_id"]]).unsqueeze(0).to(handler.device)
                    text_embed = handler.llm_model(pad_token)
                    
                    speech_token = torch.tensor([[current_speech_token]]).to(handler.device)
                    speech_embed = handler.wavtokenizer.codes_to_features(speech_token).permute(0, 2, 1).to(handler.device)
                    
                    speech_decoder_input = torch.cat([text_embed, speech_embed], dim=2)
                    speech_decoder_input = F.normalize(speech_decoder_input, p=2, dim=2, eps=1e-8)
                    
                    output, _, kvcache = handler.model(speech_decoder_input, kvcache=kvcache)
                    logits = output[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    current_speech_token = probs.argmax(dim=-1).item()
                    speech_outputs.append(current_speech_token)
                    if current_speech_token == eoa_token_id: break

                # Convert tokens to audio
                if speech_outputs:
                    tokens_tensor = torch.tensor([speech_outputs]).to(handler.device)
                    features = handler.wavtokenizer.codes_to_features(tokens_tensor)
                    audio_out = handler.wavtokenizer.decode(features, bandwidth_id=torch.tensor([0]).to(handler.device)).squeeze(0)
                    audio = audio_out.cpu().float().numpy()
                else:
                    audio = np.zeros(24000)

            sync_gpu()
            inf_time = time.perf_counter() - t0
            ttfa_ms = inf_time * 1000

            if not is_warmup:
                sr = 24000
                out_path = os.path.abspath(args.output_path.replace('.wav', f'_{run_idx}.wav'))
                sf.write(out_path, audio, sr, subtype='PCM_16')
                duration = len(audio) / sr
                results.append({
                    "run_index": run_idx,
                    "load_time_s": load_time,
                    "ttfa_ms": ttfa_ms,
                    "inference_time_s": inf_time,
                    "audio_duration_s": duration,
                    "vram_before_mb": vram_before,
                    "vram_after_mb": vram_after_load,
                    "vram_peak_mb": get_vram_peak_mb(),
                    "sample_rate": sr,
                    "output_wav": out_path,
                    "success": True,
                })

        del handler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results
    finally:
        os.chdir(_orig_cwd)


def run_kani(args):
    """Kani-TTS 인퍼런스 (LFM2 기반, 다국어: KO/EN/ZH/DE/AR/ES)"""
    import torch, gc, numpy as np, soundfile as sf
    from kani_tts import KaniTTS, suppress_all_logs

    suppress_all_logs()
    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    model = KaniTTS(args.model_dir, device_map="auto", suppress_logs=True, show_info=False)
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        audio, _ = model(args.text)
        sync_gpu()
        inf_time = time.perf_counter() - t0

        if not is_warmup:
            sr = model.sample_rate
            audio = np.array(audio, dtype=np.float32)
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio /= max_val
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio, sr, subtype='PCM_16')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": inf_time * 1000,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_supertonic(args):
    """Supertonic v2 인퍼런스"""
    import torch, gc, numpy as np, soundfile as sf

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    from supertonic import SupertonicTTS
    model = SupertonicTTS.from_pretrained("supertone-inc/supertonic-v2")
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        
        # supertonic doesn't need ref audio for base testing
        wav = model.synthesize(args.text, language="ko")
        
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            if hasattr(wav, 'cpu'):
                audio = wav.squeeze().cpu().float().numpy()
            else:
                audio = np.array(wav).squeeze()
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio /= max_val
            sr = getattr(model, 'sample_rate', 44100) # fallback
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio, sr, subtype='PCM_16')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def run_chatterbox_multilingual(args):
    """Chatterbox TTS Multilingual 인퍼런스 (ResembleAI)"""
    import torch, gc, numpy as np, soundfile as sf

    vram_before = get_vram_mb()
    t_load = time.perf_counter()
    from chatterbox.tts import ChatterboxTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTTS.from_pretrained("resemble-ai/chatterbox-multilingual", device=device)
    load_time = time.perf_counter() - t_load
    vram_after_load = get_vram_mb()

    # 표준 측정 규칙 v2: 참조 음성 루프 밖에서 1회만 로드
    REF_WAV, _ = get_reference_data(args)

    results = []
    for run_idx in range(-WARMUP_RUNS, args.runs):
        is_warmup = run_idx < 0
        reset_vram_peak()
        t0 = time.perf_counter()
        
        wav = model.generate(
            text=args.text,
            audio_prompt_path=REF_WAV,
            language="ko"  # Add language param for Multilingual model
        )
        
        sync_gpu()
        inf_time = time.perf_counter() - t0
        ttfa_ms = inf_time * 1000

        if not is_warmup:
            if hasattr(wav, 'cpu'):
                audio = wav.squeeze().cpu().float().numpy()
            else:
                audio = np.array(wav).squeeze()
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio /= max_val
            sr = getattr(model, 'sr', 24000)
            out_path = args.output_path.replace('.wav', f'_{run_idx}.wav')
            sf.write(out_path, audio, sr, subtype='PCM_16')
            duration = len(audio) / sr
            results.append({
                "run_index": run_idx,
                "load_time_s": load_time,
                "ttfa_ms": ttfa_ms,
                "inference_time_s": inf_time,
                "audio_duration_s": duration,
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after_load,
                "vram_peak_mb": get_vram_peak_mb(),
                "sample_rate": sr,
                "output_wav": out_path,
                "success": True,
            })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


RUNNERS = {
    "cosyvoice2": run_cosyvoice2,
    "cosyvoice3": run_cosyvoice3,
    "fish_speech": run_fish_speech,
    "qwen3_tts": run_qwen3_tts,
    "qwen3_tts_int8": run_qwen3_tts_int8,
    "spark_tts": run_spark_tts,
    "glm_tts": run_glm_tts,
    "index_tts2": run_index_tts2,
    "bark": run_bark,
    "mio_tts": run_mio_tts,
    "gpt_sovits": run_gpt_sovits,
    "f5tts": run_f5tts,
    "xtts": run_xtts,
    "melotts": run_melotts,
    "chatterbox": run_chatterbox,
    "outetss": run_outetss,
    "kokoro": run_kokoro,
    "parler": run_parler,
    "styletts2": run_styletts2,
    "dia": run_dia,
    "openvoice": run_openvoice,
    "mars5": run_mars5,
    "zipvoice": run_zipvoice,
    "zipvoice_finetuned": run_zipvoice_finetuned,
    "chattts": run_chattts,
    "piper": run_piper,
    "voicecraftx": run_voicecraftx,
    "maskgct": run_maskgct,
    "hierspeech": run_hierspeech,
    "llmvox": run_llmvox,
    "kani": run_kani,
    "supertonic": run_supertonic,
    "chatterbox_multilingual": run_chatterbox_multilingual,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(RUNNERS.keys()))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--ref-key", default="iu_long", help="참조 오디오 키 (iu_long, kbs_short, male_docu 등)")
    parser.add_argument("--ref-audio", default=None, help="수동 지정 참조 오디오 경로 (ref-key 무시)")
    parser.add_argument("--ref-text", default=None, help="수동 지정 참조 대본 (ref-key 무시)")
    args = parser.parse_args()

    try:
        results = RUNNERS[args.model](args)
        print(json.dumps({"status": "ok", "results": results}, ensure_ascii=False))
    except Exception as e:
        import traceback
        print(json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()[-1000:],
        }, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
