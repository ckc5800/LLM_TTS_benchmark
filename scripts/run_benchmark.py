"""
TTS Benchmark Runner
사용법: python run_benchmark.py [--models all|cosyvoice2|...] [--runs 1]

각 모델은 자체 venv의 Python으로 서브프로세스 실행
→ 의존성 충돌 없이 격리 실행 가능
"""
import argparse
import os
import sys
import json
import subprocess
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(SCRIPTS_DIR)   # d:\tts-benchmark
INFRA_DIR = os.path.join(BENCH_DIR, "infra")
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, INFRA_DIR)

from benchmark.core import (
    BenchmarkLogger, BenchmarkResult, DEFAULT_TEST_TEXT, TEST_TEXTS, TEXT_SUITES,
)

# ─── 모델 레지스트리 ─────────────────────────────────────────────────────────
# python_exe: 각 모델이 사용할 venv의 Python 실행파일
# run_model_key: adapters/run_model.py의 --model 인자
# BENCH_DIR (d:\tts-benchmark) 기준 상대 경로로 설정
ROOT_DIR = BENCH_DIR   # engines/, models/ 등은 BENCH_DIR 하위

MODEL_REGISTRY = {
    "cosyvoice2": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "cosyvoice", ".venv", "Scripts", "python.exe"),
        "run_model_key": "cosyvoice2",
        "model_dir": os.path.join(ROOT_DIR, "models", "CosyVoice2-0.5B"),
        "model_name": "CosyVoice2-0.5B",
        "model_size_params": "500M",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "cross-lingual <|ko|>, campplus CPU",
        "extra_pythonpath": [os.path.join(ROOT_DIR, "engines", "cosyvoice")],
    },
    "cosyvoice3": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "cosyvoice", ".venv", "Scripts", "python.exe"),
        "run_model_key": "cosyvoice3",
        "model_dir": os.path.join(ROOT_DIR, "models", "CosyVoice3-0.5B"),
        "model_name": "CosyVoice3-0.5B",
        "model_size_params": "500M",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "zero-shot with <|ko|>",
        "extra_pythonpath": [os.path.join(ROOT_DIR, "engines", "cosyvoice")],
    },
    "fish_speech": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "fish_speech", ".venv", "Scripts", "python.exe"),
        "run_model_key": "fish_speech",
        "model_dir": os.path.join(ROOT_DIR, "models", "fish-speech-1.5"),
        "model_name": "Fish-Speech-1.5",
        "model_size_params": "~500M",
        "korean_support": False,
        "supported_languages": ["en", "ja", "zh"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "FireflyGAN 보코더, 다국어",
    },
    "qwen3_tts_0.6b": {
        "python_exe": r"D:\list\Scripts\python.exe", # Not in benchmark dir
        "run_model_key": "qwen3_tts",
        "model_dir": os.path.join(ROOT_DIR, "models", "Qwen3-TTS-0.6B"),
        "model_name": "Qwen3-TTS-0.6B",
        "model_size_params": "0.6B",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "10개 언어 지원, qwen_tts 패키지",
    },
    "qwen3_tts_1.7b": {
        "python_exe": r"D:\list\Scripts\python.exe", # Not in benchmark dir
        "run_model_key": "qwen3_tts",
        "model_dir": os.path.join(ROOT_DIR, "models", "Qwen3-TTS-1.7B"),
        "model_name": "Qwen3-TTS-1.7B",
        "model_size_params": "1.7B",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "10개 언어 지원, 고품질",
    },
    "qwen3_tts_0.6b_int8": {
        "python_exe": r"D:\list\Scripts\python.exe",
        "run_model_key": "qwen3_tts_int8",
        "model_dir": os.path.join(ROOT_DIR, "models", "Qwen3-TTS-0.6B"),
        "model_name": "Qwen3-TTS-0.6B-INT8",
        "model_size_params": "0.6B",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "INT8 quantization (bitsandbytes)",
    },
    "qwen3_tts_1.7b_int8": {
        "python_exe": r"D:\list\Scripts\python.exe",
        "run_model_key": "qwen3_tts_int8",
        "model_dir": os.path.join(ROOT_DIR, "models", "Qwen3-TTS-1.7B"),
        "model_name": "Qwen3-TTS-1.7B-INT8",
        "model_size_params": "1.7B",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "INT8 quantization (bitsandbytes)",
    },
    "orpheus": {
        "python_exe": r"D:\list\Scripts\python.exe", # Not in benchmark dir
        "run_model_key": "orpheus",
        "model_dir": os.path.join(ROOT_DIR, "models", "orpheus-3b"),
        "model_name": "Orpheus-TTS-3B",
        "model_size_params": "3B",
        "korean_support": False,
        "supported_languages": ["en"],
        "is_llm_based": True,
        "status": "skip",
        "notes": "gated 모델 - 접근 권한 없음",
    },
    "spark_tts": {
        "python_exe": r"D:\list\Scripts\python.exe", # Not in benchmark dir
        "run_model_key": "spark_tts",
        "model_dir": os.path.join(ROOT_DIR, "models", "Spark-TTS-0.5B"),
        "model_name": "Spark-TTS-0.5B",
        "model_size_params": "500M",
        "korean_support": False,
        "supported_languages": ["en", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "중/영 2개 언어, controllable TTS",
    },
    "index_tts2": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "indextts", ".venv", "Scripts", "python.exe"),
        "run_model_key": "index_tts2",
        "model_dir": os.path.join(ROOT_DIR, "models", "IndexTTS-2"),
        "model_name": "IndexTTS-2",
        "model_size_params": "~1B",
        "korean_support": False,
        "supported_languages": ["en", "ja", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "timeout_s": 3600,
        "notes": "중/영/일/스페인어, Qwen GPT + BigVGAN",
    },
    "mio_tts_1.7b": {
        "python_exe": r"D:\list\Scripts\python.exe", # Not in benchmark dir
        "run_model_key": "mio_tts",
        "model_dir": os.path.join(ROOT_DIR, "models", "MioTTS-1.7B"),
        "model_name": "MioTTS-1.7B",
        "model_size_params": "1.7B",
        "korean_support": False,
        "supported_languages": ["en", "ja"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "영/일 지원, MioCodec, Qwen3 기반",
    },
    "mio_tts_2.6b": {
        "python_exe": r"D:\list\Scripts\python.exe", # Not in benchmark dir
        "run_model_key": "mio_tts",
        "model_dir": os.path.join(ROOT_DIR, "models", "MioTTS-2.6B"),
        "model_name": "MioTTS-2.6B",
        "model_size_params": "2.6B",
        "korean_support": False,
        "supported_languages": ["en", "ja"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "영/일 지원, MioCodec, LFM2 기반",
    },
    "chattts": {
        "python_exe": r"D:\list\Scripts\python.exe",
        "run_model_key": "chattts",
        "model_dir": os.path.join(ROOT_DIR, "models", "ChatTTS"),
        "model_name": "ChatTTS",
        "model_size_params": "~200M",
        "korean_support": False,
        "supported_languages": ["en", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "대화체 특화, 2noise 기반",
    },
    "piper_ko": {
        "python_exe": r"D:\list\Scripts\python.exe",
        "run_model_key": "piper",
        "model_dir": os.path.join(ROOT_DIR, "models", "piper-ko"),
        "model_name": "Piper-KO",
        "model_size_params": "28M",
        "korean_support": True,
        "supported_languages": ["ko"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "ONNX 기반 초고속 CPU 실행",
    },
    "voicecraftx": {
        "python_exe": r"C:\ProgramData\anaconda3\envs\voicecraftx\python.exe",
        "run_model_key": "voicecraftx",
        "model_dir": os.path.join(ROOT_DIR, "models", "VoiceCraft-X"),
        "model_name": "VoiceCraft-X",
        "model_size_params": "~500M",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh", "es", "fr", "de", "nl", "it", "pt", "pl"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "11개 언어, Zero-shot voice cloning",
    },
    "maskgct": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "maskgct", ".venv", "Scripts", "python.exe"),
        "run_model_key": "maskgct",
        "model_dir": os.path.join(ROOT_DIR, "models", "Amphion"),
        "model_name": "MaskGCT",
        "model_size_params": "~500M",
        "korean_support": True,
        "supported_languages": ["ko", "en", "zh"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "Fast non-autoregressive TTS",
    },
    "hierspeech": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "hierspeech", ".venv", "Scripts", "python.exe"),
        "run_model_key": "hierspeech",
        "model_dir": os.path.join(ROOT_DIR, "models", "hierspeechpp"),
        "model_name": "Hierspeech++",
        "model_size_params": "~200M",
        "korean_support": True,
        "supported_languages": ["ko", "en"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "Hierarchical Speech Synthesis",
    },
    "kani": {
        "python_exe": r"C:\ProgramData\anaconda3\envs\kani\python.exe",
        "run_model_key": "kani",
        "model_dir": "nineninesix/kani-tts-370m",
        "model_name": "Kani-TTS-370M",
        "model_size_params": "370M",
        "korean_support": True,
        "supported_languages": ["ko", "en", "zh", "de", "ar", "es"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "LFM2 기반, 다국어 TTS (NanoCodec)",
    },
    "llmvox": {
        "python_exe": r"D:\list\Scripts\python.exe",
        "run_model_key": "llmvox",
        "model_dir": os.path.join(ROOT_DIR, "models", "LLMVoX"),
        "model_name": "LLMVoX",
        "model_size_params": "30M",
        "korean_support": False,
        "supported_languages": ["en"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "Lightweight LLM-addon TTS",
    },
    "glm_tts": {
        "python_exe": r"D:\list\Scripts\python.exe", # Not in benchmark dir
        "run_model_key": "glm_tts",
        "model_dir": os.path.join(ROOT_DIR, "models", "GLM-TTS"),
        "model_name": "GLM-TTS",
        "model_size_params": "~1.5B",
        "korean_support": False,
        "supported_languages": ["en", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "중/영 only, RL 기반",
    },
    "bark": {
        "python_exe": r"D:\list\Scripts\python.exe", # Not in benchmark dir
        "run_model_key": "bark",
        "model_dir": os.path.join(ROOT_DIR, "models", "bark"),
        "model_name": "Bark",
        "model_size_params": "~900M",
        "korean_support": False,
        "supported_languages": ["en"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "느림, 한국어 제한적",
    },
    "gpt_sovits": {
        "python_exe": r"D:\list\Scripts\python.exe", # Not in benchmark dir
        "run_model_key": "gpt_sovits",
        "model_dir": os.path.join(ROOT_DIR, "models", "GPT-SoVITS"),
        "model_name": "GPT-SoVITS-V3",
        "model_size_params": "~200M",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "V3 (BigVGAN vocoder), 한국어 지원",
    },
    "gpt_sovits_v4": {
        "python_exe": r"D:\list\Scripts\python.exe",
        "run_model_key": "gpt_sovits_v4",
        "model_dir": os.path.join(ROOT_DIR, "models", "GPT-SoVITS"),
        "model_name": "GPT-SoVITS-V4",
        "model_size_params": "~200M",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "V4 (HiFiGAN vocoder, 48kHz), 한국어 지원",
    },
    "f5tts": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "f5tts", ".venv", "Scripts", "python.exe"),
        "run_model_key": "f5tts",
        "model_dir": r"SWivid/F5-TTS",
        "model_name": "F5-TTS",
        "model_size_params": "~335M",
        "korean_support": False,
        "supported_languages": ["en", "zh"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "flow matching, zero-shot, EN+ZH 학습",
    },
    "xtts": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "xtts", ".venv", "Scripts", "python.exe"),
        "run_model_key": "xtts",
        "model_dir": r"tts_models/multilingual/multi-dataset/xtts_v2",
        "model_name": "XTTS-v2",
        "model_size_params": "~1.8B",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "17개 언어, 한국어 공식 지원, Coqui",
    },
    "melotts": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "melo", ".venv", "Scripts", "python.exe"),
        "run_model_key": "melotts",
        "model_dir": r"myshell-ai/MeloTTS-Korean",
        "model_name": "MeloTTS-KO",
        "model_size_params": "~50M",
        "korean_support": True,
        "supported_languages": ["ko", "en"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "VITS 기반, 한국어 전용, CPU 실시간",
    },
    "chatterbox": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "chatterbox", ".venv", "Scripts", "python.exe"),
        "run_model_key": "chatterbox",
        "model_dir": r"ResembleAI/chatterbox",
        "model_name": "Chatterbox-TTS",
        "model_size_params": "~500M",
        "korean_support": False,
        "supported_languages": ["en"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "zero-shot 음성 클론, EN 위주, ResembleAI",
    },
    "outetss": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "outetss", ".venv", "Scripts", "python.exe"),
        "run_model_key": "outetss",
        "model_dir": r"OuteAI/OuteTTS-1.0-0.6B",
        "model_name": "OuteTTS-1.0-0.6B",
        "model_size_params": "0.6B",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "LLM 기반 TTS, 9개 언어, zero-shot",
    },
    "kokoro": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "kokoro", ".venv", "Scripts", "python.exe"),
        "run_model_key": "kokoro",
        "model_dir": r"hexgrad/Kokoro-82M",
        "model_name": "Kokoro-82M",
        "model_size_params": "82M",
        "korean_support": False,
        "supported_languages": ["en", "ja", "zh"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "초경량 VITS, EN/ZH/JP/ES/FR 지원, CPU 실시간 이상",
    },
    "parler": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "parler", ".venv", "Scripts", "python.exe"),
        "run_model_key": "parler",
        "model_dir": r"parler-tts/parler-tts-mini-v1",
        "model_name": "Parler-TTS-mini",
        "model_size_params": "~880M",
        "korean_support": False,
        "supported_languages": ["en"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "description 기반 음성 제어, EN 전용, Huggingface",
    },
    "styletts2": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "styletts2", ".venv", "Scripts", "python.exe"),
        "run_model_key": "styletts2",
        "model_dir": r"yl4579/StyleTTS2-LibriTTS",
        "model_name": "StyleTTS2",
        "model_size_params": "~50M",
        "korean_support": False,
        "supported_languages": ["en"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "EN 고품질, diffusion style, 2단계 추론",
    },
    "dia": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "dia", ".venv", "Scripts", "python.exe"),
        "run_model_key": "dia",
        "model_dir": r"nari-labs/Dia-1.6B",
        "model_name": "Dia-1.6B",
        "model_size_params": "1.6B",
        "korean_support": False,
        "supported_languages": ["en"],
        "is_llm_based": True,
        "status": "ready",
        "timeout_s": 3600,
        "notes": "대화형 TTS, EN 위주, Nari Labs",
    },
    "openvoice": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "openvoice", ".venv", "Scripts", "python.exe"),
        "run_model_key": "openvoice",
        "model_dir": os.path.join(ROOT_DIR, "engines", "openvoice", "checkpoints_v2"),
        "model_name": "OpenVoice-v2",
        "model_size_params": "~200M",
        "korean_support": True,
        "supported_languages": ["ko", "en"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "한국어 공식 지원, 음성 클론, MeloTTS 기반",
    },
    "mars5": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "mars5", ".venv", "Scripts", "python.exe"),
        "run_model_key": "mars5",
        "model_dir": r"camb-ai/mars5-tts",
        "model_name": "MARS5-TTS",
        "model_size_params": "~700M",
        "korean_support": False,
        "supported_languages": ["en"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "영어 전용 zero-shot 음성 클론, shallow/deep clone 모드",
        "timeout_s": 3600,
    },
    "zipvoice_finetuned": {
        "python_exe": r"D:\list\Scripts\python.exe", # Not in benchmark dir
        "run_model_key": "zipvoice_finetuned",
        "model_dir": r"D:\aicess-tts-engine-refactor-tts-engine-v1\models\zipvoice\v0.0.3",
        "model_name": "ZipVoice-FT",
        "model_size_params": "123M",
        "korean_support": True,
        "supported_languages": ["ko"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "ZipVoice Flow Matching, KO 파인튜닝, REST API (port 9102), Vocos 24kHz",
        "timeout_s": 600,
    },
    "zipvoice": {
        "python_exe": r"D:\list\Scripts\python.exe",
        "run_model_key": "zipvoice",
        "model_dir": os.path.join(ROOT_DIR, "models", "ZipVoice"),
        "model_name": "ZipVoice-Official",
        "model_size_params": "123M",
        "korean_support": False,
        "supported_languages": ["en", "zh"],
        "is_llm_based": False,
        "status": "ready",
        "notes": "Official ZipVoice (EN/ZH), Flow Matching, zero-shot",
        "timeout_s": 600,
    },
    "supertonic": {
        "python_exe": r"D:\list\Scripts\python.exe",
        "run_model_key": "supertonic",
        "model_dir": "",
        "model_name": "Supertonic-v2",
        "model_size_params": "?",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "Supertonic v2",
    },
    "chatterbox_multilingual": {
        "python_exe": os.path.join(ROOT_DIR, "engines", "chatterbox", ".venv", "Scripts", "python.exe"),
        "run_model_key": "chatterbox_multilingual",
        "model_dir": r"resemble-ai/chatterbox-multilingual",
        "model_name": "Chatterbox-ML",
        "model_size_params": "~500M",
        "korean_support": True,
        "supported_languages": ["ko", "en", "ja", "zh"],
        "is_llm_based": True,
        "status": "ready",
        "notes": "Multilingual zero-shot 음성 클론",
    },
}

RUN_MODEL_SCRIPT = os.path.join(INFRA_DIR, "adapters", "run_model.py")
import glob
from statistics import mean, stdev as _stdev


def run_model_subprocess(
    info: dict,
    test_text: str,
    output_dir: str,
    runs: int,
    ref_key: str = "iu_long",
    timeout_s: int = 1800,
    output_path: str = None,
) -> list[dict]:
    """모델 전용 Python으로 서브프로세스 실행 후 JSON 결과 파싱"""
    python_exe = info["python_exe"]
    if not os.path.exists(python_exe):
        raise FileNotFoundError(f"Python not found: {python_exe}\n환경 설정 필요")

    out_path = output_path if output_path else os.path.join(output_dir, f"{info['model_name']}.wav")
    cmd = [
        python_exe,
        RUN_MODEL_SCRIPT,
        "--model", info["run_model_key"],
        "--model-dir", info["model_dir"],
        "--text", test_text,
        "--output-path", out_path,
        "--runs", str(runs),
        "--ref-key", ref_key,
    ]

    print(f"  실행: {python_exe.split(os.sep)[-3]}/{info['run_model_key']}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONNOUSERSITE"] = "1"  # Windows user site-packages (Python 3.13 등) 오염 방지
    # 모델별 추가 Python 경로 주입 (cosyvoice, glm_tts 등 repo-level 임포트)
    extra_paths = info.get("extra_pythonpath", [])
    if extra_paths:
        existing = env.get("PYTHONPATH", "")
        merged = os.pathsep.join(extra_paths + ([existing] if existing else []))
        env["PYTHONPATH"] = merged
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        cwd=BENCH_DIR,
        env=env,
    )
    try:
        out_bytes, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise RuntimeError(f"Subprocess timeout after {timeout_s}s")

    # stdout에서 마지막 JSON 라인 파싱
    stdout = out_bytes.decode('utf-8', errors='replace').strip()
    last_json_line = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            last_json_line = line

    if last_json_line is None:
        raise RuntimeError(f"No JSON output from subprocess.\nstdout:\n{stdout[-500:]}")

    data = json.loads(last_json_line)
    if data["status"] == "error":
        raise RuntimeError(f"Subprocess error: {data.get('error')}\n{data.get('traceback', '')}")

    return data["results"]


def get_output_dir(model_key: str, results_dir: str) -> str:
    wav_dir = os.path.join(results_dir, "wav", model_key)
    os.makedirs(wav_dir, exist_ok=True)
    return wav_dir


def _load_cached_result(text_results_dir: str, model_key: str) -> dict | None:
    """이미 완료된 text_key 디렉토리에서 해당 모델의 결과 dict 로드."""
    info = MODEL_REGISTRY.get(model_key, {})
    target_name = info.get("model_name", "")
    json_files = sorted(glob.glob(os.path.join(text_results_dir, "*_results.json")))
    for jf in reversed(json_files):
        try:
            with open(jf, encoding="utf-8") as f:
                results = json.load(f)
            for r in results:
                if r.get("model_name") == target_name and r.get("success") and r.get("rtf", -1) >= 0:
                    return r
        except Exception:
            pass
    return None


def compute_and_save_averages(results_dir: str, all_results: dict, text_keys: list):
    """모델별 텍스트 평균 RTF·VRAM 계산 → {results_dir}/averages.json 저장."""
    averages = {}
    for model_key, text_results in sorted(all_results.items()):
        info = MODEL_REGISTRY.get(model_key, {})
        rtfs, infers, durations, vrams = [], [], [], []
        for text_key in text_keys:
            r = text_results.get(text_key)
            if r and r.get("success") and r.get("rtf", -1) >= 0:
                rtfs.append(r["rtf"])
                infers.append(r.get("inference_time_s", 0))
                durations.append(r.get("audio_duration_s", 0))
                if r.get("vram_peak_mb", -1) >= 0:
                    vrams.append(r["vram_peak_mb"])
        if not rtfs:
            continue
        averages[model_key] = {
            "model_name": info.get("model_name", model_key),
            "model_size_params": info.get("model_size_params", ""),
            "korean_support": info.get("korean_support", True),
            "n_texts": len(rtfs),
            "avg_rtf": round(mean(rtfs), 4),
            "std_rtf": round(_stdev(rtfs), 4) if len(rtfs) > 1 else 0.0,
            "avg_inference_time_s": round(mean(infers), 3),
            "avg_audio_duration_s": round(mean(durations), 3),
            "avg_vram_peak_mb": round(mean(vrams), 1) if vrams else -1,
            "per_text": {
                tk: {
                    "rtf": text_results[tk].get("rtf", -1),
                    "inference_time_s": text_results[tk].get("inference_time_s", -1),
                    "audio_duration_s": text_results[tk].get("audio_duration_s", -1),
                    "success": text_results[tk].get("success", False),
                }
                for tk in text_keys if tk in text_results
            },
        }

    out_path = os.path.join(results_dir, "averages.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(averages, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  평균 결과 저장: {out_path}")
    print(f"  모델별 평균 RTF (낮을수록 빠름):")
    for mk, v in sorted(averages.items(), key=lambda x: x[1]["avg_rtf"]):
        bar = "█" * min(20, int(v["avg_rtf"] * 5))
        print(f"  {mk:22s}  avg={v['avg_rtf']:.3f} ±{v['std_rtf']:.3f}  "
              f"({v['n_texts']}texts)  {bar}")
    print(f"{'='*60}")


def run_text_suite(args, model_keys: list):
    """--text-suite 모드: 여러 텍스트로 전체 모델 실행 후 평균 계산."""
    suites = ["ko", "en"] if args.text_suite == "all" else [args.text_suite]
    text_keys = []
    for s in suites:
        text_keys.extend(TEXT_SUITES[s])

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TTS Multi-Text Benchmark  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print(f"스위트: {args.text_suite}  |  텍스트 {len(text_keys)}개  |  모델 {len(model_keys)}개")
    print(f"결과 디렉토리: {args.results_dir}")
    print(f"{'='*60}\n")

    all_results: dict[str, dict[str, dict]] = {}   # {model_key: {text_key: result_dict}}

    for text_key in text_keys:
        text = TEST_TEXTS[text_key]
        text_results_dir = os.path.join(args.results_dir, text_key)

        print(f"\n{'═'*60}")
        print(f"  [{text_key}]  \"{text[:70]}\"")
        print(f"{'═'*60}")

        logger = BenchmarkLogger(text_results_dir)

        for model_key in model_keys:
            if model_key not in MODEL_REGISTRY:
                continue
            info = MODEL_REGISTRY[model_key]
            if info["status"] != "ready":
                continue

            # KO suite → 한국어 지원 모델만
            if text_key.startswith("ko") and not info.get("korean_support", True):
                continue

            output_dir = get_output_dir(model_key, text_results_dir)
            wav_check = os.path.join(output_dir, f"{info['model_name']}_0.wav")

            # Resume: WAV 이미 존재하면 스킵
            if os.path.exists(wav_check) and not getattr(args, "force", False):
                print(f"  SKIP [{text_key}] {model_key} (완료됨)")
                cached = _load_cached_result(text_results_dir, model_key)
                if cached:
                    all_results.setdefault(model_key, {})[text_key] = cached
                continue

            print(f"\n  {'─'*46}")
            print(f"  {info['model_name']}  ({info['model_size_params']})")

            try:
                raw_results = run_model_subprocess(
                    info, text, output_dir, runs=1,
                    ref_key=getattr(args, "ref_key", "iu_long"),
                    timeout_s=info.get("timeout_s", 1800),
                )
                for r in raw_results:
                    result = BenchmarkResult(
                        model_name=info["model_name"],
                        model_size_params=info["model_size_params"],
                        korean_support=info["korean_support"],
                        test_text=text,
                        notes=info.get("notes", ""),
                        **{k: v for k, v in r.items()
                           if k in BenchmarkResult.__dataclass_fields__},
                    )
                    logger.add(result)
                    if r.get("success", False):
                        all_results.setdefault(model_key, {})[text_key] = result.to_dict()
            except Exception as e:
                print(f"  [ERROR] {model_key}/{text_key}: {e}")
                err_result = BenchmarkResult(
                    model_name=info["model_name"],
                    model_size_params=info["model_size_params"],
                    korean_support=info["korean_support"],
                    test_text=text,
                    notes=info.get("notes", ""),
                    success=False,
                    error=str(e)[:300],
                )
                logger.add(err_result)

        logger.finalize()

    compute_and_save_averages(args.results_dir, all_results, text_keys)


def main():
    parser = argparse.ArgumentParser(description="TTS Benchmark Runner")
    parser.add_argument(
        "--models", nargs="+", default=["cosyvoice2"],
        help="모델 목록 또는 'all' 또는 'ready'",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--text", choices=list(TEST_TEXTS.keys()), default="ko_medium")
    parser.add_argument("--text-suite", choices=["ko", "en", "all"], default=None,
                        help="10개 텍스트 스위트 전체 실행 (ko/en/all). --text 무시됨.")
    parser.add_argument("--force", action="store_true",
                        help="이미 완료된(WAV 존재) 결과도 재실행")
    parser.add_argument("--results-dir", default=r"D:\tts-benchmark\results")
    parser.add_argument("--ref-key", default="iu_long", 
                        help="참조 오디오 키 (iu_long, kbs_short, male_docu 등)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'모델':20s} {'상태':8s} {'한국어':6s} {'Python venv'}")
        print("-" * 70)
        for key, info in MODEL_REGISTRY.items():
            ko = "O" if info["korean_support"] else "X"
            try:
                venv = info["python_exe"].split(os.sep)[-3]
            except Exception:
                venv = "unknown"
            print(f"{key:20s} {info['status']:8s} {ko:6s}  {venv}")
        return

    if args.models == ["all"]:
        model_keys = list(MODEL_REGISTRY.keys())
    elif args.models == ["ready"]:
        model_keys = [k for k, v in MODEL_REGISTRY.items() if v["status"] == "ready"]
    else:
        model_keys = args.models

    # ── 멀티 텍스트 스위트 모드 ──────────────────────────────────────────────────
    if args.text_suite:
        run_text_suite(args, model_keys)
        return

    test_text = TEST_TEXTS[args.text]
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TTS Benchmark  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print(f"모델: {model_keys}")
    print(f"반복: {args.runs}회  |  텍스트: {args.text}")
    print(f"텍스트: \"{test_text[:60]}...\"")
    print(f"{'='*60}\n")

    logger = BenchmarkLogger(args.results_dir)

    for model_key in model_keys:
        if model_key not in MODEL_REGISTRY:
            print(f"[SKIP] 알 수 없는 모델: {model_key}")
            continue

        info = MODEL_REGISTRY[model_key]
        if info["status"] != "ready":
            print(f"[SKIP] {model_key} ({info['status']})")
            continue

        print(f"\n{'─'*50}")
        print(f"  {info['model_name']}  ({info['model_size_params']})")
        print(f"{'─'*50}")

        output_dir = get_output_dir(model_key, args.results_dir)
        try:
            raw_results = run_model_subprocess(info, test_text, output_dir, args.runs,
                                               ref_key=args.ref_key,
                                               timeout_s=info.get("timeout_s", 1800))
            for r in raw_results:
                result = BenchmarkResult(
                    model_name=info["model_name"],
                    model_size_params=info["model_size_params"],
                    korean_support=info["korean_support"],
                    test_text=test_text,
                    notes=info.get("notes", ""),
                    **{k: v for k, v in r.items() if k in BenchmarkResult.__dataclass_fields__},
                )
                logger.add(result)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            r = BenchmarkResult(
                model_name=info["model_name"],
                model_size_params=info["model_size_params"],
                korean_support=info["korean_support"],
                test_text=test_text,
                notes=info.get("notes", ""),
                success=False,
                error=str(e)[:300],
            )
            logger.add(r)

    logger.finalize()


if __name__ == "__main__":
    main()
