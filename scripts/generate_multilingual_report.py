"""
다국어 TTS 벤치마크 HTML 보고서 생성기
사용법: python generate_multilingual_report.py --results-dir results_multilingual
"""
import os, sys, json, argparse
from datetime import datetime

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(SCRIPTS_DIR)   # d:\tts-benchmark
INFRA_DIR = os.path.join(BENCH_DIR, "infra")
sys.path.insert(0, BENCH_DIR)
sys.path.insert(0, INFRA_DIR)

try:
    from benchmark.core import TEST_TEXTS
except ImportError:
    TEST_TEXTS = {}

REF_DIR = os.path.join(INFRA_DIR, "references")
LANG_REF_KEYS = {
    "ko": "iu_long",
    "en": "en_female",
    "ja": "ja_female",
    "zh": "zh_female",
}
LANG_REF_WAV = {
    lang: os.path.join(REF_DIR, f"{ref_key}.wav")
    for lang, ref_key in LANG_REF_KEYS.items()
}

PIVOT_CONFIG = {
    "ko": {"tk": "ko_medium", "rk": "iu_long"},
    "en": {"tk": "en_medium", "rk": "en_female"},
    "ja": {"tk": "ja_medium", "rk": "ja_female"},
    "zh": {"tk": "zh_medium", "rk": "zh_female"},
}

EXCLUDED_MODELS = {
    "라이선스 문제 (비상업 전용)": [
        {"model": "Fish-Speech",    "org": "Fish Audio",   "lic": "CC BY-NC-SA 4.0", "reason": "영리 목적 사용 불가"},
        {"model": "F5-TTS",         "org": "SWivid",       "lic": "코드 MIT / 가중치 CC BY-NC 4.0", "reason": "모델 가중치가 Emilia 학습 데이터 라이선스(CC BY-NC) 상속 — 상업 배포 불가"},
        {"model": "ChatTTS",        "org": "2Noise",       "lic": "CC BY-NC 4.0",    "reason": "비상업 용도 및 한국어 미지원"},
        {"model": "SeamlessM4T v2", "org": "Meta",         "lic": "CC BY-NC",        "reason": "상업적 활용 불가 라이선스"},
        {"model": "XTTS-v2",        "org": "Coqui",        "lic": "CPML",            "reason": "비상업 용도로만 공개 (공식 지원 중단)"},
        {"model": "Fish Audio S2 Pro", "org": "Fish Audio", "lic": "Research License (Apache 2.0 병기 혼재)",  "reason": "HF 기본 Research License(비상업), Apache 2.0 병기되나 상업 시 별도 계약 필요 + Linux only / 24GB VRAM 제약"},
        {"model": "VoiceCraft-X",   "org": "UT Austin",    "lic": "CC BY-NC-SA 4.0", "reason": "비상업 전용 + RTF 1.2+ 실시간 불가"},
        {"model": "MARS5-TTS",      "org": "Camb-AI",      "lic": "CC BY-NC-SA 4.0", "reason": "비상업 전용 + RTF ~3.3 느림"},
        {"model": "IndexTTS-2",     "org": "Bilibili",     "lic": "CC BY-NC-SA 4.0", "reason": "비상업 전용 + 한국어 미지원"},
    ],
    "한국어 미지원 (OOV 문제)": [
        {"model": "Spark-TTS",       "org": "SparkAudio",   "lic": "Apache 2.0",  "reason": "ZH/EN만 지원, 한국어 어휘 없음"},
        {"model": "Dia",             "org": "Nari Labs",    "lic": "Apache 2.0",  "reason": "영어 전용 대화형 모델"},
        {"model": "StyleTTS2 (S2)",  "org": "yl4579",       "lic": "MIT",         "reason": "영어 전용 합성 모델"},
        {"model": "Parler-TTS-mini", "org": "HuggingFace",  "lic": "Apache 2.0",  "reason": "영어 전용 합성 모델"},
        {"model": "Kokoro",          "org": "hexgrad",      "lic": "Apache 2.0",  "reason": "다국어 지원하나 한국어 미포함"},
        {"model": "Chatterbox-TTS",  "org": "Resemble AI",  "lic": "MIT",         "reason": "기본 모델(500M) 영어 전용 — 한국어 포함 멀티링구얼은 Chatterbox-ML로 별도 공개"},
        {"model": "LLMVoX",          "org": "MBZUAI",       "lic": "MIT",         "reason": "영어 전용 단일 모델"},
    ],
    "기술적 제약 (G2P 필수 등)": [
        {"model": "Piper-TTS", "org": "Home Assistant", "lic": "MIT",      "reason": "phoneme 변환(G2P) 전처리 필수 (End-to-End 아님)"},
        {"model": "MMS-TTS",   "org": "Meta",           "lic": "CC BY-NC", "reason": "uroman 로마자 변환 필수 및 비상업"},
        {"model": "VITS2",     "org": "jaywalnut310",   "lic": "불명확",   "reason": "논문·비공식 구현만 존재, 공식 가중치/라이선스 불명확, G2P 의존도 있음 — 상업·실시간 KO TTS 비교 대상으로 가치 낮음"},
    ],
    "속도 기준 미달 (RTF > 1.0)": [
        {"model": "Kani-TTS", "org": "Kani Team", "lic": "Apache 2.0", "reason": "KO RTF 4.9 (심각하게 느림)"},
        {"model": "OuteTTS",  "org": "OuteAI",    "lic": "Apache 2.0", "reason": "RTF 3.3 내외로 실시간 활용 불가"},
        {"model": "Bark",     "org": "Suno AI",   "lic": "MIT",        "reason": "한국어 공식 지원하나 Transformer 3단계 생성(semantic→coarse→fine) 구조로 RTF 극도로 높음 — 실시간 서비스 불가"},
    ],
    "가중치 비공개 (Closed)": [
        {"model": "Seed-TTS",           "org": "ByteDance",  "lic": "비공개", "reason": "가중치·모델카드 미공개, 블로그만 존재, 상업 불가 — 공식 수치·라이선스 기반 조사 대상 아님"},
        {"model": "NaturalSpeech 1/2/3","org": "Microsoft",  "lic": "비공개", "reason": "가중치 미공개"},
        {"model": "VALL-E / BASE TTS",  "org": "Microsoft",  "lic": "비공개", "reason": "코드/가중치 비공개"},
        {"model": "E2-TTS",             "org": "Microsoft",   "lic": "비공개", "reason": "논문(Embarrassingly Easy TTS)만 공개, 공식 가중치 미공개"},
    ],
    "TTS 범위 외 (음악·효과음 등)": [
        {"model": "Stable Audio Open 1.0", "org": "Stability AI", "lic": "Stability AI Community", "reason": "음악·효과음 생성 전용 오디오 모델, 한국어 TTS 범위 밖"},
    ],
    "중복 또는 구조적 비호환": [
        {"model": "Chatterbox-ML",    "org": "Resemble AI", "lic": "MIT",              "reason": "기존 Chatterbox-TTS(동일 아키텍처) 결과 중복 — 멀티링구얼 버전은 언어 확장판이며 한국어 품질이 V1 대비 퇴보했다는 사용자 보고 존재"},
        {"model": "Supertonic-v2",    "org": "Supertone",   "lic": "MIT+OpenRAIL-M",   "reason": "레퍼런스 오디오 기반 음성 복제 미지원 (preset 화자 고정) — 동일 레퍼런스로 공정 비교하는 벤치마크 프레임워크와 구조적으로 비호환"},
        {"model": "Higgs Audio V2.5", "org": "Boson AI",    "lic": "Custom",           "reason": "V2.5(1B) 가중치 미공개(API 전용) — V2(3B)는 VRAM 24GB 권장으로 RTX 5080 16GB 단독 운용 불가"},
    ],
    "테스트 진행 예정 (후보)": [
        {"model": "GPT-SoVITS V4",    "org": "RVC-Boss",    "lic": "MIT",              "reason": "V3 측정 완료 — V4는 48kHz 출력·metallic 아티팩트 제거 개선판. 별도 환경 구성 후 추가 측정 예정"},
    ]
}

# ─── 모델 설명 (장단점, 아키텍처) ─────────────────────────────────────────────
MODEL_INFO = {
    "cosyvoice2": {
        "display_name": "CosyVoice2",
        "developer": "Alibaba DAMO",
        "arch_type": "LLM",
        "arch_detail": "Qwen2-0.5B AR + Flow Matching + BigVGAN",
        "langs": ["ko","en","ja","zh"],
        "license": "Apache-2.0",
        "streaming": "지원",
        "official_summary": "ZH CER 1.45% / UTMOS 4.23",
        "official_perf": [
            "UTMOS 4.23 (Spark-TTS 논문 비교 기준)",
            "중국어 CER ~1.45%",
            "스트리밍 화자 유사도 SS 0.629~0.630 (비스트리밍 대비 거의 무손실)",
            "인간 수준 자연스러움(human-parity naturalness) 달성 주장",
        ],
        "paper_url": "https://arxiv.org/abs/2412.10117",
        "pros": [
            "4개 언어(KO/EN/JA/ZH) 교차 언어 클론 지원 — 한국어 참조 음성으로 영어를 생성하는 등 언어를 넘어선 화자 특성 이전 가능",
            "Flow Matching 기반 안정적 운율 — 억양·리듬이 자연스럽고 대부분 텍스트에서 발음 오류 없음",
            "BigVGAN 고품질 보코더 — 파형 품질이 높아 청취 피로감 낮음",
        ],
        "cons": [
            "레퍼런스 오디오 내용 재출력 현상 — 일부 경우 타겟 텍스트 대신 참조 음성의 내용을 그대로 발화하는 심각한 hallucination 발생 (한국어·영어·일본어·중국어 모두 확인됨)",
            "RTF ~0.68로 실시간보다 느림 — 긴 텍스트 처리 시 지연이 체감될 수 있어 실시간 스트리밍 서비스에는 부적합",
            "VRAM 4.6GB 상시 점유 — GPU 공유 환경에서 다른 모델과 동시 운용 어려움",
        ],
    },
    "cosyvoice3": {
        "display_name": "CosyVoice3",
        "developer": "Alibaba DAMO",
        "arch_type": "LLM",
        "arch_detail": "CosyVoice2 개선판 (EOS 기반 스트리밍)",
        "langs": ["ko","en","ja","zh"],
        "license": "Apache-2.0",
        "streaming": "지원",
        "official_summary": "ZH→KO CER 14.4%",
        "official_perf": [
            "CosyVoice2 기반 스트리밍 개선판 — 청크 단위 출력으로 TTFA 단축",
            "교차 언어 클론 zh→ko CER: 14.4% (Qwen3-TTS 4.82% 대비 약 3배 높음)",
        ],
        "paper_url": "https://arxiv.org/abs/2412.10117",
        "pros": [
            "EOS 토큰 기반 스트리밍 TTS — 청크 단위로 음성을 출력하여 TTFA(첫 음성 대기 시간)를 대폭 단축 가능",
            "CosyVoice2 대비 발음 정확도 개선 — 동일 아키텍처에서 추가 학습으로 한국어 억양 품질 향상",
        ],
        "cons": [
            "레퍼런스 오디오 내용 재출력 현상 — CosyVoice2와 동일한 hallucination 문제가 한국어·영어·일본어·중국어 전 언어에서 관찰됨. 참조 음성 선택에 매우 주의 필요",
            "현재 adapter 호환성 이슈 — 일부 환경에서 로딩 오류 발생, 공식 레포에서 지속적으로 업데이트 중",
        ],
    },
    "qwen3_tts_0.6b": {
        "display_name": "Qwen3-TTS-0.6B",
        "developer": "Alibaba Qwen",
        "arch_type": "LLM",
        "arch_detail": "Qwen3-0.6B LLM + 음성 코덱 디코더",
        "langs": ["ko","en","ja","zh"],
        "license": "Apache-2.0",
        "streaming": "지원",
        "official_summary": "EN WER 2.8% / ZH WER 1.9% / MOS 4.53",
        "official_perf": [
            "MOS 4.53 (업계 평균 4.1 대비 우월, 48kHz)",
            "영어 WER 2.8%, 중국어 WER 1.9% (Azure TTS 대비 각각 18%·24% 감소)",
            "10개 평가 언어 중 6개에서 최저 WER — MiniMax-Speech·ElevenLabs Multilingual v2 대비 SOTA",
            "교차 언어 클론 zh→ko CER 4.82% (CosyVoice3 14.4% 대비 66% 감소)",
        ],
        "paper_url": "https://arxiv.org/abs/2601.15621",
        "pros": [
            "10개 언어 동시 지원 — 단일 모델로 한국어·영어·일본어·중국어를 포함한 10개 언어 처리 가능. zero-shot 음성 클론 내장",
            "0.6B 소형 모델로 VRAM 2.9GB — 같은 성능 대비 메모리 효율적",
        ],
        "cons": [
            "RTF 약 1.9로 실시간보다 거의 2배 느림 — LLM 기반 자기회귀 생성 특성상, 빠른 응답이 필요한 서비스엔 부적합. 텍스트가 길어질수록 대기 시간이 선형 증가",
            "VRAM 2.9GB로 소형임에도 GPU 필수 — CPU 전용 환경에서는 사용 불가",
            "FlashAttention2 미적용",
        ],
        "opt_note": "INT8 quantization 적용 시 VRAM 44% 절감 (2,697→1,502MB), RTF 72% 증가 (2.64→4.56)",
    },
    "qwen3_tts_1.7b": {
        "display_name": "Qwen3-TTS-1.7B",
        "developer": "Alibaba Qwen",
        "arch_type": "LLM",
        "arch_detail": "Qwen3-1.7B LLM + 음성 코덱 디코더",
        "langs": ["ko","en","ja","zh"],
        "license": "Apache-2.0",
        "streaming": "지원",
        "official_summary": "EN WER 2.8% / ZH WER 1.9% / MOS 4.53",
        "official_perf": [
            "MOS 4.53 (0.6B와 동일 평가 기준, 대형 모델로 감정·운율 표현 향상)",
            "영어 WER 2.8%, 중국어 WER 1.9% (0.6B와 동일 벤치마크 기준)",
            "10개 언어 중 6개에서 최저 WER — MiniMax-Speech·ElevenLabs 대비 SOTA",
        ],
        "paper_url": "https://arxiv.org/abs/2601.15621",
        "pros": [
            "10개 언어 고품질 — 0.6B 대비 더 풍부한 감정 표현과 억양 다양성. 특히 긴 문장에서 운율 유지 능력 우수",
            "감정·스타일 제어 — LLM 기반이므로 프롬프트 엔지니어링으로 말투 조절 가능성 높음",
        ],
        "cons": [
            "RTF 약 1.7로 실시간보다 70% 더 오래 걸림 — 0.6B보다 약간 빠르지만 여전히 실시간 응답 불가. 배치 추론이나 비실시간 용도에 적합",
            "VRAM 4.8GB — RTF 5080 16GB 기준으로는 여유 있으나, 12GB 이하 GPU에서는 다른 프로세스와 공존 어려움",
            "FlashAttention2 미적용",
        ],
    },
    "gpt_sovits": {
        "display_name": "GPT-SoVITS-V3",
        "developer": "RVC-Boss",
        "arch_type": "LLM",
        "arch_detail": "GPT-2 AR + SoVITS + BigVGAN v2",
        "langs": ["ko","en","ja","zh"],
        "license": "MIT",
        "streaming": "지원",
        "official_summary": None,
        "official_perf": [
            "공식 논문 없음 (GitHub Wiki 문서 기반)",
            "학습 데이터 7,000시간 (MOS 기반 음질 필터링 적용)",
            "아키텍처: shortcut-CFM-DiT (S2), 330M+77M 파라미터",
            "SeedTTS test-en 기준 제3자 MOS 비교에서 상위권 평가 (구체적 수치 미공개)",
        ],
        "paper_url": "https://github.com/RVC-Boss/GPT-SoVITS/wiki",
        "pros": [
            "4개 언어 교차 언어 클론 — 한국어 참조 음성으로 일본어·중국어를 자연스럽게 생성. 커뮤니티 활성도가 높아 지속적인 개선",
            "V3 BigVGAN v2 보코더 — 이전 버전 대비 파형 품질 대폭 향상, 고주파 디테일이 풍부",
        ],
        "cons": [
            "Prefix hallucination — 영어 참조 음성 사용 시 레퍼런스 오디오 내용이 앞에 먼저 출력된 후 한국어가 이어지는 현상. 참조 음성 언어와 타겟 언어 불일치 시 발생",
            "짧은(3초 미만) 참조 음성에서 kernel 에러 — 참조 음성 길이가 너무 짧으면 음성 분석 실패. 최소 3~5초 이상의 참조 오디오 필요",
        ],
    },
    "xtts": {
        "display_name": "XTTS-v2",
        "developer": "Coqui AI",
        "arch_type": "NAR",
        "arch_detail": "GPT-2 + HiFi-GAN 보코더, 17개 언어",
        "langs": ["ko","en","ja","zh"],
        "license": "CPAL-1.0 (Non-commercial)",
        "streaming": "지원",
        "official_summary": "UTMOS 4.007",
        "official_perf": [
            "CMOS(자연스러움/음질/인간 유사도): HierSpeech++ 및 Mega-TTS 2 대비 우월",
            "SMOS(화자 유사도): HierSpeech++ 대비 약간 열세",
            "16개 언어 지원, Whisper Large v3 기준 우수한 CER 달성",
        ],
        "paper_url": "https://arxiv.org/abs/2406.04904",
        "pros": [
            "17개 언어 공식 지원 — 한국어 포함 17개 언어를 단일 모델로 처리. RTF ~0.28로 실시간 대비 3배 이상 빠르며 zero-shot 클론 내장",
            "한국어 공식 훈련 데이터 포함 — 별도 파인튜닝 없이 한국어 TTS로 바로 사용 가능한 유일한 상용급 오픈소스 모델 중 하나",
        ],
        "cons": [
            "Coqui AI 서비스 종료 (2024년) — 모델 업데이트 및 공식 유지보수 중단. 보안 취약점이 발생해도 공식 패치 없음. 커뮤니티 fork에 의존",
            "단문(3음절 이하) 입력에서 CER 급상승 — 매우 짧은 텍스트 입력 시 발음 품질 저하 및 오류 발생. 입력 텍스트 최소 길이 약 10자 이상 권장",
        ],
    },
    "fish_speech": {
        "display_name": "Fish-Speech",
        "developer": "Fish Audio",
        "arch_type": "NAR",
        "arch_detail": "FireflyGAN 보코더 + VQGAN 음성 토크나이저",
        "langs": ["en","ja","zh"],
        "license": "CC BY-NC-SA 4.0",
        "streaming": "지원",
        "official_summary": "WER 0.008 / RTF 0.195",
        "official_perf": [
            "후속 S2 Pro 시스템: RTF=0.195, TTFA≈100ms, 처리량 3,000+ acoustic tokens/s (H200 GPU)",
            "S1 모델: WER=0.008, CER=0.004 (공식 레포 기준)",
        ],
        "paper_url": "https://arxiv.org/abs/2603.08823",
        "pros": [
            "FireflyGAN 고품질 보코더 — 24kHz 출력으로 영어/일본어/중국어에서 뛰어난 음성 품질. zero-shot 클론으로 다양한 화자 스타일 재현 가능",
            "빠른 추론 속도 — RTF 0.195 (S2 Pro, H200 기준), TTFA ~100ms로 실시간 스트리밍 서비스에 충분",
            "활발한 유지보수 — S1→S2→S2 Pro로 지속 업데이트. 멀티링궐 확장 버전도 라인업 운영",
        ],
        "cons": [
            "한국어 학습 데이터 없음 — 모델 어휘(vocabulary)에 한국어가 포함되지 않아 한글 입력 자체가 불가. 한국어 서비스에 활용하려면 전체 재학습 필요",
            "비상업용 라이선스 (CC BY-NC-SA 4.0) — 상업 목적 서비스에는 별도 계약 또는 상용 API 이용 필요",
        ],
    },
    "melotts": {
        "display_name": "MeloTTS",
        "developer": "MyShell AI",
        "arch_type": "NAR",
        "arch_detail": "VITS 기반 비자기회귀, MeCab 형태소 분석",
        "langs": ["ko","en"],
        "license": "MIT",
        "streaming": "미지원",
        "official_summary": None,
        "official_perf": [
            "공식 MOS/WER/RTF 벤치마크 미발표 (GitHub 오픈소스 릴리즈만 존재)",
            "VITS 기반 고품질 다국어 TTS (영어, 스페인어, 프랑스어, 중국어, 일본어, 한국어 지원) 주장",
        ],
        "paper_url": "https://github.com/myshell-ai/MeloTTS",
        "pros": [
            "RTF 0.04 — 벤치마크 전체에서 최고 속도. GPU 없이 CPU만으로도 실시간의 25배 빠른 추론. Edge 서버·온디바이스 배포에 최적",
            "VRAM 861MB — 저사양 장치에서도 구동 가능. 다른 서비스와 메모리 공유가 쉬움",
        ],
        "cons": [
            "고정 화자, 음성 클론 불가 — 학습된 한국어 화자 1명만 사용 가능. 사용자 맞춤 목소리 생성을 위해서는 파인튜닝이 필요하며 이는 상당한 추가 작업",
            "한국어·영어 2개 언어만 지원 — 일본어·중국어 요구사항이 있다면 다른 모델로 대체 필요",
        ],
    },
    "chattts": {
        "display_name": "ChatTTS",
        "developer": "2Noise",
        "arch_type": "LLM",
        "arch_detail": "Conversational TTS with prosody control",
        "langs": ["en","zh"],
        "license": "CC BY-NC-SA 4.0",
        "streaming": "미지원",
        "official_summary": None,
        "official_perf": [
            "학습 데이터 40,000시간 기반 고품질 대화형 음성 생성",
            "공식 벤치마크(MOS, WER) 통계표 미발표 (보안 목적으로 오디오에 고주파 노이즈 추가됨)"
        ],
        "pros": [
            "구어체 자연스러움 — 웃음·한숨·말줄임·휴지기 등 비언어적 요소를 삽입 가능. 일반 TTS가 어색하게 읽는 대화체 텍스트에서 강점",
            "대규모 학습 데이터 — 40,000시간 학습 기반. 영어·중국어 억양 자연스러움이 동급 오픈소스 중 우수한 편",
        ],
        "cons": [
            "한국어 미지원 — 영어와 중국어 전용 훈련. 한국어 입력 시 무응답 또는 오류",
            "생성 결과 불안정 — 동일 텍스트를 반복 실행해도 결과가 크게 달라지는 경우가 많음. 재현성이 낮아 프로덕션 환경에서 QA 어려움",
        ],
    },
    "piper_ko": {
        "display_name": "Piper-KO",
        "developer": "Piper (Home Assistant)",
        "arch_type": "NAR",
        "arch_detail": "VITS-based ONNX model",
        "langs": ["ko"],
        "license": "MIT / CC",
        "streaming": "미지원",
        "official_summary": None,
        "official_perf": [
            "라즈베리파이 4 기준 실시간 대비 빠른 생성 가능 (공식 수치 미발표)"
        ],
        "pros": [
            "ONNX 기반 초고속 CPU 추론 — Raspberry Pi 등 SBC에서도 실시간 이상 동작. IoT·홈 자동화 환경에 최적화",
            "초경량 설치 — ONNX 런타임만으로 구동, GPU·CUDA 불필요. 오프라인 엣지 디바이스 배포에 가장 적합한 선택지",
        ],
        "cons": [
            "기계적·평탄한 음색 — 감정 표현이나 억양 변화가 거의 없어 장시간 청취 시 피로감. 안내 방송 등 짧은 용도에는 충분하나 감성적 콘텐츠엔 부적합",
            "고정 화자 — 학습 데이터에 포함된 단일 화자만 사용 가능. 음성 커스터마이징을 위해서는 직접 새 모델 학습 필요",
        ],
    },
    "voicecraftx": {
        "display_name": "VoiceCraft-X",
        "developer": "UT Austin",
        "arch_type": "LLM",
        "arch_detail": "Code-based AR model, 11 languages",
        "langs": ["ko","en","ja","zh"],
        "license": "CC BY-NC-SA 4.0",
        "streaming": "미지원",
        "official_summary": "WER 2.8% / MOS 4.26",
        "official_perf": [
            "원본 VoiceCraft 830M: MOS 4.26, WER 2.8% (ACL 2024) — 화자 유사도 비교 대상 중 최고",
            "VoiceCraft-X (EMNLP 2025): 11개 언어 음성 편집 지원, NMOS·IMOS가 원본 녹음과 동등 수준",
            "음성 편집 선호도: 인간이 편집 음성을 원본 녹음과 48% 동등하게 선호",
        ],
        "paper_url": "https://arxiv.org/abs/2511.12347",
        "pros": [
            "11개 언어 광범위 지원 — 한국어 포함 유럽어 등 11개 언어 zero-shot 클론. 단일 모델로 글로벌 서비스 대응 가능",
            "음성 편집 특화 (EMNLP 2025) — 기존 녹음의 일부 구간만 수정하는 in-context editing 기능. 재녹음 없이 오디오 포스트프로덕션 가능",
        ],
        "cons": [
            "RTF가 높아 배치 처리 필요 — 자기회귀 코드 생성 방식으로 실시간 응답 어려움. 미리 생성해두는 비실시간 파이프라인에 적합",
            "높은 VRAM 요구 — 대형 코드 AR 모델 특성상 GPU 메모리 사용량이 많아 소형 GPU 환경에서 OOM 위험",
            "FlashAttention2 미적용",
        ],
    },
    "maskgct": {
        "display_name": "MaskGCT",
        "developer": "Amphion",
        "arch_type": "NAR",
        "arch_detail": "Non-autoregressive with Masked modeling",
        "langs": ["ko","en","zh"],
        "license": "Apache-2.0",
        "streaming": "미지원",
        "official_summary": "EN WER 2.62 / SIM 0.717",
        "official_perf": [
            "WER 2.623 (SeedTTS test-en) / CER 2.273 (SeedTTS test-zh) — ICLR 2025 채택",
            "WER 2.634 (LibriSpeech test-clean)",
            "화자 유사도 SIM-O: 0.717 (EN) / 0.774 (ZH) — 모든 베이스라인 대비 최고 SMOS",
            "25 steps에서 SIM 안정화 (NaturalSpeech 3·VALL-E·VoiceBox·XTTS-v2·CosyVoice 대비 우월)",
        ],
        "paper_url": "https://arxiv.org/abs/2409.00750",
        "pros": [
            "비자기회귀(NAR) 병렬 생성 — 토큰을 순차적으로 생성하지 않아 AR 모델 대비 빠른 추론. 발음 안정성 높고 zero-shot 클론 지원",
            "ICLR 2025 채택 — SeedTTS 기준 화자 유사도 SIM-O 0.717(EN)/0.774(ZH)로 모든 베이스라인 중 최고 SMOS 기록",
        ],
        "cons": [
            "한국어 억양 평탄화 경향 — 한국어 학습 데이터가 상대적으로 부족하여 의문문·감탄문에서 억양 곡선이 다소 단조롭게 출력될 수 있음",
            "높은 VRAM — 9GB+ 사용으로 소형 GPU 환경에서 OOM 위험. RTF ~1.1로 실시간보다 느린 편",
        ],
    },
    "hierspeech": {
        "display_name": "Hierspeech++",
        "developer": "SNU",
        "arch_type": "NAR",
        "arch_detail": "Hierarchical TTS system",
        "langs": ["ko","en"],
        "license": "MIT",
        "streaming": "미지원",
        "official_summary": None,
        "official_perf": [
            "자연스러움 MOS에서 Ground Truth(실제 녹음) 초과 달성 주장 — 'first human-level quality zero-shot TTS' (arXiv 2311.12454)",
            "화자 유사도(EER·SECS) 기준 VALL-E-X·YourTTS·HierSpeech 대비 우월",
            "운율 MOS(pMOS): XTTS가 더 높은 pMOS 기록 (더 많은 학습 데이터 필요성 인정)",
        ],
        "paper_url": "https://arxiv.org/abs/2311.12454",
        "pros": [
            "한국어 특화 학습 — 서울대 연구팀이 한국어 데이터로 학습. 한국어 억양·리듬 표현력이 범용 모델 대비 우수",
            "초경량 고속 — VRAM 809MB, RTF 0.031(KO)로 모든 벤치마크 모델 중 한국어 추론 최고 속도. 소형 GPU에서도 실시간 서비스 가능",
        ],
        "cons": [
            "한국어 텍스트 정확도 매우 낮음 — 실측 KO CER 84%. 한국어 입력 시 엉뚱한 음절 조합을 출력하는 경향이 있어 실제 서비스 투입 불가 수준",
            "복잡한 설치 의존성 — espeak-ng·phonemizer·k2 등 다수 외부 라이브러리 필요. Windows 환경에서 특히 설치 난이도가 높으며, 버전 충돌 발생 빈번",
        ],
    },
    "llmvox": {
        "display_name": "LLMVoX",
        "developer": "MBZUAI",
        "arch_type": "LLM",
        "arch_detail": "30M Tiny LLM-addon TTS",
        "langs": ["en"],
        "license": "MIT",
        "streaming": "지원",
        "official_summary": None,
        "official_perf": [
            "파라미터 크기 30M 내외로 오버헤드 최소화 (정량 벤치마크 미비)"
        ],
        "pros": [
            "초경량 30M 구조 — 기존 LLM에 애드온 형태로 붙여 TTS 기능 추가. 모델 자체 크기가 작아 메모리 오버헤드 최소화",
            "스트리밍 지원 — 청크 단위 실시간 출력 가능. LLM 응답과 TTS를 동시에 파이프라이닝하는 음성 AI 어시스턴트 구현에 적합",
        ],
        "cons": [
            "영어 단일 언어만 지원 — 한국어·일본어·중국어 어떤 다국어도 처리 불가",
            "단독 서비스 활용 제한 — LLM 백본에 의존하는 설계로 독립 TTS 서버로 구성하려면 추가 아키텍처 작업 필요",
        ],
    },
    "openvoice": {
        "display_name": "OpenVoice-v2",
        "developer": "MyShell AI",
        "arch_type": "NAR",
        "arch_detail": "MeloTTS + ToneColorConverter 음색 변환",
        "langs": ["ko","en"],
        "license": "MIT",
        "streaming": "미지원",
        "official_summary": None,
        "official_perf": [
            "V1 논문: 음색·감정·악센트·리듬·포즈·억양 정밀 제어 성능 주장 (arXiv 2312.01479)",
            "V2 공식 수치 미발표 — 엔지니어링 개선(네이티브 다국어 추가) 위주, 별도 벤치마크 테이블 없음",
        ],
        "paper_url": "https://arxiv.org/abs/2312.01479",
        "pros": [
            "음색 변환(ToneColorConverter) 기반 클론 — MeloTTS로 고속 생성 후 참조 음성의 음색을 사후 변환. 빠른 속도와 음성 클론을 동시에 제공",
            "한국어+영어 공식 지원, 음조·감정·리듬 세밀 조정 파라미터 제공",
        ],
        "cons": [
            "MeloTTS + ToneColorConverter 2단계 파이프라인 — 두 모델이 직렬로 실행되어 환경 설정이 복잡하고, ffmpeg·MeCab 등 외부 의존성이 많아 Windows 설치 시 오류를 겪는 경우가 많음",
            "음색 유사도 한계 — ToneColorConverter는 음색 특성을 근사하는 방식이라 동일 참조 화자를 사용해도 고음·저음 극단에서 원본과 차이 발생 가능",
        ],
    },
    "outetss": {
        "display_name": "OuteTTS",
        "developer": "OuteAI",
        "arch_type": "LLM",
        "arch_detail": "LLaMA 기반 LLM 음성 생성, 9개 언어",
        "langs": ["ko","en","ja","zh"],
        "license": "Apache-2.0",
        "streaming": "미지원",
        "official_summary": None,
        "official_perf": [
            "공식 MOS/WER/RTF 벤치마크 미발표 (GitHub 릴리즈 전용)",
            "30초 이하 생성 배치에서 최적 성능 권장",
        ],
        "paper_url": "https://github.com/edwko/OuteTTS",
        "pros": [
            "9개 언어 LLM 기반 표현력 — 텍스트 LLM의 언어 이해력을 음성 생성에 활용, 복잡한 문장에서도 자연스러운 억양 유지",
            "zero-shot 음성 클론 — 짧은 참조 오디오만으로 화자 음색 재현. 한국어·일본어를 포함한 다국어 클론이 단일 모델로 가능",
        ],
        "cons": [
            "RTF 약 3.2 — 벤치마크에서 측정된 실제 처리 시간이 오디오 길이의 3배 이상. 10초 문장 생성에 30초가 걸려 실시간 응용 완전히 불가",
            "VRAM 4GB 상시 점유 — 0.6B 모델임에도 LLaMA 아키텍처 특성상 메모리 사용량이 높음",
        ],
    },
    "kokoro": {
        "display_name": "Kokoro",
        "developer": "hexgrad",
        "arch_type": "NAR",
        "arch_detail": "경량 VITS 변형 모델, 82M 파라미터",
        "langs": ["en","ja","zh"],
        "license": "Apache-2.0",
        "streaming": "미지원",
        "official_summary": "Arena #1 (TTS Spaces)",
        "official_perf": [
            "TTS Spaces Arena #1 랭킹 달성 (v0.19 기준, XTTS v2·MetaVoice·Fish Speech 등 상위)",
            "공식 MOS/WER 벤치마크 미발표 — Arena 순위가 유일한 공식 비교",
            "훈련 비용 약 $400 (A100 80GB, ~500 GPU시간, 학습 데이터 100시간 미만)",
        ],
        "paper_url": "https://huggingface.co/hexgrad/Kokoro-82M",
        "pros": [
            "RTF 0.011 — 벤치마크 전체 2위 속도(ZipVoice-FT 제외). GPU 없이 CPU로도 초실시간 추론 가능. 82M 초경량으로 배포가 매우 쉬움",
            "영어 MOS 최고 수준 — 영어 발음 정확도와 자연스러움이 이 벤치마크에서 최고 평가",
        ],
        "cons": [
            "한국어 완전 미지원 — 한글 입력 자체를 처리할 어휘가 없어 한국어 문자 입력 시 오류 또는 무음 출력",
            "프리셋 화자만 사용 가능 — 음성 클론 기능 없음. 제공된 약 50개 내장 화자 중에서만 선택해야 함",
        ],
    },
    "bark": {
        "display_name": "Bark",
        "developer": "Suno AI",
        "arch_type": "LLM",
        "arch_detail": "GPT 계열 오디오 언어 모델",
        "langs": ["en"],
        "license": "MIT",
        "streaming": "미지원",
        "official_summary": None,
        "official_perf": [
            "공식 MOS/WER/RTF 벤치마크 미발표 (GitHub 오픈소스 릴리즈만 존재)",
            "다국어 음성 + 음악·효과음 생성, 비언어 표현(웃음·한숨 등) 지원이 주요 차별화 포인트",
        ],
        "paper_url": "https://github.com/suno-ai/bark",
        "pros": [
            "비언어 표현 삽입 — [laughter]·[sighs]·[music] 등 특수 토큰으로 웃음·한숨·배경음악을 음성에 삽입 가능. 감정 표현 범위가 타 모델과 차원이 다름",
            "오디오 LM 기반 다국어 — 한국어 포함 다국어 텍스트를 입력할 수 있으며, 비영어 화자 목소리 클론도 시도 가능 (품질 불안정하지만 구조적으로 지원)",
        ],
        "cons": [
            "RTF 매우 높음 — GPT 계열 오디오 LM의 3단계(텍스트→semantic→coarse→fine) 순차 생성으로 처리 시간이 길어 실용적 서비스 도입 어려움",
            "영어 외 언어에서 불안정 — 한국어 등 비영어 입력 시 영어가 섞이거나 무관한 오디오 출력되는 사례 다수 보고됨",
        ],
    },
    "spark_tts": {
        "display_name": "Spark-TTS",
        "developer": "SparkAudio",
        "arch_type": "LLM",
        "arch_detail": "Qwen2.5-0.5B + BiCodec 음성 코덱",
        "langs": ["en","zh"],
        "license": "Apache-2.0",
        "streaming": "미확인",
        "official_summary": "WER 1.98 / UTMOS 4.35",
        "official_perf": [
            "UTMOS 4.35 (GT=4.08 초과, CosyVoice2=4.23 대비 우월)",
            "영어 WER 1.98 (SeedTTS test-en) — F5-TTS 다음 2위",
            "화자 유사도 SIM 0.584",
            "성별 제어 정확도 99.77%",
            "BiCodec 코덱: STOI=0.92, PESQ NB=3.13, UTMOS=4.18, SIM=0.80",
        ],
        "paper_url": "https://arxiv.org/abs/2503.01710",
        "pros": [
            "텍스트 기반 음성 속성 제어 — 성별·말하기 속도·음조를 자연어 프롬프트로 지정 가능. 캐릭터 설정이 필요한 게임·애니 더빙에 유용",
            "UTMOS 4.35로 GT 초과 — 실제 녹음(GT=4.08) 대비 더 높은 음질 평가 달성. 영어 WER 1.98% (SeedTTS 기준 2위)",
        ],
        "cons": [
            "한국어·일본어 어휘 없음 — BiCodec 음성 코덱이 영어·중국어로만 학습되어 한국어 입력 시 토크나이저 단계에서 처리 실패",
            "스트리밍 미확인 — 공식 문서에 실시간 스트리밍 지원 여부 명시 없음. 추론 파이프라인 구조상 청크 출력이 어려울 수 있음",
        ],
    },
    "index_tts2": {
        "display_name": "IndexTTS-2",
        "developer": "Bilibili Index",
        "arch_type": "LLM",
        "arch_detail": "Qwen GPT + BigVGAN, 개선된 다국어 처리",
        "langs": ["en","ja","zh"],
        "license": "CC BY-NC-SA 4.0",
        "streaming": "미확인",
        "official_summary": "T2S RTF 0.119",
        "official_perf": [
            "IndexTTS 2.5: T2S 모듈 RTF 0.232→0.119 (2.28× 속도 향상, 품질 저하 무감지)",
            "IndexTTS2: SMOS·PMOS·QMOS·EMOS 평가에서 SeedTTS test-zh 기준 최고점 달성 주장",
            "IndexTTS v1: XTTS·CosyVoice2·Fish-Speech·F5-TTS 대비 자연스러움·내용 일관성에서 우월",
        ],
        "paper_url": "https://arxiv.org/abs/2506.21619",
        "pros": [
            "중국어·영어·일본어 고품질 — Bilibili 자체 데이터로 학습된 중국어 발음이 특히 우수. BigVGAN 보코더로 높은 오디오 충실도",
            "T2S RTF 0.119 고속 — IndexTTS 2.5 기준 이전 버전 대비 2.28× 속도 향상 달성. SMOS·PMOS·QMOS·EMOS 전 항목 SeedTTS test-zh 최고점",
        ],
        "cons": [
            "한국어 미지원 — 학습 어휘에 한국어 포함되지 않아 한글 처리 불가",
            "RTF 높은 편 — Qwen 기반 LLM 자기회귀 추론으로 긴 텍스트 처리에 시간 소요. 짧은 문장은 상대적으로 빠르나 긴 문장에서 병목 발생",
        ],
    },
    "mio_tts_1.7b": {
        "display_name": "MioTTS-1.7B",
        "developer": "Mio Team",
        "arch_type": "LLM",
        "arch_detail": "Qwen3-1.7B + MioCodec",
        "langs": ["en","ja"],
        "license": "Apache-2.0",
        "streaming": "지원",
        "official_summary": "EN WER ~3% / RTF < 0.5 (A100)",
        "official_perf": [
            "EN WER ~3% (MioTeam 내부 평가, 제3자 미검증), JA 공식 평가 미공개",
            "RTF < 0.5 (A100 기준, 내부 측정)",
        ],
        "pros": [
            "Qwen3-1.7B 기반 영어·일본어 고품질 — 최신 LLM 백본으로 복잡한 영어 문장과 일본어 경어·구어체를 자연스럽게 처리",
            "스트리밍 지원 — 청크 단위 실시간 출력 가능, 지연 시간 최소화",
            "경량 모델 — 1.7B 파라미터로 소비자급 GPU(VRAM 8GB)에서 실시간 추론 가능",
        ],
        "cons": [
            "한국어·중국어 학습 데이터 미포함 — 영어·일본어 2개 언어 전용. 한국어 서비스에는 사용 불가",
            "공식 벤치마크 데이터 미공개 — 제3자 독립 평가 없음, 내부 수치에만 의존",
        ],
    },
    "mio_tts_2.6b": {
        "display_name": "MioTTS-2.6B",
        "developer": "Mio Team",
        "arch_type": "LLM",
        "arch_detail": "LFM2-2.6B + MioCodec",
        "langs": ["en","ja"],
        "license": "Apache-2.0",
        "streaming": "지원",
        "official_summary": "EN WER ~2% / RTF < 0.6 (A100)",
        "official_perf": [
            "EN WER ~2% (MioTeam 내부 평가, 1.7B 대비 개선, 제3자 미검증)",
            "RTF < 0.6 (A100 기준, 내부 측정)",
        ],
        "pros": [
            "LFM2-2.6B 대형 모델 — 1.7B 대비 더 풍부한 표현력. 영어·일본어 중 높은 품질이 필요한 콘텐츠 제작에 최적",
            "스트리밍 지원 — 청크 단위 실시간 출력, 방송·게임 등 실시간 서비스 적합",
            "Liquid Foundation Model 아키텍처 — 기존 Transformer 대비 긴 시퀀스 처리 효율 개선",
        ],
        "cons": [
            "한국어·중국어 미지원 — 영어·일본어 2개 언어 전용",
            "VRAM 사용량 높음 — 2.6B 파라미터 LFM2 모델로 GPU 메모리 소비가 커서 VRAM 8GB 이하 환경에서는 배치 크기 제한 필요",
            "공식 벤치마크 데이터 미공개 — 제3자 독립 평가 없음",
        ],
    },
    "glm_tts": {
        "display_name": "GLM-TTS",
        "developer": "Zhipu AI",
        "arch_type": "LLM",
        "arch_detail": "GLM 기반 + RL 강화학습 품질 최적화",
        "langs": ["en","zh"],
        "license": "Apache-2.0",
        "streaming": "지원",
        "official_summary": "ZH CER 0.89%",
        "official_perf": [
            "중국어 CER 0.89% (RL 최적화 후) — 오픈소스 모델 중 최고, 상용 MiniMax(0.83%)와 경쟁적",
            "학습 데이터 100,000시간",
            "GRPO 기반 다중 보상 강화학습 (발음·화자 유사도·운율 공동 최적화)",
        ],
        "paper_url": "https://arxiv.org/abs/2512.14291",
        "pros": [
            "RL(강화학습) 기반 품질 최적화 — GRPO 다중 보상으로 발음·화자 유사도·운율을 공동 최적화. 중국어 CER 0.89% (상용 MiniMax 0.83% 근접)",
            "대규모 학습 — 100,000시간 학습 데이터. 스트리밍 지원으로 실시간 서비스 투입 가능",
        ],
        "cons": [
            "한국어 OOV(어휘 미포함) 문제 — GLM 학습 어휘에 한글이 없어 한국어 문자는 Unknown 토큰으로 처리됨. 한국어 입력 불가",
            "영어·중국어 전용 — 한국어·일본어 서비스에는 완전히 사용 불가. 벤치마크에서 한국어 CUDA assertion 오류 발생",
        ],
    },
    "f5tts": {
        "display_name": "F5-TTS",
        "developer": "SWivid",
        "arch_type": "Flow",
        "arch_detail": "Flow Matching + DiT (Diffusion Transformer)",
        "langs": ["en","zh"],
        "license": "CC BY-NC-SA 4.0",
        "streaming": "미지원",
        "official_summary": "WER 2.42 / RTF 0.15",
        "official_perf": [
            "WER 2.42 (LibriSpeech-PC test-clean, 32 NFE + Sway Sampling)",
            "RTF 0.15 (16 NFE 기준, 10초 음성 추론)",
            "CMOS 0.31 / SMOS 3.89 (SeedTTS test-en)",
            "중국어 CER 1.53% (SeedTTS test-zh, GT duration + 32 NFE)",
        ],
        "paper_url": "https://arxiv.org/abs/2410.06885",
        "pros": [
            "RTF 0.16 빠른 Flow Matching — 확산 모델 계열임에도 불구하고 단 few-step으로 고품질 생성. 영어·중국어 zero-shot 클론에서 자연스러운 운율",
            "zh_short 단문 처리 실패 제외 시 중국어 품질 우수",
        ],
        "cons": [
            "영어+중국어 2개 언어만 지원 — 한국어·일본어 미포함",
            "zh_short(2~3음절 중국어) 입력 시 빈 오디오 출력 — 매우 짧은 중국어 입력에서 Flow Matching이 수렴 실패하는 버그 존재 (이 벤치마크에서도 확인됨)",
        ],
    },
    "chatterbox": {
        "display_name": "Chatterbox-TTS",
        "developer": "Resemble AI",
        "arch_type": "NAR",
        "arch_detail": "Conditional Flow Matching 기반",
        "langs": ["en"],
        "license": "CC BY-NC-SA 4.0",
        "streaming": "미지원",
        "official_summary": "선호도 63.75% vs ElevenLabs",
        "official_perf": [
            "맹목(blind) 비교에서 ElevenLabs 대비 63.75% 평가자가 Chatterbox 선호 (Podonos 플랫폼, 7-20초 클립, zero-shot)",
            "Chatterbox Turbo RTF 0.499 (RTX 4090 기준), 첫 청크 지연 약 472ms",
            "공식 MOS/WER 수치 미발표",
        ],
        "paper_url": "https://github.com/resemble-ai/chatterbox",
        "pros": [
            "Flow Matching 기반 zero-shot 클론 — 짧은 참조 오디오만으로 자연스러운 음성 클론. 영어 발음 자연스러움 우수",
            "ElevenLabs 대비 63.75% 선호도 — 맹목 비교에서 상용 최고 수준 TTS를 오픈소스가 앞선 결과. MIT 기반 Chatterbox Multilingual 버전 별도 제공",
        ],
        "cons": [
            "영어 단일 언어 — 한국어를 포함한 타 언어 처리 불가 (Multilingual 버전은 별도 모델)",
            "비상업용 라이선스 (CC BY-NC-SA 4.0) — 상업 서비스에는 Resemble AI 별도 계약 필요",
        ],
    },
    "parler": {
        "display_name": "Parler-TTS-mini",
        "developer": "Hugging Face",
        "arch_type": "NAR",
        "arch_detail": "T5-3B 인코더 + 오디오 디코더",
        "langs": ["en"],
        "license": "Apache-2.0",
        "streaming": "미지원",
        "official_summary": None,
        "official_perf": [
            "TTSDS 평가에서 GT 대비 MOS 0.05 이내 시스템 13개 중 하나로 포함",
            "InstructTTSEval: mini와 large 간 유의미한 성능 차이 없음, 일부 케이스에서 mini가 large 초과",
            "공식 독립 MOS/WER 테이블 미발표",
        ],
        "paper_url": "https://github.com/huggingface/parler-tts",
        "pros": [
            "자연어 설명으로 음성 스타일 제어 — 'A warm female voice with slow pace'처럼 텍스트 설명으로 화자 특성 지정. 참조 오디오 없이도 원하는 스타일 구현 가능",
            "TTSDS 평가 상위권 — GT 대비 MOS 0.05 이내 시스템 13개 중 포함. 소형(mini)이 large와 유사한 성능 유지",
        ],
        "cons": [
            "RTF ~1.1로 실시간보다 느림 — T5-3B 인코더 처리 오버헤드로 인해 실시간 응답 어려움. 영어 전용, VRAM 3.4GB 필요",
            "스타일 설명 표현 범위 제한 — 학습된 스타일 어휘 밖의 묘사(예: 특정 사투리·지역 억양)는 잘 반영되지 않으며, 설명이 구체적일수록 결과 예측이 어려워짐",
        ],
    },
    "styletts2": {
        "display_name": "StyleTTS2 (S2)",
        "developer": "yl4579",
        "arch_type": "NAR",
        "arch_detail": "Style Diffusion + SLM 스타일 전이 (비자기회귀)",
        "langs": ["en"],
        "license": "MIT",
        "streaming": "미지원",
        "official_summary": "MOS 4.38 (OOD)",
        "official_perf": [
            "MOS 3.83 (LJSpeech in-distribution) / 4.38 (OOD 텍스트, Ground Truth 초과) — NeurIPS 2023",
            "CMOS vs Ground Truth +0.28 (p<0.05, LJSpeech) — GT를 통계적으로 초과",
            "CMOS vs NaturalSpeech +1.07 (p≪0.01) — 압도적 우위",
            "CMOS vs VALL-E +0.67 (p≪0.01) — 학습 데이터 245시간으로 60,000시간 대비 달성",
            "VCTK 제로샷: CMOS vs GT -0.02 (p≫0.05) — 통계적으로 인간 수준 동등",
        ],
        "paper_url": "https://arxiv.org/abs/2306.07691",
        "pros": [
            "RTF 0.306 초고속 생성 — 비자기회귀(NAR) Diffusion 방식을 사용하여 LLM 계열 대비 5~10배 빠름",
            "VRAM 0MB (CPU 추론 가능) — 모델이 가벼워 GPU 없이 CPU로 실시간 추론이 가능하며 상시 메모리 점유 부담 없음",
            "최고 수준의 영어 품질 — 오픈소스 영어 TTS 중 가장 자연스러운 억양과 스타일 전이 성능 보유",
        ],
        "cons": [
            "영어 단일 언어 — 현재 한국어·일본어·중국어를 지원하지 않아 국내 서비스 직접 적용 불가",
            "음성 클론의 정밀도 한계 — 스타일 전이 방식의 특성상 화자의 절대적 음색 일치도보다는 스타일의 유사성에 집중",
        ],
    },
    "dia": {
        "display_name": "Dia",
        "developer": "Nari Labs",
        "arch_type": "LLM",
        "arch_detail": "1.6B AR 모델, [S1]/[S2] 대화 포맷",
        "langs": ["en"],
        "license": "Apache-2.0",
        "streaming": "미지원",
        "official_summary": None,
        "official_perf": [
            "공식 MOS/WER 벤치마크 미발표 (GitHub 오픈소스 릴리즈, 2025년 4월)",
            "비공식 초기 평가: Sesame CSM-1B·ElevenLabs 대비 자연스러움·표현력에서 우월 주장 (공식 수치 아님)",
            "추론 속도: NVIDIA A4000 기준 약 40 tokens/second",
        ],
        "paper_url": "https://github.com/nari-labs/dia",
        "pros": [
            "2화자 대화형 TTS — [S1]·[S2] 태그로 두 화자의 대화를 하나의 오디오로 자연스럽게 합성 가능. 팟캐스트·대화 시나리오 제작에 특화",
            "표현력 높은 영어 — 비공식 평가에서 ElevenLabs·Sesame CSM-1B 대비 자연스러움·감정 표현 우위 주장. 영어 억양 다양성이 뛰어남",
        ],
        "cons": [
            "[S1]/[S2] 포맷이 CER 지표를 왜곡 — 화자 태그 포함 텍스트와 순수 음성 전사 비교 시 불일치 발생 → 실제 발음 품질은 CER 수치보다 좋음. 벤치마크 해석 시 주의 필요",
            "RTF ~2.6 (느림), 영어 전용, VRAM 3.4GB",
            "FlashAttention2·torch.compile 미적용",
        ],
    },
    "mars5": {
        "display_name": "MARS5-TTS",
        "developer": "Camb-AI",
        "arch_type": "LLM",
        "arch_detail": "AR Codec LM, Shallow/Deep Clone 2단계",
        "langs": ["en"],
        "license": "CC BY-NC-SA 4.0",
        "streaming": "미지원",
        "official_summary": None,
        "official_perf": [
            "공식 MOS/WER 벤치마크 미발표 — GitHub README에 \"표준 데이터셋 벤치마크 성능 수치는 개선 작업 중\"으로 명시",
            "2단계 AR-NAR 파이프라인: L0 encodec을 AR Transformer로 생성 후 DDPM으로 정제",
            "참조 오디오 5초 이상 필요, 스포츠 중계·애니메이션 등 prosodically hard 시나리오 지원 주장",
        ],
        "paper_url": "https://github.com/Camb-ai/MARS5-TTS",
        "pros": [
            "Shallow/Deep Clone 2단계 선택 — Shallow는 빠르게, Deep는 더 정밀하게 참조 화자 특성 이전. 품질과 속도를 목적에 맞게 선택 가능",
            "prosodically hard 시나리오 지원 — 스포츠 중계·애니메이션·연기 등 극적 억양 변화가 많은 텍스트에서 강점. 5초 이상 참조 오디오로 화자 특성을 정밀하게 재현",
        ],
        "cons": [
            "RTF ~3.3으로 벤치마크에서 가장 느린 축 — 2단계 AR 생성 파이프라인으로 처리 시간이 매우 길어 실시간 서비스 부적합",
            "영어 전용 — 한국어 및 기타 언어 미지원",
        ],
    },
    "zipvoice_finetuned": {
        "display_name": "ZipVoice-FT",
        "developer": "자체 파인튜닝 (Aicess)",
        "arch_type": "FM",
        "arch_detail": "Flow Matching, ZipVoice 123M, Vocos 24kHz",
        "langs": ["ko", "en", "zh", "ja"],
        "license": "GPL-3.0",
        "streaming": "지원",
        "official_summary": "KO RTF 0.003 (벤치마크 최고속)",
        "official_perf": [
            "KO RTF 0.003 — 이 벤치마크 전체 모델 중 한국어 추론 최고속 (REST API 캐시 적중 기준)",
            "KO CER 0% — 한국어 발음 정확도 완벽 (Whisper large-v3 기준)",
            "내부 평가 수치이며 독립 제3자 검증 미완료",
        ],
        "pros": [
            "한국어 특화 파인튜닝 — 자체 한국어 데이터로 추가 학습하여 원본 ZipVoice 대비 한국어 발음 정확도 개선",
            "Flow Matching 빠른 추론 — 123M 경량 모델로 RTF가 낮고 스트리밍 가능",
        ],
        "cons": [
            "자체 수집 데이터 한계 — 파인튜닝에 사용한 한국어 학습 데이터 규모·다양성이 상용 모델 대비 제한적으로, 특수 단어·억양에서 일반화 성능이 낮을 수 있음",
            "k2 미설치 시 속도 저하 — 선택적 의존성인 k2(k2-fsa)가 없을 경우 내부 처리 경로가 slow path로 전환되어 RTF가 2~3배 높아짐. Windows에서 k2 설치가 비교적 까다로움",
        ],
    },
    "zipvoice": {
        "display_name": "ZipVoice-Official",
        "developer": "N/A (Original)",
        "arch_type": "Flow",
        "arch_detail": "Original ZipVoice (Flow Matching)",
        "langs": ["en", "zh"],
        "license": "GPL-3.0",
        "streaming": "미지원",
        "pros": [
            "오리지널 ZipVoice 아키텍처 — 논문에서 검증된 Flow Matching 기반 구조. 영어·중국어에서 안정적 품질",
        ],
        "cons": [
            "한국어 학습 데이터 미포함 — 공식 모델은 영어·중국어만 지원. 한국어 사용 시 ZipVoice-FT(파인튜닝 버전) 필요",
        ],
    },
    "kani": {
        "display_name": "Kani-TTS",
        "developer": "Kani Team",
        "arch_type": "LLM",
        "arch_detail": "370M AR LLM 기반 다국어 TTS",
        "langs": ["ko", "en", "zh"],
        "license": "Apache-2.0",
        "streaming": "미확인",
        "official_summary": "WER <5% / MOS 4.3",
        "official_perf": [
            "MOS 4.3/5 (공식 자체 주장)",
            "WER 5% 미만 (공식 자체 주장)",
            "주의: 독립 벤치마크 테이블 없음 — 자체 주장 수치이며 제3자 검증 미완료",
        ],
        "paper_url": "https://github.com/nineninesix-ai/kani-tts",
        "pros": [
            "370M 경량 AR 모델로 한국어·영어·중국어 지원 — 소형 LLM 기반으로 여러 언어를 단일 모델에서 처리",
            "zero-shot 음성 클론 — 참조 오디오 기반 화자 재현 지원. 한국어 zero-shot 클론이 가능한 소수 모델 중 하나",
        ],
        "cons": [
            "RTF 약 4.0으로 매우 느림 — LLM 자기회귀 특성이 강하게 나타나 실시간 응용 불가. 오프라인 배치 처리 용도에만 현실적",
            "일본어 미지원 — 4개 벤치마크 언어 중 일본어 처리 불가",
        ],
    },
    "supertonic": {
        "display_name": "Supertonic-v2",
        "developer": "Supertone",
        "arch_type": "LLM",
        "arch_detail": "Supertonic v2, 다국어 zero-shot TTS",
        "langs": ["ko", "en", "es", "pt", "fr"],
        "license": "MIT + OpenRAIL-M",
        "streaming": "미확인",
        "pros": [
            "RTF ~0.001 보고 — 현존 가장 빠른 수준의 추론 속도, 대규모 배치·실시간 모두 적합",
            "MIT+OpenRAIL-M 상업 가능 라이선스",
            "KO/EN/ES/PT/FR 5개 언어 zero-shot 지원",
        ],
        "cons": [
            "벤치마크 미실행 — RTF·CER 직접 측정 전, 설치·검증 진행 중",
            "OpenRAIL-M 조항 — 일부 금지 사용 사례(딥페이크 등) 제한",
        ],
    },
    "chatterbox_multilingual": {
        "display_name": "Chatterbox-ML",
        "developer": "Resemble AI",
        "arch_type": "LLM",
        "arch_detail": "Conditional Flow Matching, 23언어 Multilingual",
        "langs": ["ko", "en", "ja", "zh"],
        "license": "MIT",
        "streaming": "미지원",
        "official_summary": "ELO 1501 (TTS Arena V2 #1)",
        "official_perf": [
            "TTS Arena V2 오픈소스 1위 — ELO 1501 (2025-09 기준)",
            "맹목 비교에서 ElevenLabs 대비 63.75% 선호 (단일 언어 Chatterbox 기준)",
            "23개 언어 지원 (KO 포함), zero-shot 음성 클론",
        ],
        "paper_url": "https://github.com/resemble-ai/chatterbox",
        "pros": [
            "TTS Arena V2 오픈소스 1위 — ELO 1501, 오픈소스 모델 중 자연스러움 최고 수준",
            "MIT 상업 라이선스 — 제약 없이 상업 서비스 배포 가능",
            "23언어 zero-shot 클론 — 한국어 포함, 짧은 참조 음성으로 준수한 품질",
        ],
        "cons": [
            "RTF 미측정 — 벤치마크 실행 전, 실시간성 검증 필요",
            "내부 한국어 G2P 등 세부 처리 능력 검증 필요",
        ],
    },
    "fish_speech_s2": {
        "display_name": "Fish Audio S2 Pro",
        "developer": "Fish Audio",
        "arch_type": "LLM",
        "arch_detail": "Dual-AR (Qwen3-4B slow + 400M fast), DAC codec, 44.1kHz",
        "langs": ["ko", "en", "ja", "zh"],
        "license": "Research License (Apache 2.0 병기 혼재 — 상업 시 별도 계약)",
        "streaming": "지원",
        "official_summary": "EN WER 0.99% / ZH WER 0.54%",
        "official_perf": [
            "EN WER 0.99% / ZH WER 0.54% (SeedTTS test 기준, 공식 레포)",
            "RTF 0.195 (H200 GPU), TTFA ~100ms, 처리량 3,000+ acoustic tokens/s",
            "80+ 언어 지원, 44.1kHz 고품질 출력 (DAC codec, 10 RVQ codebooks)",
            "출시: 2026-03-10 (arXiv 2603.08823)",
        ],
        "paper_url": "https://arxiv.org/abs/2603.08823",
        "pros": [
            "Dual-AR 초고속 스트리밍 — 4B slow AR이 의미론을 처리하고 400M fast AR이 음향 디테일을 병렬 생성. TTFA ~100ms로 실시간 스트리밍에 최적화",
            "80+ 언어 지원, 44.1kHz 고품질 — DAC 코덱으로 고해상도 오디오 출력, Fish-Speech v1 계열 대비 대폭 성능 향상",
        ],
        "cons": [
            "Linux only, Python 3.12, 24GB+ VRAM 요구 — Windows 미지원 및 고사양 GPU 필수. 현재 환경(Windows RTX 5080)에서 직접 실행 불가",
            "CLI 3단계 파이프라인 — 간단한 Python API 없음. VQ 토크나이저 → AR 생성 → DAC 디코딩 순 설치/배포 복잡",
            "Fish Audio Research License — 비상업 무료이나 상업 사용 별도 계약 필요 (사실상 상업 불가)",
        ],
    },
    "higgs_audio": {
        "display_name": "Higgs Audio V2.5",
        "developer": "Boson AI",
        "arch_type": "LLM",
        "arch_detail": "LLM 기반 다국어 TTS, ~10만 시간(100K hr) 학습 데이터",
        "langs": ["ko", "en", "zh"],
        "license": "Apache 2.0",
        "streaming": "미확인",
        "official_summary": None,
        "official_perf": [
            "한국어 자연스러움 1위 주장 — 약 10만 시간(100K hr) 다국어 학습 데이터 기반",
            "RTF 미공개 — 실시간 추론 성능 공식 수치 없음",
            "KO/EN/ZH 다국어 zero-shot 지원",
        ],
        "paper_url": "https://github.com/boson-ai/higgs-audio",
        "pros": [
            "Apache 2.0 상업 라이선스 — 제약 없는 상업 서비스 배포 가능",
            "한국어 품질 강점 주장 — 대규모 데이터 학습으로 KO 자연스러움 최상위 주장",
        ],
        "cons": [
            "벤치마크 미실행 — RTF·CER 직접 측정 전, 라이선스·설치 검증 중",
            "공식 RTF 미공개 — 실시간성 불확실",
        ],
    },
    "gpt_sovits_v4": {
        "display_name": "GPT-SoVITS V4",
        "developer": "RVC-Boss",
        "arch_type": "LLM",
        "arch_detail": "GPT + SoVITS, 48kHz 고품질, V3 개선판",
        "langs": ["ko", "en", "ja", "zh"],
        "license": "MIT",
        "streaming": "지원",
        "pros": [
            "V3 대비 48kHz 고해상도 오디오 — 음질 대폭 향상",
            "MIT 상업 라이선스",
            "GPT-SoVITS V3 이미 벤치마크 완료 — 비교 기준 확보됨",
        ],
        "cons": [
            "벤치마크 미실행 — V4 RTF·CER 직접 측정 전",
            "V3와 동일 아키텍처 계열 — 속도 이점은 미확인",
        ],
    },
    # ── 벤치마크 제외 모델 (MODEL_INFO 등록 — 표 표시용) ─────────────────────
    "seamless_m4t_v2": {
        "display_name": "SeamlessM4T v2",
        "developer": "Meta",
        "arch_type": "LLM",
        "arch_detail": "Transformer 기반 음성·번역 통합 모델 (ASR/TTS/ST/MT)",
        "langs": ["ko", "en", "ja", "zh"],
        "license": "CC BY-NC 4.0",
        "streaming": "미지원",
        "pros": ["200개 언어 음성 인식·번역·합성 통합 지원"],
        "cons": ["CC BY-NC — 비상업 전용", "TTS 품질보다 번역 특화 모델"],
    },
    "piper_tts": {
        "display_name": "Piper-TTS",
        "developer": "Home Assistant",
        "arch_type": "NAR",
        "arch_detail": "VITS 기반 ONNX 경량 TTS (per-language 모델)",
        "langs": ["en"],
        "license": "MIT",
        "streaming": "지원",
        "pros": ["MIT 상업 라이선스", "ONNX 경량 — 라즈베리파이 수준 저사양 실시간 구동"],
        "cons": ["한국어 공식 미지원 — G2P(pygoruut) 의존, piper 1.4.1 비호환"],
    },
    "mms_tts": {
        "display_name": "MMS-TTS",
        "developer": "Meta",
        "arch_type": "NAR",
        "arch_detail": "VITS 기반 per-language 모델, 1,100+ 언어",
        "langs": ["ko", "en"],
        "license": "CC BY-NC 4.0",
        "streaming": "미지원",
        "pros": ["1,100+ 언어 지원 — 세계 최다 언어 커버리지"],
        "cons": ["CC BY-NC — 비상업 전용", "uroman 로마자 변환 G2P 필수 전처리"],
    },
    "vits2": {
        "display_name": "VITS2",
        "developer": "jaywalnut310",
        "arch_type": "NAR",
        "arch_detail": "VITS 개선판 — Transformer 기반 텍스트 인코더, 단일 단계 합성",
        "langs": ["en", "zh"],
        "license": "MIT",
        "streaming": "미지원",
        "pros": ["VITS 대비 자연스러움 개선"],
        "cons": ["공식 다국어 가중치 미공개", "G2P 전처리 의존"],
    },
    "seed_tts": {
        "display_name": "Seed-TTS",
        "developer": "ByteDance",
        "arch_type": "LLM",
        "arch_detail": "자기회귀 코덱 언어 모델 기반 (비공개 아키텍처)",
        "langs": ["zh", "en"],
        "license": "비공개",
        "streaming": "미확인",
        "pros": ["자체 평가 기준 SOTA급 자연스러움·화자 유사도 주장"],
        "cons": ["가중치·모델카드 완전 비공개", "상업 불가"],
    },
    "naturalspeech": {
        "display_name": "NaturalSpeech 1/2/3",
        "developer": "Microsoft",
        "arch_type": "Flow",
        "arch_detail": "NS1: VAE+GAN, NS2: Latent Diffusion, NS3: DiTTo Diffusion",
        "langs": ["en", "zh"],
        "license": "비공개",
        "streaming": "미지원",
        "pros": ["논문 기준 인간 수준(human-parity) 자연스러움 달성 주장"],
        "cons": ["가중치 비공개", "상업 불가"],
    },
    "valle": {
        "display_name": "VALL-E / BASE TTS",
        "developer": "Microsoft",
        "arch_type": "LLM",
        "arch_detail": "Neural Codec LM (EnCodec 토큰 기반 AR 생성)",
        "langs": ["en", "zh"],
        "license": "비공개",
        "streaming": "미지원",
        "pros": ["3초 참조 음성으로 zero-shot 클론 개념 최초 제안"],
        "cons": ["코드·가중치 비공개", "상업 불가"],
    },
    "e2_tts": {
        "display_name": "E2-TTS",
        "developer": "Microsoft",
        "arch_type": "Flow",
        "arch_detail": "Flow Matching 기반 (Embarrassingly Easy TTS)",
        "langs": ["en"],
        "license": "비공개",
        "streaming": "미지원",
        "pros": ["텍스트 인코더 없이 문자 시퀀스 직접 사용 — 구조 단순화"],
        "cons": ["공식 가중치 미공개 (논문만 공개)", "상업 불가"],
    },
    "stable_audio_open": {
        "display_name": "Stable Audio Open 1.0",
        "developer": "Stability AI",
        "arch_type": "Flow",
        "arch_detail": "Latent Diffusion (DiT) 기반 오디오 생성",
        "langs": ["en"],
        "license": "Stability AI Community",
        "streaming": "미지원",
        "pros": ["오픈소스 음악·효과음 생성 품질 우수"],
        "cons": ["TTS 아님 — 음악·효과음 전용, 한국어 TTS 범위 밖"],
    },
}

MODEL_RELEASE_INFO = {
    "cosyvoice2":         {"first_release": "2024-12",  "latest_ver": "Fun-CosyVoice3 (2025.08)",     "update_freq": "수개월 간격 신버전"},
    "cosyvoice3":         {"first_release": "2025-08",  "latest_ver": "0.5B-2512",                    "update_freq": "수개월 간격 신버전"},
    "qwen3_tts_0.6b":     {"first_release": "2026-01",  "latest_ver": "v1.0",                         "update_freq": "수개월 간격 업데이트"},
    "qwen3_tts_1.7b":     {"first_release": "2026-01",  "latest_ver": "v1.0",                         "update_freq": "수개월 간격 업데이트"},
    "gpt_sovits":         {"first_release": "2024-02",  "latest_ver": "V4 / V2Pro (2025.06)",         "update_freq": "수개월 간격 신버전"},
    "xtts":               {"first_release": "2023-11",  "latest_ver": "커뮤니티 포크 유지",           "update_freq": "원회사 폐업, 공식 업데이트 없음"},
    "fish_speech":        {"first_release": "2024-06",  "latest_ver": "v1.5.1 (2025.03)",             "update_freq": "수개월 간격 업데이트"},
    "melotts":            {"first_release": "2024-02",  "latest_ver": "v0.1.2 (2024.03)",             "update_freq": "마지막 업데이트 2024.03"},
    "openvoice":          {"first_release": "2023-05",  "latest_ver": "V2 (2024.04)",                 "update_freq": "마지막 업데이트 2024.04"},
    "outetss":            {"first_release": "2024-11",  "latest_ver": "v1.0 (2025.04)",               "update_freq": "수개월 간격 업데이트"},
    "kokoro":             {"first_release": "2024-12",  "latest_ver": "v0.19+ (2025.11)",             "update_freq": "수개월 간격 업데이트"},
    "bark":               {"first_release": "2023-04",  "latest_ver": "초기 릴리즈 그대로",           "update_freq": "개발 중단 (2023 이후 업데이트 없음)"},
    "spark_tts":          {"first_release": "2025-02",  "latest_ver": "0.5B",                         "update_freq": "마지막 업데이트 2025.02"},
    "index_tts2":         {"first_release": "2025-09",  "latest_ver": "IndexTTS-2",                   "update_freq": "수개월 간격 업데이트"},
    "mio_tts_1.7b":       {"first_release": "2026-01",  "latest_ver": "0.1B~2.6B 라인업",            "update_freq": "수개월 간격 신버전"},
    "mio_tts_2.6b":       {"first_release": "2026-01",  "latest_ver": "0.1B~2.6B 라인업",            "update_freq": "수개월 간격 신버전"},
    "glm_tts":            {"first_release": "2025-12",  "latest_ver": "v1.0 (Base+RL)",               "update_freq": "신규 공개, 이력 미확인"},
    "f5tts":              {"first_release": "2024-10",  "latest_ver": "v1.x (2025)",                  "update_freq": "수개월 간격 업데이트"},
    "chatterbox":         {"first_release": "2025-05",  "latest_ver": "Turbo+Multilingual (2025.09)", "update_freq": "수개월 간격 신버전"},
    "parler":             {"first_release": "2024-08",  "latest_ver": "v1.1+Multilingual (2024.12)",  "update_freq": "마지막 업데이트 2024.12"},
    "styletts2":          {"first_release": "2023-06",  "latest_ver": "초기 릴리즈 그대로",           "update_freq": "학술 프로젝트 종료 (2023 이후 업데이트 없음)"},
    "dia":                {"first_release": "2025-04",  "latest_ver": "Dia-1.6B-0626 (2025.06)",      "update_freq": "수개월 간격 업데이트"},
    "mars5":              {"first_release": "2024-05",  "latest_ver": "v0.4 (2024.07)",               "update_freq": "마지막 업데이트 2024.07"},
    "zipvoice_finetuned": {"first_release": "2025-06",  "latest_ver": "초기 공개 (arXiv 2506.13053)", "update_freq": "신규 공개, 이력 미확인"},
    "voicecraftx":        {"first_release": "2024-04",  "latest_ver": "VoiceCraft-X (2024.11)",       "update_freq": "학술 프로젝트, 마지막 업데이트 2024.11"},
    "maskgct":            {"first_release": "2024-10",  "latest_ver": "Amphion v0.2 (2025.01)",       "update_freq": "수개월 간격 업데이트 (Amphion 툴킷)"},
    "hierspeech":         {"first_release": "2023-07",  "latest_ver": "초기 릴리즈 유지",             "update_freq": "학술 프로젝트, 마지막 업데이트 2023.07"},
    "kani":               {"first_release": "2025-03",  "latest_ver": "초기 공개",                    "update_freq": "신규 공개, 이력 미확인"},
    "llmvox":             {"first_release": "2025-02",  "latest_ver": "초기 공개",                    "update_freq": "학술 프로젝트, 마지막 업데이트 2025.02"},
    "chattts":            {"first_release": "2024-05",  "latest_ver": "v0.2.1 (2024.08)",             "update_freq": "마지막 업데이트 2024.08"},
    "piper_ko":           {"first_release": "2023-06",  "latest_ver": "v1.4.1 (2024.12)",             "update_freq": "수개월 간격 업데이트"},
    "zipvoice":           {"first_release": "2025-06",  "latest_ver": "초기 공개 (arXiv 2506.13053)", "update_freq": "신규 공개, 이력 미확인"},
    "supertonic":         {"first_release": "2025-XX",  "latest_ver": "Supertonic v2",                "update_freq": "테스트 예정"},
    "chatterbox_multilingual": {"first_release": "2025-09", "latest_ver": "Multilingual (2025.09)",   "update_freq": "최신 다국어 버전"},
    "fish_speech_s2":          {"first_release": "2026-03", "latest_ver": "S2 Pro (2026.03.10)",       "update_freq": "신규 공개"},
    "higgs_audio":             {"first_release": "2025-XX",  "latest_ver": "V2.5",                     "update_freq": "테스트 예정"},
    "gpt_sovits_v4":           {"first_release": "2025-XX",  "latest_ver": "V4 (48kHz)",               "update_freq": "테스트 예정"},
}

# ─── 평가 특이사항 (Anomalies) ───────────────────────────────────────────────
# severity: "high" = 완전 실패 (환각/레퍼런스 재생), "med" = 부분 실패 또는 지표 왜곡
MODEL_ANOMALIES = {
    "ko|cosyvoice2":  {"model_name": "CosyVoice2",   "severity": "high",
                       "note": "레퍼런스 오디오 내용 재생성 (\"한글자막 by 한효정\", \"다음 영상에서 만나요\") — 타겟 텍스트 무시"},
    "ko|cosyvoice3":  {"model_name": "CosyVoice3",   "severity": "high",
                       "note": "레퍼런스 오디오 내용 재생성 — 타겟 텍스트 무시"},
    "ko|gpt_sovits":  {"model_name": "GPT-SoVITS",   "severity": "med",
                       "note": "참조 음성 언어(EN) 내용이 앞에 삽입 후 한국어 출력 — Prefix Hallucination"},
    "ko|xtts":        {"model_name": "XTTS-v2",       "severity": "med",
                       "note": "단문(ko_short)에서 품질 저하, CER 비정상 높음 (짧은 입력 취약)"},
    "en|cosyvoice2":  {"model_name": "CosyVoice2",   "severity": "high",
                       "note": "\"Thank you for watching (시청해 주셔서 감사합니다)\" 등 레퍼런스 오디오 내용 출력 — 영어 텍스트 처리 실패"},
    "en|cosyvoice3":  {"model_name": "CosyVoice3",   "severity": "high",
                       "note": "레퍼런스 오디오 내용 출력 — 타겟 텍스트 무시"},
    "en|dia":         {"model_name": "Dia-1.6B",      "severity": "med",
                       "note": "[S1]/[S2] 2화자 포맷으로 인해 WER 지표 왜곡 — 실제 음성 품질과 무관"},
    "en|mars5":       {"model_name": "MARS5-TTS",     "severity": "med",
                       "note": "일부 텍스트에서 환각(Hallucination) — 팟캐스트 내용 등 무관한 텍스트 출력"},
    "ja|cosyvoice2":  {"model_name": "CosyVoice2",   "severity": "high",
                       "note": "레퍼런스 오디오 내용 출력 (\"ご視聴ありがとうございました\" = 시청해 주셔서 감사합니다) — 타겟 텍스트 무시"},
    "ja|cosyvoice3":  {"model_name": "CosyVoice3",   "severity": "high",
                       "note": "레퍼런스 오디오 내용 출력 — 타겟 텍스트 무시"},
    "ja|gpt_sovits":  {"model_name": "GPT-SoVITS",   "severity": "med",
                       "note": "일본어 처리 불안정, 일부 텍스트에서 환각 발생"},
    "ja|kokoro":      {"model_name": "Kokoro-82M",    "severity": "med",
                       "note": "ja_short에서 \"日本語 (일본어)\" 단어 무한 반복 루프 → CER=3.249 (수치 왜곡, 실제 품질과 무관)"},
    "zh|cosyvoice2":  {"model_name": "CosyVoice2",   "severity": "high",
                       "note": "레퍼런스 오디오 내용 출력 (\"请不吝点赞 订阅...\" = 좋아요·구독 부탁드립니다...) — 타겟 텍스트 무시"},
    "zh|cosyvoice3":  {"model_name": "CosyVoice3",   "severity": "high",
                       "note": "레퍼런스 오디오 내용 출력 — 타겟 텍스트 무시"},
    "zh|f5tts":       {"model_name": "F5-TTS",        "severity": "med",
                       "note": "zh_short에서 빈 출력 → CER=1.0 (짧은 중국어 입력 처리 실패)"},
    "zh|gpt_sovits":  {"model_name": "GPT-SoVITS",   "severity": "med",
                       "note": "zh_short에서 환각 발생 → CER=2.5 (무관한 텍스트 출력)"},
    "zh|kokoro":      {"model_name": "Kokoro-82M",    "severity": "high",
                       "note": "중국어 미지원 — \"中文字幕志愿者 (중국어 자막 자원봉사자)\" 워터마크 텍스트 출력"},
}


def load_data(results_dir):
    averages_path = os.path.join(results_dir, "averages.json")
    jsonl_path    = os.path.join(results_dir, "detailed_metrics.jsonl")

    averages = {}
    if os.path.exists(averages_path):
        with open(averages_path, encoding="utf-8") as f:
            averages = json.load(f)

    entries = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    return averages, entries


def build_audio_map(entries, results_dir, output_path):
    """{ lang: { text_key: { model_key: rel_wav_path } } }
    output_path 기준 상대경로로 계산 (HTML 저장 위치에 상관없이 정확한 경로)"""
    audio = {}
    out_dir = os.path.dirname(os.path.abspath(output_path))
    abs_results = os.path.abspath(results_dir)
    bench_root = os.path.dirname(abs_results)

    for e in entries:
        if not e.get("success") or not e.get("wav_path"):
            continue
        # 오디오 재생은 첫 번째 run만 사용
        if e.get("run_index", 0) != 0:
            continue
            
        lang = e["lang"]
        tk   = e["text_key"]
        mk   = e["model_key"]
        rk   = e.get("ref_key", "default")
        method = e.get("method", "A")
        
        # Method B(교차언어)도 표시하려면 필터 완화
        if method not in ("A", "B", "C"):
            continue
            
        wav = e["wav_path"]
        try:
            if not os.path.isabs(wav):
                abs_wav = os.path.normpath(os.path.join(bench_root, wav))
            else:
                abs_wav = wav
            rel = os.path.relpath(abs_wav, out_dir).replace("\\", "/")
        except Exception:
            rel = wav.replace("\\", "/")
        
        # 모델|방식|참조음성 키를 사용하여 중복 방지
        m_key = f"{mk}|{method}|{rk}"
        audio.setdefault(lang, {}).setdefault(tk, {})[m_key] = rel
    return audio

LANG_LABELS = {
    "ko": "🇰🇷 한국어 (Korean)",
    "en": "🇺🇸 영어 (English)",
    "ja": "🇯🇵 일본어 (Japanese)",
    "zh": "🇨🇳 중국어 (Chinese)",
}

ARCH_COLORS = {
    "LLM": "#8b5cf6",
    "NAR": "#f59e0b",
    "Flow": "#14b8a6",
    "FM": "#14b8a6",
    "?": "#64748b"
}

TEXT_LABELS = {
    # KO
    "ko_short":          "단문",
    "ko_medium":         "중문",
    "ko_long":           "장문",
    "ko_formal":         "격식체",
    "ko_conversational": "구어체",
    "ko_technical":      "기술용어",
    "ko_numbers":        "숫자/날짜",
    "ko_mixed":          "영단어혼합",
    "ko_question":       "질문/감탄",
    "ko_proverb":        "속담/관용구",
    # EN
    "en_short":          "단문",
    "en_medium":         "중문",
    "en_long":           "장문",
    "en_formal":         "격식체",
    "en_conversational": "구어체",
    "en_technical":      "기술용어",
    "en_numbers":        "숫자/날짜",
    "en_names":          "고유명사",
    "en_punctuation":    "구두점",
    "en_emotional":      "감정표현",
    # JA
    "ja_short":          "단문",
    "ja_medium":         "중문",
    "ja_long":           "장문",
    "ja_formal":         "격식체",
    "ja_conversational": "구어체",
    "ja_technical":      "기술용어",
    # ZH
    "zh_short":          "단문",
    "zh_medium":         "중문",
    "zh_long":           "장문",
    "zh_formal":         "격식체",
    "zh_conversational": "구어체",
    "zh_technical":      "기술용어",
    # legacy keys
    "ko_news": "격식체(뉴스)",
    "ko_conv": "구어체",
    "ko_num": "숫자/날짜",
    "ko_eng": "영단어혼합",
    "ko_emo": "질문/감탄",
    "ko_tech": "기술용어",
    "ko_dial": "사투리/억양",
    "en_conv": "구어체",
    "en_tech": "기술용어",
    "en_num": "숫자포함",
    "en_emo": "감정표현",
    "en_name": "고유명사",
    "en_punct": "구두점",
    "en_news": "격식체",
    "en_dial": "방언/억양",
    "ja_keigo": "경어체",
    "ja_conv": "구어체",
    "ja_tech": "기술용어",
    "zh_conv": "구어체",
    "zh_tech": "기술용어",
}

SPEAKER_ORDER = ["iu_long", "en_female", "ja_female", "zh_female", "default", "preset"]
SPEAKER_DISPLAY = {
    "iu_long": "IU (KO 여성)",
    "en_female": "EN 여성",
    "ja_female": "JA 여성",
    "zh_female": "ZH 여성",
    "default": "기본 화자",
    "preset": "프리셋"
}

def build_ref_audio_map(output_path):
    """모든 레퍼런스 WAV를 HTML 위치 기준 상대경로로 반환"""
    out_dir = os.path.dirname(os.path.abspath(output_path))
    result = {}
    if not os.path.exists(REF_DIR):
        return result
        
    for f in os.listdir(REF_DIR):
        if f.endswith(".wav"):
            abs_wav = os.path.join(REF_DIR, f)
            ref_key = f.replace(".wav", "")
            result[ref_key] = os.path.relpath(abs_wav, out_dir).replace("\\", "/")
    return result


def _streaming_badge(val) -> str:
    val = val or "미확인"
    cfg = {
        "지원":    ("✅ 지원",    "#dcfce7", "#16a34a"),
        "부분 지원":("⚡ 부분",   "#fef9c3", "#92400e"),
        "미지원":  ("✗ 미지원",  "#fee2e2", "#b91c1c"),
        "미확인":  ("? 미확인",  "#f1f5f9", "#64748b"),
    }
    label, bg, fg = cfg.get(val, ("? 미확인", "#f1f5f9", "#64748b"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 6px;'
            f'border-radius:10px;font-size:10px;font-weight:600;white-space:nowrap">'
            f'{label}</span>')


def rtf_color(rtf):
    if rtf < 0: return "#94a3b8"
    if rtf < 0.2: return "#22c55e"
    if rtf < 0.5: return "#84cc16"
    if rtf < 1.0: return "#f59e0b"
    if rtf < 2.0: return "#f97316"
    return "#ef4444"


def cer_color(cer):
    if cer <= 0.02: return "#22c55e"
    if cer <= 0.10: return "#84cc16"
    if cer <= 0.30: return "#f59e0b"
    return "#ef4444"


def load_speaker_averages(results_dir):
    """speaker_averages.json → {model_key|lang|method|ref_key: entry} 로드"""
    path = os.path.join(results_dir, "speaker_averages.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_cer_scores():
    """quality/cer_multilingual.json에서 {lang|model_key: avg_cer} 로드"""
    cer_path = os.path.join(INFRA_DIR, "quality", "cer_multilingual.json")
    if not os.path.exists(cer_path):
        return {}
    with open(cer_path, encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for k, v in raw.items():
        if k.endswith("|AVG"):
            parts = k.split("|")  # lang|model_key|AVG
            if len(parts) == 3:
                lang, mk = parts[0], parts[1]
                full_key = f"{lang}|{mk}"
                result[full_key] = v
                if "__" in mk:
                    base_mk = mk.split("__")[0]
                    base_key = f"{lang}|{base_mk}"
                    if base_key not in result:
                        result[base_key] = v
    return result


def load_sim_scores():
    """quality/sim_multilingual.json에서 {lang|model_key: avg_sim} 로드"""
    sim_path = os.path.join(INFRA_DIR, "quality", "sim_multilingual.json")
    if not os.path.exists(sim_path):
        return {}
    with open(sim_path, encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for k, v in raw.items():
        if k.endswith("|AVG"):
            parts = k.split("|")  # lang|model_key|AVG
            if len(parts) == 3:
                lang, mk = parts[0], parts[1]
                full_key = f"{lang}|{mk}"
                result[full_key] = v
                if "__" in mk:
                    base_mk = mk.split("__")[0]
                    base_key = f"{lang}|{base_mk}"
                    if base_key not in result:
                        result[base_key] = v
    return result


def build_methodology_section():
    return """
  <div class="methodology-section">
    <details>
      <summary>📋 측정 방법론 — 이 보고서는 어떻게 만들어졌나요?</summary>
      <div class="meth-body">

        <div class="meth-grid">

          <div class="meth-card">
            <div class="meth-title">🖥️ 테스트 환경</div>
            <ul>
              <li><strong>GPU</strong>: NVIDIA RTX 5080 (16GB VRAM)</li>
              <li><strong>OS</strong>: Windows 11 + WSL2 / Git Bash</li>
              <li><strong>측정 도구</strong>: Python subprocess 격리 실행 — 모델별 독립 venv/conda에서 단독 실행하여 GPU 간섭 없음</li>
              <li><strong>VRAM 측정</strong>: <code>torch.cuda.max_memory_allocated()</code> (추론 직전 리셋 후 피크 기록)</li>
            </ul>
          </div>

          <div class="meth-card">
            <div class="meth-title">📝 테스트 텍스트 구성</div>
            <ul>
              <li><strong>한국어(KO)</strong>: 10개 — 짧은/중간/긴 문장, 숫자/날짜, 영한 혼합, 격식체, 구어체, 기술문, 질문/감탄, 속담</li>
              <li><strong>영어(EN)</strong>: 10개 — 짧은/중간/긴 문장, 숫자, 기술문, 구어체, 격식체, 구두점, 고유명사, 감정</li>
              <li><strong>일본어(JA)</strong>: 5개 — 짧은/중간 문장, 경어체, 구어체, 기술문</li>
              <li><strong>중국어(ZH)</strong>: 5개 — 짧은/중간 문장, 격식체, 구어체, 기술문</li>
            </ul>
          </div>

          <div class="meth-card">
            <div class="meth-title">🎙️ 참조 음성 (Voice Cloning)</div>
            <ul>
              <li><strong>한국어</strong>: iu_long.wav — 한국 여성 화자 (~29초)</li>
              <li><strong>영어</strong>: en_female.wav — 영어 여성 화자</li>
              <li><strong>일본어</strong>: ja_female.wav — 일본 여성 화자</li>
              <li><strong>중국어</strong>: zh_female.wav — 중국 여성 화자</li>
              <li>음성 클론을 지원하지 않는 모델(MeloTTS, Kokoro 등)은 모델 내장 기본 화자 사용</li>
            </ul>
          </div>

          <div class="meth-card">
            <div class="meth-title">📊 측정 지표</div>
            <ul>
              <li><strong>RTF (Real-Time Factor)</strong>: 추론 시간 ÷ 생성된 오디오 길이. 낮을수록 빠름. RTF &lt; 1.0 = 실시간보다 빠름</li>
              <li><strong>VRAM</strong>: 추론 중 최대 GPU 메모리 사용량(MB)</li>
              <li><strong>TTFA</strong>: 첫 오디오 출력까지 걸린 시간(ms) — 스트리밍 지원 모델에서 의미 있는 지표</li>
              <li><strong>CER / WER</strong>: Whisper large-v3로 생성된 음성을 재인식 → 원문과 Levenshtein 거리 비교. 한국어·일본어·중국어 = CER(문자), 영어 = WER(단어)</li>
            </ul>
          </div>

          <div class="meth-card">
            <div class="meth-title">🏆 순위 기준</div>
            <ul>
              <li>모델 상세 테이블 및 음성 비교 섹션은 <strong>avg RTF 오름차순</strong> (빠른 모델이 위)</li>
              <li>RTF는 해당 언어의 모든 테스트 텍스트에 대한 평균값</li>
              <li>RTF만으로 "좋은 모델"을 판단하지 않도록 CER·VRAM·음성 청취 비교를 함께 제공</li>
            </ul>
          </div>

          <div class="meth-card">
            <div class="meth-title">⚠️ 한계 및 주의사항</div>
            <ul>
              <li>RTF는 최초 모델 로딩(warm-up) 시간 제외 — 실제 서비스 첫 요청은 더 느릴 수 있음</li>
              <li>CER/WER는 Whisper의 STT 오류도 포함될 수 있어 절대 정확도가 아님</li>
              <li>VRAM은 단일 추론 기준 — 배치 처리나 병렬 요청 시 증가</li>
              <li>ZipVoice-FT는 REST API(Docker) 방식으로 측정하여 VRAM 수치 없음</li>
            </ul>
          </div>

        </div>
      </div>
    </details>
  </div>
"""


def build_anomalies_section():
    """MODEL_ANOMALIES 딕셔너리를 HTML 테이블로 변환"""
    lang_order = ["ko", "en", "ja", "zh"]
    lang_label = {"ko": "한국어", "en": "영어", "ja": "일본어", "zh": "중국어"}
    severity_label = {"high": "심각", "med": "경미"}
    rows = []
    for lang in lang_order:
        for key, info in MODEL_ANOMALIES.items():
            k_lang, k_model = key.split("|", 1)
            if k_lang != lang:
                continue
            
            # Filter for LLM only
            m_info = MODEL_INFO.get(k_model, {})
            if m_info.get("arch_type") != "LLM":
                continue
                
            sev = info["severity"]
            sev_cls = f"severity-{sev}"
            sev_txt = severity_label.get(sev, sev)
            note = info["note"]
            rows.append(
                f'<tr>'
                f'<td><span class="lang-badge lang-{lang}">{lang_label[lang]}</span></td>'
                f'<td style="font-weight:600;color:#fff">{info["model_name"]}</td>'
                f'<td><span class="{sev_cls}">{sev_txt}</span></td>'
                f'<td style="color:#cbd5e1">{note}</td>'
                f'</tr>'
            )
    rows_html = "\n      ".join(rows)
    return f"""
  <div class="anomalies-section">
    <h2>⚠️ 평가 특이사항 (Anomalies)</h2>
    <p class="desc">평가 과정에서 발견된 비정상 동작 및 CER/WER 수치 왜곡 요인 — 심각: 타겟 텍스트 미출력(완전 실패), 경미: 일부 텍스트 실패 또는 지표 왜곡</p>
    <table class="anomaly-table">
      <thead>
        <tr>
          <th>언어</th><th>모델</th><th>심각도</th><th>특이사항</th>
        </tr>
      </thead>
      <tbody>
      {rows_html}
      </tbody>
    </table>
    <p style="margin-top:12px;color:var(--text2);font-size:12px;">* CER &gt; 1.0은 수학적으로 가능합니다 — 가설 텍스트가 참조보다 훨씬 길 때 (무한 반복, 환각 현상 등).</p>
  </div>"""

def build_overview_section(averages, cer_scores, sim_scores):
    """카테고리별 1위 모델 + 전체 비교 매트릭스"""
    # ── 모델별 종합 스탯 수집 ───────────────────────────────────────────────
    models_summary = {}
    for gk, v in averages.items():
        if v.get("method", "A") != "A": continue
        lang = v["lang"]
        mk = v["model_key"]
        
        pending_names = {m["model"] for m in EXCLUDED_MODELS.get("테스트 진행 예정 (후보)", [])}
        info = MODEL_INFO.get(mk, {})
        dname = info.get("display_name", mk)
        if dname in pending_names:
            continue
            
        if mk not in models_summary:
            models_summary[mk] = {
                "name": info.get("display_name", mk),
                "arch": info.get("arch_type", "?"),
                "developer": info.get("developer", ""),
                "license": info.get("license", "-"),
                "langs": info.get("langs", []),
                "ko_rtf": None, "en_rtf": None, "ja_rtf": None, "zh_rtf": None,
                "ko_cer": None, "en_cer": None,
                "vram": None,
            }
        rtf = v.get("avg_rtf", -1)
        vram = v.get("avg_vram_peak_mb", -1)
        if rtf > 0:
            models_summary[mk][f"{lang}_rtf"] = rtf
        if vram > 0:
            prev = models_summary[mk]["vram"]
            models_summary[mk]["vram"] = int(min(vram, prev) if prev else vram)

    for mk in models_summary:
        for lng in ["ko", "en"]:
            c = cer_scores.get(f"{lng}|{mk}")
            if c is not None:
                models_summary[mk][f"{lng}_cer"] = c

    def _rtf_col(v): return rtf_color(v if v is not None else -1)
    def _cer_col(v): return "#22c55e" if v and v <= 0.02 else ("#84cc16" if v and v <= 0.10 else ("#f59e0b" if v and v <= 0.30 else "#ef4444"))

    # ── 1위 계산 + 렌더링 헬퍼 ────────────────────────────────────────────────
    def _best(key, models_dict, lower_is_better=True):
        valid = [(mk, d) for mk, d in models_dict.items() if d.get(key) is not None]
        if not valid: return None, None
        fn = min if lower_is_better else max
        mk, d = fn(valid, key=lambda x: x[1][key])
        return mk, d

    def _compute_winners(models_dict):
        ko_spd_mk, ko_spd = _best("ko_rtf", models_dict)
        en_spd_mk, en_spd = _best("en_rtf", models_dict)
        ko_acc_mk, ko_acc = _best("ko_cer", models_dict)
        en_acc_mk, en_acc = _best("en_cer", models_dict)
        balanced = {mk: d for mk, d in models_dict.items()
                    if d["ko_rtf"] and d["ko_rtf"] < 1.0 and d["en_rtf"] and d["en_rtf"] < 1.0}
        bal_mk, _ = _best("ko_cer", {mk: dict(d, ko_cer=(d.get("ko_cer") or 1.0) + (d.get("en_cer") or 1.0))
                                      for mk, d in balanced.items()}) if balanced else (None, None)
        bal_d = models_dict.get(bal_mk) if bal_mk else None
        multilang = {mk: d for mk, d in models_dict.items()
                     if all(d.get(f"{l}_rtf") for l in ["ko","en","ja","zh"])}
        ml_mk, ml_d = _best("ko_rtf", multilang) if multilang else (None, None)
        return dict(ko_spd_mk=ko_spd_mk, ko_spd=ko_spd, en_spd_mk=en_spd_mk, en_spd=en_spd,
                    ko_acc_mk=ko_acc_mk, ko_acc=ko_acc, en_acc_mk=en_acc_mk, en_acc=en_acc,
                    bal_mk=bal_mk, bal_d=bal_d, ml_mk=ml_mk, ml_d=ml_d)

    def _render_winners_block(w):
        ko_spd_mk = w["ko_spd_mk"]; ko_spd = w["ko_spd"]
        en_spd_mk = w["en_spd_mk"]; en_spd = w["en_spd"]
        ko_acc_mk = w["ko_acc_mk"]; ko_acc = w["ko_acc"]
        en_acc_mk = w["en_acc_mk"]; en_acc = w["en_acc"]
        bal_mk = w["bal_mk"];       bal_d  = w["bal_d"]
        ml_mk  = w["ml_mk"];        ml_d   = w["ml_d"]

        items = []
        if ko_spd_mk:
            items.append(f'<div class="tldr-item"><span class="tldr-label">⚡ 한국어 속도</span><span class="tldr-model">{ko_spd["name"]}</span><span class="tldr-val" style="color:#4ade80">RTF {ko_spd["ko_rtf"]:.3f}</span></div>')
        if en_spd_mk:
            items.append(f'<div class="tldr-item"><span class="tldr-label">⚡ 영어 속도</span><span class="tldr-model">{en_spd["name"]}</span><span class="tldr-val" style="color:#4ade80">RTF {en_spd["en_rtf"]:.3f}</span></div>')
        if ko_acc_mk:
            acc_v = ko_acc["ko_cer"]
            items.append(f'<div class="tldr-item"><span class="tldr-label">🎯 한국어 정확도</span><span class="tldr-model">{ko_acc["name"]}</span><span class="tldr-val" style="color:#60a5fa">CER {acc_v*100:.1f}%</span></div>')
        if bal_mk:
            items.append(f'<div class="tldr-item"><span class="tldr-label">⚖️ KO+EN 균형</span><span class="tldr-model">{bal_d["name"]}</span><span class="tldr-val" style="color:#fb923c">RTF {bal_d["ko_rtf"]:.3f}/{bal_d["en_rtf"]:.3f}</span></div>')
        if ml_mk:
            items.append(f'<div class="tldr-item"><span class="tldr-label">🌐 4개 언어</span><span class="tldr-model">{ml_d["name"]}</span><span class="tldr-val" style="color:#a78bfa">KO+EN+JA+ZH</span></div>')
        tldr = f'<div class="tldr-bar"><span class="tldr-head">결론</span>{"".join(items)}</div>'

        def card(emoji, title, mk, d, hkey, hlabel, cls="card-speed"):
            if mk is None: return ""
            arch = d["arch"]; color = ARCH_COLORS.get(arch, "#64748b")
            hval = d.get(hkey)
            hval_s = (f"{hval:.3f}" if isinstance(hval, float) and hkey.endswith("rtf")
                      else f"{hval*100:.1f}%" if isinstance(hval, float) else "—")
            return f'''<div class="winner-card {cls}">
              <div class="winner-emoji-title"><span class="winner-emoji">{emoji}</span><span class="winner-title">{title}</span></div>
              <div class="winner-model-row"><span class="arch-badge" style="background:{color}">{arch}</span><strong class="winner-name">{d["name"]}</strong></div>
              <div class="winner-dev">{d["developer"]} &nbsp;·&nbsp; {d["license"]}</div>
              <div class="winner-metric"><span class="winner-metric-label">{hlabel}</span><strong class="winner-metric-val">{hval_s}</strong></div>
            </div>'''

        cards = "".join([
            card("⚡", "KO 최고 속도",       ko_spd_mk, ko_spd, "ko_rtf", "KO RTF",  "card-speed"),
            card("⚡", "EN 최고 속도",       en_spd_mk, en_spd, "en_rtf", "EN RTF",  "card-speed"),
            card("🎯", "KO 발음 정확도 1위", ko_acc_mk, ko_acc, "ko_cer", "KO CER",  "card-accuracy"),
            card("🎯", "EN 발음 정확도 1위", en_acc_mk, en_acc, "en_cer", "EN WER",  "card-accuracy"),
            card("⚖️", "KO+EN 균형 최고",   bal_mk,   bal_d,  "ko_rtf", "KO RTF",  "card-balance"),
        ])
        return tldr + f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin-bottom:20px;">{cards}</div>'

    # 전체 + LLM/FM 두 버전 렌더링
    block_all = _render_winners_block(_compute_winners(models_summary))
    llm_models = {mk: d for mk, d in models_summary.items() if d["arch"] in ("LLM", "FM")}
    block_llm = _render_winners_block(_compute_winners(llm_models))

    # ── 전체 모델 매트릭스 테이블 (기본 접힘) ────────────────────────────────
    all_sorted = sorted(models_summary.items(),
                        key=lambda x: x[1]["ko_rtf"] if x[1]["ko_rtf"] else 999)
    matrix_rows = []

    def rtf_td(v):
        if v is None: return '<td class="ov-num" style="color:#475569">—</td>'
        return f'<td class="ov-num" style="color:{_rtf_col(v)};font-weight:700">{v:.3f}</td>'
    def cer_td(v, suffix="CER"):
        if v is None: return '<td class="ov-num" style="color:#475569">—</td>'
        return f'<td class="ov-num" style="color:{_cer_col(v)};font-weight:700">{v*100:.1f}%</td>'

    for idx, (mk, d) in enumerate(all_sorted, 1):
        arch = d["arch"]
        color = ARCH_COLORS.get(arch, "#64748b")
        vram_s = f"{d['vram']}MB" if d["vram"] else "—"
        langs_html = " ".join(f'<span class="lang-badge lang-{l}" style="margin-right:2px">{l.upper()}</span>' for l in d.get("langs", []))
        matrix_rows.append(
            f'<tr data-arch="{arch}">'
            f'<td style="text-align:center; color:#94a3b8; font-size:12px; font-weight:700;">{idx}</td>'
            f'<td class="ov-name"><strong>{d["name"]}</strong></td>'
            f'<td style="text-align:center"><span class="arch-badge sm" style="background:{color}">{arch}</span></td>'
            f'<td style="white-space:nowrap; padding:4px;">{langs_html}</td>'
            + rtf_td(d["ko_rtf"]) + rtf_td(d["en_rtf"])
            + rtf_td(d["ja_rtf"]) + rtf_td(d["zh_rtf"])
            + cer_td(d["ko_cer"], "CER") + cer_td(d["en_cer"], "WER")
            + f'<td class="ov-num" style="color:#94a3b8">{vram_s}</td>'
            + f'</tr>'
        )

    # ── 제외 모델 행 추가 ─────────────────────────────────────────────────────
    excl_rows = []
    excl_start = len(all_sorted) + 1  # 벤치마크 번호 이어서
    for excl_idx, (cat, models_list) in enumerate(
            [(cat, m) for cat, ml in EXCLUDED_MODELS.items() for m in ml], excl_start):
        cat, m = cat, models_list  # unpack
        # MODEL_INFO에서 상세 정보 조회 (display_name 역방향 매핑)
        mk_excl = next((k for k, v in MODEL_INFO.items()
                        if v.get("display_name") == m["model"]), None)
        info_excl = MODEL_INFO.get(mk_excl, {}) if mk_excl else {}
        arch_excl = info_excl.get("arch_type", "—")
        color_excl = ARCH_COLORS.get(arch_excl, "#94a3b8")
        langs_excl = info_excl.get("langs", [])
        official_s = info_excl.get("official_summary", "")
        arch_badge = (f'<span class="arch-badge sm" style="background:{color_excl};opacity:.6">{arch_excl}</span> '
                      if arch_excl != "—" else "")
        langs_html_excl = (" ".join(
            f'<span class="lang-badge lang-{l}" style="margin-right:2px;opacity:.6">{l.upper()}</span>'
            for l in langs_excl) if langs_excl else
            f'<span style="color:#475569;font-size:11px">{m["lic"]}</span>')
        reason_cell = (f'<td class="ov-num" colspan="6" style="color:#b91c1c;font-size:11px;text-align:left;padding-left:8px">'
                       f'🚫 {m["reason"]}'
                       + (f' <span style="color:#94a3b8;font-size:10px">| {official_s}</span>' if official_s else "")
                       + '</td>')
        excl_rows.append(
            f'<tr data-arch="{arch_excl}" style="opacity:.7;background:#fafafa;">'
            f'<td style="text-align:center;color:#94a3b8;font-size:11px;font-weight:700">{excl_idx}</td>'
            f'<td class="ov-name" style="color:var(--text2);">'
            f'<strong>{m["model"]}</strong>'
            f'<span style="color:var(--text2);font-size:10px;margin-left:4px">{m["org"]}</span>'
            f'</td>'
            f'<td style="text-align:center">{arch_badge}</td>'
            f'<td style="white-space:nowrap;padding:4px;">{langs_html_excl}</td>'
            + reason_cell
            + f'<td class="ov-num" style="color:var(--text2)">—</td>'
            f'</tr>'
        )

    # 구분 행
    sep_row = (f'<tr style="background:rgba(99,102,241,.08)">'
               f'<td colspan="11" style="text-align:center;padding:6px 8px;font-size:11px;'
               f'color:#7c85a2;font-weight:600;letter-spacing:.5px;">'
               f'── 아래는 벤치마크 제외 모델 ({len(excl_rows)}개) ──'
               f'</td></tr>')

    total_cnt = len(all_sorted) + len(excl_rows)
    matrix_html = f'''
    <details class="matrix-details">
      <summary>📊 전체 모델 목록 ({len(all_sorted)}개 벤치마크 + {len(excl_rows)}개 제외, KO RTF순) — 클릭해서 펼치기</summary>
      <div style="overflow-x:auto;margin-top:12px">
      <table class="ov-table" style="min-width:700px">
        <thead><tr>
          <th style="width:30px; text-align:center;">#</th>
          <th style="text-align:left;min-width:160px">모델</th>
          <th>구조</th>
          <th>지원 언어</th>
          <th>KO RTF</th><th>EN RTF</th><th>JA RTF</th><th>ZH RTF</th>
          <th>KO CER</th><th>EN WER</th><th>VRAM</th>
        </tr></thead>
        <tbody>{"".join(matrix_rows)}{sep_row}{"".join(excl_rows)}</tbody>
      </table>
      </div>
    </details>'''

    return f'''
    <div class="overview-section" style="margin-bottom: 36px;">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;padding-bottom:12px;border-bottom:2px solid var(--border);">
        <span style="font-size:26px">🏆</span>
        <div>
          <h2 style="font-size:22px;font-weight:800;color:var(--text);letter-spacing:-.4px;margin:0;">한눈에 보기</h2>
          <p style="color:var(--text2);font-size:13px;margin:2px 0 0;">카테고리별 1위 모델 요약. 상세 순위와 음성 비교는 아래 탭에서 확인하세요.</p>
        </div>
      </div>
      <div id="overview-winners-all">{block_all}</div>
      <div id="overview-winners-llm" style="display:none">{block_llm}</div>
      {matrix_html}
    </div>
    '''

def generate_html(results_dir, output_path):
    averages, entries = load_data(results_dir)
    
    # --- LLM Focus (No filter needed as per user request) ---
    # averages = { gk: v for gk, v in averages.items() if ... }
    # ---------------------------------
    
    audio_map = build_audio_map(entries, results_dir, output_path)
    ref_map = build_ref_audio_map(output_path)
    text_content_js = json.dumps(TEST_TEXTS, ensure_ascii=False)
    cer_scores = load_cer_scores()
    sim_scores = load_sim_scores()
    spk_avgs = load_speaker_averages(results_dir)

    # 언어별 데이터 그룹화
    lang_data = {}
    for gk, v in averages.items():
        lang = v["lang"]
        method = v.get("method", "A")
        # Whisper CER/WER 주입
        ck = f"{lang}|{v['model_key']}"
        if ck in cer_scores:
            v["avg_cer"] = cer_scores[ck]
        
        if ck in sim_scores:
            v["avg_sim"] = sim_scores[ck]
        
        # Method A, B, C를 전부 포함하되 우선순위 정렬 등 필요시 처리
        lang_data.setdefault(lang, []).append(v)

    for lang in lang_data:
        lang_data[lang].sort(key=lambda x: x["avg_rtf"] if x["avg_rtf"] > 0 else 999)

    available_langs = [l for l in ["ko","en","ja","zh"] if l in lang_data]
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── 언어 탭 HTML ─────────────────────────────────────────────────────────
    def tab_buttons():
        btns = []
        for i, lang in enumerate(available_langs):
            active = "active" if i == 0 else ""
            label = LANG_LABELS.get(lang, lang.upper())
            btns.append(f'<button class="tab-btn {active}" onclick="switchLang(\'{lang}\')" id="btn-{lang}">{label}</button>')
        return "\n    ".join(btns)

    # ── 언어별 콘텐츠 ─────────────────────────────────────────────────────────
    # 1. Rankings Table (오디오 통합)
    def rankings_table(lang, audio_lang, ref_src):
        p_cfg = PIVOT_CONFIG.get(lang, {"tk": "default", "rk": "default"})
        pivot_tk = p_cfg["tk"]
        pivot_rk = p_cfg["rk"]
        pivot_text = TEST_TEXTS.get(pivot_tk, pivot_tk)
        spk_label = SPEAKER_DISPLAY.get(pivot_rk, pivot_rk)

        rows = []
        # 테스트 미완료 모델만 제외 (averages.json에 데이터 있는 모델은 모두 표시)
        pending_names = {m["model"] for m in EXCLUDED_MODELS.get("테스트 진행 예정 (후보)", [])}
        lang_data_filtered = []
        for v in averages.values():
            if v["lang"] == lang and v.get("method", "A") == "A":
                mk = v["model_key"]
                dname = MODEL_INFO.get(mk, {}).get("display_name", mk)
                if dname not in pending_names:
                    lang_data_filtered.append(v)

        m_list = sorted(lang_data_filtered, key=lambda x: x.get("avg_rtf", 999))

        for i, d in enumerate(m_list):
            mk = d["model_key"]
            info = MODEL_INFO.get(mk, {})
            name = info.get("display_name", mk)
            arch = info.get("arch_type", "?")
            lic  = info.get("license", "-")
            rtf  = d.get("avg_rtf", -1)
            cer  = cer_scores.get(f"{lang}|{mk}", -1)
            rtf_s = f"{rtf:.3f}" if rtf > 0 else "N/A"
            cer_s = f"{cer * 100:.2f}%" if cer >= 0 else "N/A"

            # 오디오 소스 찾기
            pivot_key = f"{mk}|A|{pivot_rk}"
            src = audio_lang.get(pivot_tk, {}).get(pivot_key)
            if not src:
                for k, v in audio_lang.get(pivot_tk, {}).items():
                    if k.split("|")[0] == mk:
                        src = v; break

            # 인라인 오디오 (항상 표시)
            if src:
                inline_audio = (
                    f'<div class="mini-player">'
                    f'<audio preload="none" src="{src}"></audio>'
                    f'<button class="play-btn" onclick="togglePlay(this)" title="재생/일시정지">▶</button>'
                    f'</div>'
                )
            else:
                inline_audio = '<span style="color:var(--text2);font-size:11px">—</span>'

            medals = ["🥇", "🥈", "🥉"]
            rank_display = f"{medals[i]}<br><span style='font-size:10px;color:var(--text2)'>{i+1}</span>" if i < 3 else str(i+1)

            # ── 전 언어 성능 미니 테이블 ──────────────────────────────────
            perf_rows_html = []
            for lng in ["ko", "en", "ja", "zh"]:
                e = averages.get(f"{mk}|{lng}|A", {})
                r2 = e.get("avg_rtf", -1)
                vr = e.get("avg_vram_peak_mb", -1)
                c2 = cer_scores.get(f"{lng}|{mk}")
                if r2 <= 0 and c2 is None:
                    continue
                rtf2_s  = f"{r2:.3f}" if r2 > 0 else "—"
                rtf2_col = rtf_color(r2) if r2 > 0 else "#475569"
                cer2_s  = f"{c2*100:.1f}%" if c2 is not None else "—"
                cer2_col = cer_color(c2) if c2 is not None else "#475569"
                vram2_s = f"{int(vr)}MB" if vr > 0 else "—"
                metric_label = "WER" if lng == "en" else "CER"
                perf_rows_html.append(
                    f'<tr>'
                    f'<td><span class="lang-badge lang-{lng}">{lng.upper()}</span></td>'
                    f'<td style="color:{rtf2_col};font-weight:700">{rtf2_s}</td>'
                    f'<td style="color:{cer2_col};font-weight:700">{cer2_s} <span style="color:var(--text2);font-size:10px">{metric_label}</span></td>'
                    f'<td style="color:#64748b">{vram2_s}</td>'
                    f'</tr>'
                )
            perf_table_html = f"""
              <table class="detail-perf-mini">
                <thead><tr><th>언어</th><th>RTF</th><th>정확도</th><th>VRAM</th></tr></thead>
                <tbody>{"".join(perf_rows_html)}</tbody>
              </table>"""

            # ── 모델 메타데이터 ───────────────────────────────────────────
            rel = MODEL_RELEASE_INFO.get(mk, {})
            arch_detail = info.get("arch_detail", "")
            langs_s = info.get("langs", [])
            lang_badges_html = " ".join(
                f'<span class="lang-badge lang-{l}">{l.upper()}</span>' for l in langs_s)
            opt_items = [c for c in info.get("cons", []) if "미적용" in c]
            opt_note  = info.get("opt_note", "")
            lic_display = info.get("license", "-") or "-"
            meta_html = f"""
              <div class="detail-meta-grid">
                <div><span class="dm-label">아키텍처</span><span class="dm-val">{arch_detail}</span></div>
                <div><span class="dm-label">지원 언어</span><span class="dm-val">{lang_badges_html}</span></div>
                <div><span class="dm-label">라이선스</span><span class="dm-val" style="color:#0ea5e9;font-weight:600">{lic_display}</span></div>
                <div><span class="dm-label">최초 공개</span><span class="dm-val">{rel.get("first_release","—")}</span></div>
                <div><span class="dm-label">최신 버전</span><span class="dm-val">{rel.get("latest_ver","—")}</span></div>
                <div><span class="dm-label">업데이트</span><span class="dm-val">{rel.get("update_freq","—")}</span></div>
                <div><span class="dm-label">스트리밍</span><span class="dm-val">{_streaming_badge(info.get("streaming","미확인"))}</span></div>
                {"".join(f'<div><span class="dm-label">최적화</span><span class="dm-val" style="color:#fbbf24">⚡ {o}</span></div>' for o in opt_items)}
                {f'<div><span class="dm-label">적용됨</span><span class="dm-val" style="color:#34d399">✅ {opt_note}</span></div>' if opt_note else ""}
              </div>"""

            # ── 장단점 ───────────────────────────────────────────────────
            pros_html = "".join(f'<li>{p}</li>' for p in info.get("pros", []))
            cons_only = [c for c in info.get("cons", []) if "미적용" not in c]
            cons_html = "".join(f'<li>{c}</li>' for c in cons_only)

            # ── 공식 성능 ─────────────────────────────────────────────────
            official_items = info.get("official_perf", [])
            paper_url = info.get("paper_url", "")
            if official_items:
                official_li = "".join(f'<li>{o}</li>' for o in official_items)
                src_link = f' <a href="{paper_url}" target="_blank" style="font-size:10px;color:var(--accent);text-decoration:none;">📄 논문/출처 ↗</a>' if paper_url else ""
                official_html = f'<div class="detail-official"><strong>공식 발표 성능{src_link}</strong><ul>{official_li}</ul></div>'
            else:
                official_html = ""

            # ── 공식 성능 요약 (테이블용) ─────────────────────────────────────
            official_summary = info.get("official_summary")
            if official_summary:
                paper_link = f' <a href="{paper_url}" target="_blank" style="color:var(--accent);text-decoration:none;font-size:10px;">↗</a>' if paper_url else ""
                official_cell = f'<span class="official-val">{official_summary}{paper_link}</span>'
            else:
                official_cell = '<span class="official-none">없음</span>'

            vram_raw = d.get("avg_vram_peak_mb", -1)
            vram_s = f"{int(vram_raw)}MB" if vram_raw and vram_raw > 0 else "—"

            row = f"""
            <tr data-arch="{arch}">
                <td class="rank">{rank_display}</td>
                <td class="model-name">
                    <div>{name}</div>
                    <div style="font-size:10px; color:var(--text2);">{info.get("developer","")} &nbsp;·&nbsp; <span style="color:#64748b">{lic}</span></div>
                </td>
                <td><span class="arch-badge" style="background:{ARCH_COLORS.get(arch,'#64748b')}">{arch}</span></td>
                <td>{lang_badges_html}</td>
                <td class="rtf-cell" data-val="{rtf}">
                    <span class="rtf-val" style="color:{rtf_color(rtf)}">{rtf_s}</span>
                    <div class="rtf-bar" style="width:{int(min(100.0, 100.0/max(0.1,rtf)))}%; background:{rtf_color(rtf)}"></div>
                </td>
                <td style="color:#94a3b8; font-size:12px; text-align:center;">{vram_s}</td>
                <td style="text-align:center; font-size:11px;">{_streaming_badge(info.get("streaming","미확인"))}</td>
                <td class="ac-cer" data-val="{cer}">
                    <span style="color:{cer_color(cer)}; font-weight:700;">{cer_s}</span>
                </td>
                <td class="official-cell">{official_cell}</td>
                <td class="audio-cell">{inline_audio}</td>
                <td>
                    <button class="spk-toggle" onclick="toggleSpk('detail-{lang}-{mk}')">상세 ▿</button>
                </td>
            </tr>
            <tr id="detail-{lang}-{mk}" class="spk-detail-row">
                <td colspan="11" class="spk-detail-cell">
                    <div class="detail-full">
                        <div class="detail-section">{perf_table_html}</div>
                        <div class="detail-section">{meta_html}</div>
                        <div class="detail-section detail-pros-cons">
                            <div class="detail-pros"><strong>장점</strong><ul>{pros_html}</ul></div>
                            <div class="detail-cons"><strong>단점</strong><ul>{cons_html}</ul></div>
                            {official_html}
                        </div>
                    </div>
                </td>
            </tr>
            """
            rows.append(row)

        # 레퍼런스 오디오 + 피벗 텍스트 헤더
        ref_box = ""
        if ref_src:
            ref_box = f"""
          <div class="ref-audio-box" style="margin-bottom:10px">
            <span class="ref-label">🎙️ Reference ({spk_label})</span>
            <audio controls preload="none" src="{ref_src}"></audio>
          </div>"""

        lang_label = LANG_LABELS.get(lang, lang.upper())
        return f"""
        <div style="display:flex;align-items:center;gap:10px;margin:28px 0 14px;padding-bottom:10px;border-bottom:1px solid var(--border);">
          <span style="font-size:18px">📊</span>
          <h3 style="font-size:16px;font-weight:700;color:var(--text);margin:0;">{lang_label} 순위</h3>
          <span style="font-size:12px;color:var(--text2);margin-left:4px;">RTF 기준 정렬 · 오디오 직접 재생 가능</span>
        </div>
        {ref_box}
        <div class="text-display" style="margin-bottom:10px">{pivot_text}</div>
        <div class="audio-table-wrap">
          <table class="rankings-table" id="table-{lang}">
            <thead>
              <tr>
                <th class="sortable" onclick="sortTable(this)">#</th>
                <th class="sortable" onclick="sortTable(this)">모델</th>
                <th class="sortable" onclick="sortTable(this)">구조</th>
                <th class="sortable" onclick="sortTable(this)">언어 지원</th>
                <th class="sortable" onclick="sortTable(this)">RTF (속도)</th>
                <th class="sortable" onclick="sortTable(this)">VRAM</th>
                <th>스트리밍</th>
                <th class="sortable" onclick="sortTable(this)">{"WER" if lang == "en" else "CER"} (정확도)</th>
                <th>공식 성능</th>
                <th>🎧 오디오</th>
                <th>상세</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows) if rows else '<tr><td colspan="9" class="no-data">데이터가 없습니다.</td></tr>'}
            </tbody>
          </table>
        </div>
        """


    # ── 언어별 섹션 생성 ──────────────────────────────────────────────────────
    lang_sections = []
    lang_rtf_map = {l: {r["model_key"]: r["avg_rtf"] for r in rows} for l, rows in lang_data.items()}

    for lang in available_langs:
        rows = lang_data.get(lang, [])
        audio_lang = audio_map.get(lang, {})
        ref_src = ref_map.get(lang, "")
        display = "block" if lang == available_langs[0] else "none"

        section = (
            f'  <div class="lang-section" id="section-{lang}" style="display:{display}">\n'
            + rankings_table(lang, audio_lang, ref_src)
            + '\n  </div>'
        )
        lang_sections.append(section)


    # ── 모델 상세 테이블 ──────────────────────────────────────────────────────
    def model_table():
        shown = set()
        for gk, v in averages.items():
            if v.get("method", "A") == "A":
                shown.add(v["model_key"])
        all_mk = sorted(shown or MODEL_INFO.keys())

        excluded_names = {m["model"] for cat, models in EXCLUDED_MODELS.items() for m in models}

        rows = []
        for mk in all_mk:
            info = MODEL_INFO.get(mk, {})
            if not info:
                continue
            dname    = info.get("display_name", mk)
            
            if dname in excluded_names:
                continue
            
            dev      = info.get("developer", "")
            arch     = info.get("arch_type", "?")
            detail   = info.get("arch_detail", "")
            langs_s  = info.get("langs", [])
            pros     = info.get("pros", [])
            cons     = info.get("cons", [])
            arch_col = ARCH_COLORS.get(arch, "#64748b")

            lang_badges = " ".join(
                f'<span class="lang-badge lang-{l}">{l.upper()}</span>'
                for l in langs_s
            )

            # 언어별 RTF 셀 (±std · N 포함)
            rtf_cells = []
            for lng in ["ko", "en", "ja", "zh"]:
                entry = averages.get(f"{mk}|{lng}|A", {})
                r   = entry.get("avg_rtf", -1)
                std = entry.get("std_rtf", 0)
                n   = entry.get("n_texts", 0)
                if r > 0:
                    col = rtf_color(r)
                    bar = min(60, max(3, int(r * 20)))
                    unstable = std > r  # std > RTF → 측정 불안정
                    sub_col  = "#f87171" if unstable else "var(--text2)"
                    sub_tip  = "측정값 불안정 (std > RTF). 재실행 시 결과가 크게 달라질 수 있음." if unstable else ""
                    sub = (f'<div class="mit-sub" style="color:{sub_col}" title="{sub_tip}">±{std:.3f} · N={n}</div>' if std > 0 else
                           f'<div class="mit-sub">N={n}</div>' if n > 0 else '')
                    rtf_cells.append(
                        f'<td data-val="{r}" class="mit-num" title="avg={r:.3f}  std={std:.3f}  N={n}">'
                        f'<span style="color:{col};font-weight:700">{r:.3f}</span>'
                        f'<div style="width:{bar}px;height:3px;background:{col};border-radius:2px;margin-top:2px"></div>'
                        f'{sub}</td>'
                    )
                else:
                    rtf_cells.append('<td data-val="9999" class="mit-na">·</td>')

            # VRAM: 측정된 언어 중 첫 번째 값 사용
            vram_val = next(
                (averages.get(f"{mk}|{lng}|A", {}).get("avg_vram_peak_mb", -1)
                 for lng in ["ko", "en", "ja", "zh"]
                 if averages.get(f"{mk}|{lng}|A", {}).get("avg_vram_peak_mb", -1) > 0),
                -1
            )
            vram_cell = (f'<td data-val="{vram_val:.0f}" class="mit-num">{vram_val:.0f}</td>'
                         if vram_val > 0 else '<td data-val="99999" class="mit-na">-</td>')

            # 언어별 CER/WER 셀
            cer_cells = []
            for lang in ["ko", "en", "ja", "zh"]:
                ck = f"{lang}|{mk}"
                cer = cer_scores.get(ck)
                anom = MODEL_ANOMALIES.get(ck, {})
                if cer is not None:
                    col  = cer_color(cer)
                    tip  = f' title="{anom["note"].replace(chr(34), "&quot;")}"' if anom else ""
                    cer_cells.append(
                        f'<td data-val="{cer}"{tip} class="mit-num" style="color:{col}">'
                        f'{cer:.3f}</td>'
                    )
                elif lang in langs_s:
                    cer_cells.append('<td data-val="9999" class="mit-na">-</td>')
                else:
                    cer_cells.append('<td data-val="9999" class="mit-dot">·</td>')

            has_anomaly = any(f"{lang}|{mk}" in MODEL_ANOMALIES for lang in ["ko","en","ja","zh"])
            warn_html   = ' <span class="warn-icon" title="특이사항 있음 (하단 참조)">⚠️</span>' if has_anomaly else ""

            # 최적화 미적용 항목 분리
            opt_items = [c for c in cons if "미적용" in c]
            cons_only = [c for c in cons if "미적용" not in c]
            opt_note  = info.get("opt_note", "")

            # 장단점 tooltip용
            pros_tip = " / ".join(pros[:3]) if pros else ""
            cons_tip = " / ".join(cons_only[:3]) if cons_only else ""

            # 최적화 셀 (미적용 항목 + opt_note 결과)
            opt_parts = [f'<span style="color:#fbbf24">⚡ {o}</span>' for o in opt_items]
            if opt_note:
                opt_parts.append(f'<span style="color:#34d399">✅ {opt_note}</span>')
            opt_html = "<br>".join(opt_parts) if opt_parts else '<span style="color:#4b5563">-</span>'

            # 릴리즈 이력
            rel = MODEL_RELEASE_INFO.get(mk, {})
            first_rel  = rel.get("first_release", "-")
            latest_ver = rel.get("latest_ver", "-")
            upd_freq   = rel.get("update_freq", "-")

            rows.append(f"""
          <tr data-arch="{arch}">
            <td class="mit-model">
              <span style="font-weight:700;color:#fff">{dname}</span>{warn_html}
              <div class="mit-detail">{detail}</div>
            </td>
            <td><span class="arch-badge" style="background:{arch_col}">{arch}</span></td>
            <td class="mit-dev">{dev}</td>
            <td>{lang_badges}</td>
            {"".join(rtf_cells)}
            {vram_cell}
            {"".join(cer_cells)}
            <td class="mit-pros" title="{pros_tip}">{"✅ " + pros[0] if pros else "-"}</td>
            <td class="mit-cons" title="{cons_tip}">{"⚠️ " + cons_only[0] if cons_only else "-"}</td>
            <td class="mit-opt">{opt_html}</td>
            <td class="mit-rel">{first_rel}</td>
            <td class="mit-ver" title="{latest_ver}">{latest_ver}</td>
            <td class="mit-freq">{upd_freq}</td>
          </tr>""")

        header = """
        <div style="overflow-x:auto">
        <table class="model-info-table" id="model-info-table">
          <thead>
            <tr>
              <th onclick="sortTable(this)" class="sortable">모델</th>
              <th>타입</th>
              <th>개발사</th>
              <th>지원 언어</th>
              <th onclick="sortTable(this)" class="sortable">KO RTF</th>
              <th onclick="sortTable(this)" class="sortable">EN RTF</th>
              <th onclick="sortTable(this)" class="sortable">JA RTF</th>
              <th onclick="sortTable(this)" class="sortable">ZH RTF</th>
              <th onclick="sortTable(this)" class="sortable" title="추론 중 최대 GPU 메모리 사용량(MB). 측정된 언어 기준 첫 번째 값">VRAM(MB)</th>
              <th onclick="sortTable(this)" class="sortable">KO CER ▲</th>
              <th onclick="sortTable(this)" class="sortable">EN WER ▲</th>
              <th onclick="sortTable(this)" class="sortable">JA CER ▲</th>
              <th onclick="sortTable(this)" class="sortable">ZH CER ▲</th>
              <th>주요 장점</th>
              <th>주요 단점</th>
              <th>최적화</th>
              <th onclick="sortTable(this)" class="sortable">최초 공개 ▲</th>
              <th>현재 최신 버전</th>
              <th>업데이트 주기</th>
            </tr>
          </thead>
          <tbody>"""
        return header + "".join(rows) + "\n          </tbody>\n        </table>\n        </div>"

    # ── 전체 HTML ─────────────────────────────────────────────────────────────
    total_models = len(set(v["model_key"] for v in averages.values() if v.get("method","A")=="A"))
    total_entries = len([e for e in entries if e.get("success")])

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>다국어 TTS 벤치마크 보고서</title>
  <style>
    :root {{
      --bg: #f1f5f9; --bg2: #ffffff; --bg3: #f8fafc;
      --text: #0f172a; --text2: #64748b; --text3: #475569;
      --border: #e2e8f0; --accent: #4f46e5;
      --radius: 10px;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; line-height: 1.6; }}

    /* Header */
    .header {{ background: linear-gradient(135deg, #4338ca 0%, #4f46e5 50%, #6366f1 100%); padding: 36px 32px 28px; border-bottom: 2px solid #3730a3; position: relative; overflow: hidden; }}
    .header::after {{ content: ""; position: absolute; inset: 0; background: radial-gradient(ellipse at 70% 50%, rgba(255,255,255,.08) 0%, transparent 60%); pointer-events: none; }}
    .header h1 {{ font-size: 28px; font-weight: 800; color: #fff; margin-bottom: 6px; letter-spacing: -.5px; }}
    .header-meta {{ color: rgba(255,255,255,.75); font-size: 13px; margin-bottom: 18px; }}
    .stat-chips {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .stat-chip {{ background: rgba(255,255,255,.15); border: 1px solid rgba(255,255,255,.3); border-radius: 20px; padding: 5px 16px; font-size: 13px; color: #fff; }}
    .stat-chip span {{ color: #fde68a; font-weight: 700; }}

    /* Tabs */
    .tab-nav {{ background: var(--bg2); padding: 0 24px; border-bottom: 1px solid var(--border); display: flex; gap: 4px; }}
    .tab-btn {{ background: none; border: none; color: var(--text2); padding: 14px 20px; cursor: pointer; font-size: 14px; font-weight: 500; border-bottom: 3px solid transparent; transition: all .2s; }}
    .tab-btn:hover {{ color: var(--text); }}
    .tab-btn.active {{ color: var(--accent); border-bottom-color: var(--accent); }}

    /* Main content */
    .main {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    .lang-section {{ }}
    .section-title {{ font-size: 20px; font-weight: 700; margin-bottom: 4px; }}
    .section-meta {{ color: var(--text2); font-size: 13px; margin-bottom: 20px; }}
    .sub-title {{ font-size: 15px; font-weight: 600; margin: 24px 0 12px; color: var(--text3); }}

    /* Rankings Table */
    .rankings-table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: var(--bg2); border-radius: var(--radius); }}
    .rankings-table th {{ background: var(--bg3); padding: 10px 14px; text-align: left; font-size: 12px; color: var(--text2); font-weight: 600; text-transform: uppercase; letter-spacing: .5px; white-space: nowrap; }}
    .rankings-table td {{ padding: 10px 14px; border-top: 1px solid var(--border); vertical-align: middle; white-space: nowrap; }}
    .rankings-table tbody tr:nth-child(4n+1) td,
    .rankings-table tbody tr:nth-child(4n+2) td {{ background: rgba(0,0,0,.018); }}
    .rankings-table tr:hover td {{ background: rgba(79,70,229,.07) !important; }}
    .sortable {{ cursor: pointer; user-select: none; }}
    .sortable:hover {{ color: var(--text); }}
    .sortable.sort-asc::after  {{ content: " ▲"; font-size: 10px; color: var(--accent); }}
    .sortable.sort-desc::after {{ content: " ▼"; font-size: 10px; color: var(--accent); }}
    .rank {{ color: var(--text2); font-weight: 700; width: 32px; }}
    .model-name {{ font-weight: 600; color: var(--text); }}
    .size {{ color: var(--text2); }}
    .rtf-cell {{ }}
    .rtf-val {{ font-weight: 700; font-size: 14px; display: block; }}
    .rtf-bar {{ height: 4px; border-radius: 2px; margin-top: 3px; }}

    /* Arch badge */
    .arch-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; color: #fff; }}
    .arch-badge.sm {{ font-size: 10px; padding: 1px 6px; }}

    /* Winner Cards (한눈에 보기) */
    .winner-card {{ border-radius: 14px; padding: 20px 22px; display: flex; flex-direction: column; gap: 9px; transition: transform .18s, box-shadow .18s; border: 1px solid; position: relative; overflow: hidden; }}
    .winner-card::before {{ content: ""; position: absolute; inset: 0; opacity: .06; background: radial-gradient(circle at 80% 10%, #fff 0%, transparent 55%); pointer-events: none; }}
    .winner-card:hover {{ transform: translateY(-3px); box-shadow: 0 12px 32px rgba(0,0,0,.55); }}
    .winner-card.card-speed {{ background: linear-gradient(135deg, #052e16 0%, #14532d 100%); border-color: #166534; }}
    .winner-card.card-accuracy {{ background: linear-gradient(135deg, #0c1445 0%, #1e3a8a 100%); border-color: #1d4ed8; }}
    .winner-card.card-balance {{ background: linear-gradient(135deg, #3b1a00 0%, #7c2d12 100%); border-color: #c2410c; }}
    .winner-emoji-title {{ display: flex; align-items: center; gap: 8px; }}
    .winner-emoji {{ font-size: 24px; line-height: 1; }}
    .winner-title {{ font-size: 11px; font-weight: 700; color: rgba(255,255,255,.55); text-transform: uppercase; letter-spacing: 1px; }}
    .winner-model-row {{ display: flex; align-items: center; gap: 8px; margin-top: 2px; }}
    .winner-name {{ font-size: 18px; color: #fff; font-weight: 800; letter-spacing: -.3px; }}
    .winner-dev {{ font-size: 11.5px; color: rgba(255,255,255,.38); }}
    .winner-highlight {{ font-size: 13px; color: rgba(255,255,255,.55); margin-top: 2px; }}
    .winner-metric {{ display: flex; align-items: baseline; gap: 8px; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,.12); }}
    .winner-metric-label {{ font-size: 11px; color: rgba(255,255,255,.4); text-transform: uppercase; letter-spacing: .5px; white-space: nowrap; }}
    .winner-metric-val {{ font-size: 34px; font-weight: 900; color: #fde68a; letter-spacing: -1.5px; line-height: 1; }}

    /* TL;DR Summary Bar */
    .tldr-bar {{ display: flex; align-items: center; gap: 0; flex-wrap: wrap; background: #fff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; margin-bottom: 18px; row-gap: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
    .tldr-head {{ font-size: 10px; font-weight: 800; color: var(--text2); text-transform: uppercase; letter-spacing: 1.5px; white-space: nowrap; margin-right: 18px; padding-right: 18px; border-right: 1px solid var(--border); }}
    .tldr-item {{ display: flex; align-items: center; gap: 6px; padding: 0 16px; border-right: 1px solid var(--border); }}
    .tldr-item:last-child {{ border-right: none; }}
    .tldr-label {{ font-size: 11px; color: var(--text2); white-space: nowrap; }}
    .tldr-model {{ font-size: 13px; font-weight: 700; color: var(--text); white-space: nowrap; }}
    .tldr-val {{ font-size: 12px; font-weight: 700; white-space: nowrap; }}

    /* Matrix collapsible */
    .matrix-details {{ border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
    .matrix-details > summary {{ padding: 13px 18px; font-size: 13px; font-weight: 600; color: var(--text2); cursor: pointer; list-style: none; display: flex; align-items: center; gap: 8px; user-select: none; background: var(--bg3); }}
    .matrix-details > summary::-webkit-details-marker {{ display: none; }}
    .matrix-details > summary::before {{ content: "▶ "; font-size: 10px; transition: transform .2s; }}
    .matrix-details[open] > summary::before {{ content: "▼ "; }}
    .matrix-details > summary:hover {{ color: var(--text); }}
    .matrix-details > div {{ padding: 0 14px 14px; }}

    /* Overview Matrix Table */
    .ov-table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: var(--bg2); border-radius: 10px; font-size: 13px; }}
    .ov-table th {{ background: var(--bg3); padding: 9px 14px; text-align: center; font-size: 11px; color: var(--text2); font-weight: 600; text-transform: uppercase; letter-spacing: .5px; white-space: nowrap; }}
    .ov-table th:first-child {{ text-align: left; border-radius: 10px 0 0 0; }}
    .ov-table th:last-child {{ border-radius: 0 10px 0 0; }}
    .ov-table td {{ padding: 9px 14px; border-top: 1px solid var(--border); vertical-align: middle; }}
    .ov-table tr:hover td {{ background: rgba(99,102,241,.06); }}
    .ov-num {{ text-align: center; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .ov-name {{ white-space: nowrap; }}

    /* Collapsible details */
    details summary {{ cursor: pointer; user-select: none; list-style: none; }}
    details summary::-webkit-details-marker {{ display: none; }}
    details summary::before {{ content: "▶ "; font-size: 11px; color: var(--text2); transition: transform .2s; }}
    details[open] summary::before {{ content: "▼ "; }}

    /* Criteria Section */
    .criteria-section {{ background: var(--bg2); border-radius: 10px; padding: 18px 22px; margin-bottom: 20px; border: 1px solid var(--border); }}
    .criteria-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 12px; }}
    .criteria-item {{ background: var(--bg3); border-radius: 8px; padding: 12px 14px; border-left: 3px solid var(--accent); }}
    .criteria-item h4 {{ font-size: 12px; font-weight: 700; color: var(--accent); margin: 0 0 4px; text-transform: uppercase; letter-spacing: .5px; }}
    .criteria-item p {{ font-size: 12px; color: var(--text2); margin: 0; line-height: 1.6; }}

    /* Model Table Legend */
    .model-table-legend {{ background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; font-size: 12.5px; color: var(--text2); line-height: 1.8; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 3px 0; border-bottom: 1px solid var(--border); }}
    .legend-row:last-child {{ border-bottom: none; }}
    .legend-label {{ font-weight: 700; color: var(--accent); min-width: 130px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}

    /* Legend Bar */
    .legend-bar {{ display: flex; align-items: center; gap: 12px; padding: 10px 24px; background: var(--bg2); border-bottom: 1px solid var(--border); flex-wrap: wrap; font-size: 12px; }}
    .legend-title {{ color: var(--text2); font-weight: 700; text-transform: uppercase; letter-spacing: .5px; white-space: nowrap; }}
    .legend-group {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
    .legend-chip {{ border: 1px solid; border-radius: 12px; padding: 2px 9px; font-size: 11px; font-weight: 600; white-space: nowrap; }}
    .legend-sep {{ color: var(--border); font-size: 16px; }}

    /* Arch Filter Bar */
    .arch-filter-bar {{ display: flex; align-items: center; gap: 6px; margin-left: auto; padding: 0 4px; }}
    .af-label {{ color: var(--text2); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; margin-right: 2px; white-space: nowrap; }}
    .af-btn {{ background: #fff; border: 1px solid var(--border); color: var(--text2); padding: 4px 13px; cursor: pointer; font-size: 12px; font-weight: 600; border-radius: 6px; transition: all .15s; white-space: nowrap; }}
    .af-btn:hover {{ background: rgba(79,70,229,.08); color: var(--text); border-color: rgba(79,70,229,.4); }}
    .af-btn.active {{ background: rgba(79,70,229,.12); border-color: var(--accent); color: var(--accent); }}

    /* Reference Audio Box */
    .ref-audio-box {{ background: #eef2ff; border-left: 4px solid var(--accent); padding: 10px 14px; margin-bottom: 14px; display: flex; align-items: center; gap: 14px; border-radius: 6px; }}
    .ref-label {{ font-weight: 600; color: var(--accent); white-space: nowrap; font-size: 13px; }}
    .ref-audio-box audio {{ height: 32px; flex: 1; }}

    /* Audio Section */
    .audio-section {{ background: var(--bg2); border-radius: var(--radius); padding: 16px; }}
    .audio-controls {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
    .audio-controls label {{ color: var(--text2); font-size: 13px; }}
    .text-selector {{ background: var(--bg3); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 7px 12px; font-size: 13px; cursor: pointer; min-width: 160px; }}
    .btn-play-all, .btn-stop-all {{ background: var(--accent); color: #fff; border: none; border-radius: 6px; padding: 7px 14px; cursor: pointer; font-size: 13px; font-weight: 500; }}
    .btn-stop-all {{ background: var(--bg3); }}
    .text-display {{ background: var(--bg3); border-radius: 6px; padding: 10px 14px; margin-bottom: 14px; font-size: 13px; color: var(--text3); line-height: 1.8; min-height: 20px; }}
    .audio-table-wrap {{ overflow-x: auto; }}
    .audio-cmp-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .audio-cmp-table th {{ background: var(--bg3); padding: 7px 10px; text-align: left; font-size: 11px; color: var(--text2); font-weight: 600; text-transform: uppercase; letter-spacing: .4px; white-space: nowrap; }}
    .audio-cmp-table td {{ padding: 5px 10px; border-top: 1px solid var(--border); vertical-align: middle; white-space: nowrap; }}
    .ac-rank {{ width: 32px; text-align: center; color: var(--text2); font-size: 12px; }}
    .ac-rtf {{ width: 60px; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .ac-cer {{ width: 80px; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .ac-audio {{ width: 55%; min-width: 240px; }}
    .ac-model {{ white-space: nowrap; font-size: 13px; }}
    .spk-selector {{ display: flex; align-items: center; gap: 6px; padding: 8px 0 10px; flex-wrap: wrap; }}
    .spk-sel-label {{ color: var(--text2); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .4px; margin-right: 4px; }}
    .spk-sel {{ background: var(--bg3); color: var(--text2); border: 1px solid var(--border); border-radius: 16px; padding: 4px 14px; cursor: pointer; font-size: 12px; transition: all .15s; }}
    .spk-sel:hover {{ color: var(--text); border-color: var(--accent); }}
    .spk-sel.active {{ background: rgba(79,70,229,.12); color: var(--accent); border-color: var(--accent); font-weight: 600; }}
    .ac-filter-btns {{ display: flex; gap: 4px; }}
    .ac-filter {{ background: var(--bg3); color: var(--text2); border: 1px solid var(--border); border-radius: 5px; padding: 5px 11px; cursor: pointer; font-size: 12px; transition: background .15s; }}
    .ac-filter.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .player-row.hidden {{ display: none; }}
    audio {{ accent-color: var(--accent); }}
    .no-data {{ color: var(--text2); font-style: italic; padding: 20px 0; }}

    /* Model Info Section */
    .cards-section {{ margin-top: 48px; padding-top: 24px; border-top: 1px solid var(--border); }}
    .cards-section h2 {{ font-size: 20px; font-weight: 700; margin-bottom: 4px; }}
    .cards-section .desc {{ color: var(--text2); font-size: 13px; margin-bottom: 16px; line-height: 1.8; }}
    .cards-inner {{ max-width: 1400px; margin: 0 auto; padding: 0 24px; }}

    /* Model Info Table */
    .model-info-table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: var(--bg2); border-radius: var(--radius); font-size: 13px; }}
    .model-info-table th {{ background: var(--bg3); padding: 9px 12px; text-align: left; font-size: 11px; color: var(--text2); font-weight: 600; text-transform: uppercase; letter-spacing: .4px; white-space: nowrap; border-bottom: 2px solid var(--border); }}
    .model-info-table td {{ padding: 9px 12px; border-top: 1px solid var(--border); vertical-align: middle; }}
    .model-info-table tr:hover td {{ background: rgba(99,102,241,.05); }}
    .mit-model {{ min-width: 160px; white-space: normal !important; position: sticky; left: 0; z-index: 2; background: var(--bg2); box-shadow: 2px 0 6px rgba(0,0,0,.35); }}
    .model-info-table thead th:first-child {{ position: sticky; left: 0; z-index: 3; background: var(--bg3); box-shadow: 2px 0 6px rgba(0,0,0,.35); }}
    .model-info-table tr:hover td.mit-model {{ background: #f1f5f9; }}
    .mit-detail {{ font-size: 11px; color: var(--text2); margin-top: 4px; white-space: normal !important; line-height: 1.3; max-width: 240px; }}
    .mit-dev {{ font-size: 12px; color: var(--text2); white-space: nowrap; }}
    .mit-num {{ white-space: nowrap; font-variant-numeric: tabular-nums; }}
    .mit-sub {{ font-size: 10px; color: var(--text2); margin-top: 2px; }}
    .mit-na {{ color: #475569; text-align: center; }}
    .mit-dot {{ color: var(--text2); text-align: center; }}
    .mit-pros {{ font-size: 12px; color: #15803d; max-width: 300px; white-space: normal !important; line-height: 1.4; word-break: keep-all; }}
    .mit-cons {{ font-size: 12px; color: #dc2626; max-width: 300px; white-space: normal !important; line-height: 1.4; word-break: keep-all; }}
    .mit-opt  {{ font-size: 12px; max-width: 200px; white-space: normal !important; line-height: 1.6; }}
    .mit-rel  {{ font-size: 12px; color: var(--text2); white-space: nowrap; font-variant-numeric: tabular-nums; }}
    .mit-ver  {{ font-size: 11px; color: var(--text3); max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: default; }}
    .mit-freq {{ font-size: 12px; white-space: nowrap; }}

    /* Language badges */
    .lang-badge {{ padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
    .lang-ko {{ background: #ede9fe; color: #6d28d9; border: 1px solid #c4b5fd; }}
    .lang-en {{ background: #e0f2fe; color: #0369a1; border: 1px solid #7dd3fc; }}
    .lang-ja {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }}
    .lang-zh {{ background: #fff7ed; color: #c2410c; border: 1px solid #fdba74; }}

    /* Warning icon */
    .warn-icon {{ cursor: help; font-size: 12px; vertical-align: middle; }}

    /* Anomalies Section */
    .anomalies-section {{ margin-top: 48px; padding-top: 24px; border-top: 1px solid var(--border); }}
    .anomalies-section h2 {{ font-size: 20px; font-weight: 700; margin-bottom: 8px; }}
    .anomalies-section .desc {{ color: var(--text2); font-size: 13px; margin-bottom: 20px; }}
    .anomaly-table {{ width: 100%; border-collapse: collapse; background: var(--bg2); border-radius: var(--radius); overflow: hidden; }}
    .anomaly-table th {{ background: var(--bg3); padding: 10px 14px; text-align: left; font-size: 12px; color: var(--text2); font-weight: 600; text-transform: uppercase; letter-spacing: .5px; white-space: nowrap; }}
    .anomaly-table td {{ padding: 10px 14px; border-top: 1px solid var(--border); font-size: 13px; vertical-align: top; }}
    .anomaly-table tr:hover td {{ background: rgba(251,191,36,.04); }}
    .severity-high {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
    .severity-med  {{ background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }}

    /* Overview Section */
    .ov-card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }}
    .ov-card-header {{ display: flex; align-items: center; padding-bottom: 12px; margin-bottom: 12px; }}
    .ov-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .ov-table th {{ color: var(--text2); font-weight: 600; text-align: right; padding: 6px 4px; border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; }}
    .ov-table th:first-child {{ text-align: left; }}
    .ov-table td {{ padding: 6px 4px; border-bottom: 1px solid var(--border); text-align: right; font-variant-numeric: tabular-nums; }}
    .ov-table td:first-child {{ text-align: left; font-weight: 600; color: var(--text); }}
    .ov-table tr:last-child td {{ border-bottom: none; }}
    .ov-name {{ color: var(--text); font-weight: 600; max-width: 150px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; cursor: default; }}
    .ov-num {{ color: var(--text2); }}

    /* Footer */
    .footer {{ text-align: center; color: var(--text2); font-size: 12px; padding: 32px 0; margin-top: 48px; border-top: 1px solid var(--border); }}
    .spk-detail-row {{ display: none; }}
    .spk-detail-row.visible {{ display: table-row; }}
    .spk-detail-cell {{ background: #f8fafc; padding: 0 !important; border-top: none !important; }}
    .spk-toggle {{ background: rgba(79,70,229,.08); border: 1px solid rgba(79,70,229,.25); color: var(--accent); cursor: pointer; font-size: 11px; padding: 4px 10px; border-radius: 5px; white-space: nowrap; transition: background .15s; }}
    .spk-toggle:hover {{ background: rgba(79,70,229,.15); }}
    .spk-toggle.open {{ background: rgba(79,70,229,.15); color: var(--accent); }}

    /* 인라인 오디오 셀 */
    .audio-cell {{ min-width: 72px; width: 72px; }}
    .mini-player {{ display: flex; gap: 5px; align-items: center; justify-content: center; }}
    .play-btn {{
        border: none; border-radius: 50%; width: 28px; height: 28px;
        cursor: pointer; font-size: 11px; display: flex; align-items: center; justify-content: center;
        transition: background .15s, transform .1s;
        background: var(--accent); color: #fff;
    }}
    .play-btn:hover {{ background: #4338ca; transform: scale(1.1); }}
    .play-btn.playing {{ background: #0ea5e9; }}

    /* detail 행 전체 레이아웃 */
    .spk-detail-cell {{ white-space: normal !important; padding: 0 !important; border-top: none !important; }}
    .detail-full {{ display: flex; flex-wrap: wrap; gap: 0; background: #f8fafc; }}
    .detail-section {{ padding: 14px 18px; border-right: 1px solid var(--border); flex-shrink: 0; }}
    .detail-section:last-child {{ border-right: none; flex: 1; min-width: 260px; }}

    /* 전 언어 성능 미니 테이블 */
    .detail-perf-mini {{ border-collapse: collapse; font-size: 12px; white-space: nowrap; }}
    .detail-perf-mini th {{ font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: .4px;
                            padding: 3px 10px 6px; border-bottom: 1px solid var(--border); font-weight: 600; }}
    .detail-perf-mini td {{ padding: 5px 10px; border-bottom: 1px solid var(--border);
                            font-variant-numeric: tabular-nums; }}
    .detail-perf-mini tr:last-child td {{ border-bottom: none; }}

    /* 모델 메타데이터 */
    .detail-meta-grid {{ display: flex; flex-direction: column; gap: 7px; min-width: 200px; max-width: 340px; }}
    .dm-label {{ font-size: 10px; font-weight: 600; color: #475569; text-transform: uppercase;
                 letter-spacing: .5px; display: inline-block; min-width: 60px; margin-right: 6px; }}
    .dm-val {{ font-size: 12px; color: var(--text3); white-space: normal; word-break: break-word; }}

    /* 장단점 */
    .detail-pros-cons {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
                         box-sizing: border-box; overflow: hidden; }}
    .detail-pros strong, .detail-cons strong {{ font-size: 10px; font-weight: 700; text-transform: uppercase;
                                                letter-spacing: .4px; display: block; margin-bottom: 6px; }}
    .detail-pros strong {{ color: #15803d; }}
    .detail-cons strong {{ color: #dc2626; }}
    .detail-pros ul, .detail-cons ul {{ margin: 0; padding-left: 14px; }}
    .detail-pros li, .detail-cons li {{ font-size: 12px; color: var(--text3); line-height: 1.7;
                                        white-space: normal; word-break: break-word; overflow-wrap: anywhere; }}
    /* 공식 성능 컬럼 */
    .official-cell {{ font-size: 11px; min-width: 120px; max-width: 200px; white-space: normal; word-break: break-word; overflow-wrap: anywhere; }}
    .official-val {{ color: var(--accent); line-height: 1.5; }}
    .official-none {{ color: var(--text2); font-style: italic; }}
    /* 공식 성능 (detail) */
    .detail-official {{ grid-column: 1 / -1; margin-top: 6px; padding-top: 10px;
                        border-top: 1px solid var(--border); }}
    .detail-official strong {{ font-size: 10px; font-weight: 700; text-transform: uppercase;
                                letter-spacing: .4px; color: var(--accent); display: block; margin-bottom: 6px; }}
    .detail-official ul {{ margin: 0; padding-left: 14px; }}
    .detail-official li {{ font-size: 12px; color: var(--text3); line-height: 1.7;
                           white-space: normal; word-break: break-word; overflow-wrap: anywhere; }}
    @media (max-width: 700px) {{
      .detail-pros-cons {{ grid-template-columns: 1fr; }}
    }}
    .methodology-section {{ max-width: 1300px; margin: 0 auto; padding: 0 24px 4px; }}
    .methodology-section details {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
    .methodology-section summary {{ padding: 14px 20px; font-size: 14px; font-weight: 600; color: var(--text2); cursor: pointer; list-style: none; display: flex; align-items: center; gap: 8px; user-select: none; }}
    .methodology-section summary::-webkit-details-marker {{ display: none; }}
    .methodology-section summary::before {{ content: "▶"; font-size: 10px; transition: transform 0.2s; }}
    .methodology-section details[open] summary::before {{ transform: rotate(90deg); }}
    .methodology-section summary:hover {{ color: var(--text); }}
    .meth-body {{ padding: 4px 20px 20px; }}
    .meth-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px; margin-top: 12px; }}
    .meth-card {{ background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }}
    .meth-title {{ font-size: 13px; font-weight: 700; color: var(--accent); margin-bottom: 10px; }}
    .meth-card ul {{ margin: 0; padding-left: 18px; }}
    .meth-card li {{ font-size: 12px; color: var(--text2); line-height: 1.7; }}
    .meth-card li strong {{ color: var(--text); }}
    .meth-card code {{ background: #e2e8f0; padding: 1px 5px; border-radius: 3px; font-size: 11px; color: var(--accent); }}
  </style>
</head>
<body>

<div class="header">
  <h1>🎙️ 다국어 TTS 벤치마크 보고서</h1>
  <div class="header-meta">생성: {generated_at} &nbsp;|&nbsp; GPU: RTX 5080 &nbsp;|&nbsp; LLM-based TTS 집중 분석</div>
  <div class="stat-chips">
    <div class="stat-chip">모델 <span>{total_models}</span>개</div>
    <div class="stat-chip">언어 <span>{len(available_langs)}</span>개</div>
    <div class="stat-chip">완료 조합 <span>{total_entries}</span>개</div>
    <div class="stat-chip">정확도 평가 ASR: <span>Whisper large-v3</span></div>
  </div>
</div>

<div class="arch-filter-bar" style="position:sticky; top:0; z-index:11; background:var(--bg2); border-bottom:1px solid var(--border); padding:8px 32px;">
  <span class="af-label">구조 필터</span>
  <button class="af-btn" onclick="filterArch('all')">전체</button>
  <button class="af-btn active" onclick="filterArch('LLM')">LLM / FM</button>
</div>

<div class="legend-bar">
  <span class="legend-title">색상 기준</span>
  <span class="legend-group">
    <span class="legend-label">RTF</span>
    <span class="legend-chip" style="background:#22c55e22;color:#22c55e;border-color:#22c55e55">&lt; 0.2 매우 빠름</span>
    <span class="legend-chip" style="background:#84cc1622;color:#84cc16;border-color:#84cc1655">0.2–0.5 빠름</span>
    <span class="legend-chip" style="background:#f59e0b22;color:#f59e0b;border-color:#f59e0b55">0.5–1.0 실시간 경계</span>
    <span class="legend-chip" style="background:#ef444422;color:#ef4444;border-color:#ef444455">1.0+ 느림</span>
  </span>
  <span class="legend-sep">|</span>
  <span class="legend-group">
    <span class="legend-label">CER/WER</span>
    <span class="legend-chip" style="background:#22c55e22;color:#22c55e;border-color:#22c55e55">≤ 2% 완벽</span>
    <span class="legend-chip" style="background:#84cc1622;color:#84cc16;border-color:#84cc1655">≤ 10% 양호</span>
    <span class="legend-chip" style="background:#f59e0b22;color:#f59e0b;border-color:#f59e0b55">≤ 30% 주의</span>
  </span>
</div>

<div class="main">
    <div class="criteria-section">
      <h3 style="font-size:16px;font-weight:700;margin:0 0 4px;color:var(--text);">📋 모델 선정 기준</h3>
      <p style="font-size:12.5px;color:var(--text2);margin:0 0 12px;">벤치마크에 포함된 모델은 아래 4가지 대전제를 모두 충족한 것들입니다. 이를 충족하지 못한 모델은 하단 '제외된 모델' 항목을 참조하세요.</p>
      <div class="criteria-grid">
        <div class="criteria-item">
          <h4>1. 한국어 지원</h4>
          <p>별도 G2P/로마자 변환 없이 한국어 텍스트를 직접 합성할 수 있어야 함 (End-to-End)</p>
        </div>
        <div class="criteria-item">
          <h4>2. 상업적 라이선스</h4>
          <p>Apache 2.0, MIT, CC-BY 등 상업 활용 가능한 오픈소스 라이선스 보유</p>
        </div>
        <div class="criteria-item">
          <h4>3. 실시간 속도 (RTF &lt; 1.0)</h4>
          <p>실제 서비스 투입을 위해 RTF 1.0 미만 (또는 GPU 기준 실시간 처리 가능) 요건</p>
        </div>
        <div class="criteria-item">
          <h4>4. 가중치 공개</h4>
          <p>HuggingFace 또는 공식 저장소에서 모델 가중치를 직접 다운로드할 수 있어야 함</p>
        </div>
      </div>
    </div>

    {build_overview_section(averages, cer_scores, sim_scores)}

  <div class="tab-nav" style="position:sticky; top:36px; z-index:9; margin:0 -24px; padding:0 24px; border-radius:0;">
    {tab_buttons()}
  </div>

  {"".join(lang_sections)}

</div>

<div class="footer">
  다국어 TTS 벤치마크 보고서 &nbsp;|&nbsp; {generated_at} 생성 &nbsp;|&nbsp;
  RTF = 추론시간/오디오길이 (낮을수록 빠름, &lt;1.0 = 실시간 가능)
</div>

<script>
  const TEXT_CONTENT = {text_content_js};

  let _curAudio = null, _curPlayBtn = null;
  function togglePlay(btn) {{
    const audio = btn.closest('.mini-player').querySelector('audio');
    if (_curAudio && _curAudio !== audio) {{
      _curAudio.pause(); _curAudio.currentTime = 0;
      _curPlayBtn.textContent = '▶'; _curPlayBtn.classList.remove('playing');
    }}
    if (audio.paused) {{
      audio.play();
      btn.textContent = '⏸'; btn.classList.add('playing');
      _curAudio = audio; _curPlayBtn = btn;
      audio.onended = () => {{ btn.textContent = '▶'; btn.classList.remove('playing'); _curAudio = null; _curPlayBtn = null; }};
    }} else {{
      audio.pause(); audio.currentTime = 0;
      btn.textContent = '▶'; btn.classList.remove('playing');
      _curAudio = null; _curPlayBtn = null;
    }}
  }}
  function toggleSpk(rowId) {{
    const row = document.getElementById(rowId);
    if (!row) return;
    const isOpen = row.classList.toggle('visible');
    // 버튼 화살표 회전 업데이트
    const btn = row.previousElementSibling && row.previousElementSibling.querySelector('.spk-toggle');
    if (btn) {{
      btn.classList.toggle('open', isOpen);
      btn.textContent = isOpen ? '상세 △' : '상세 ▿';
    }}
  }}

  function switchLang(lang) {{
    document.querySelectorAll('.lang-section').forEach(s => s.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    const sec = document.getElementById('section-' + lang);
    if (sec) sec.style.display = 'block';
    const btn = document.getElementById('btn-' + lang);
    if (btn) btn.classList.add('active');
  }}

  function playAll(lang) {{
    const grid = document.getElementById('audio-grid-' + lang);
    if (!grid) return;
    grid.querySelectorAll('audio.audio-el').forEach(a => {{
      a.play().catch(() => {{}});
    }});
  }}

  function stopAll(lang) {{
    const grid = document.getElementById('audio-grid-' + lang);
    if (!grid) return;
    grid.querySelectorAll('audio.audio-el').forEach(a => {{
      a.pause(); a.currentTime = 0;
    }});
  }}

  function updateRanks(table) {{
    const medals = ['🥇', '🥈', '🥉'];
    let visibleIdx = 0;
    
    // 현재 활성화된 필터 파악
    const activeBtn = document.querySelector('.af-btn.active');
    const archFilter = (activeBtn && activeBtn.textContent.includes('LLM')) ? 'LLM' : 'all';

    function isRowVisible(a) {{
      return archFilter === 'all' || (archFilter === 'LLM' && (a === 'LLM' || a === 'FM'));
    }}

    const mainRows = Array.from(table.querySelectorAll('tr[data-arch]:not(.spk-detail-row)'));
    mainRows.forEach(row => {{
      const arch = row.getAttribute('data-arch');
      if (isRowVisible(arch)) {{
        const rankCell = row.querySelector('.rank');
        if (rankCell) {{
          rankCell.textContent = visibleIdx < 3 ? medals[visibleIdx] : String(visibleIdx + 1);
        }}
        visibleIdx++;
      }}
    }});
  }}

  function filterArch(arch) {{
    // 버튼 활성화
    document.querySelectorAll('.af-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.af-btn').forEach(b => {{
      if (arch === 'all' && b.textContent.includes('전체')) b.classList.add('active');
      if (arch === 'LLM' && b.textContent.includes('LLM')) b.classList.add('active');
    }});

    function isVisible(a) {{
      return arch === 'all' || (arch === 'LLM' && (a === 'LLM' || a === 'FM'));
    }}

    // 테이블 행 (rankings + matrix)
    document.querySelectorAll('tr[data-arch]').forEach(row => {{
      if (row.classList.contains('spk-detail-row')) return;
      const show = isVisible(row.getAttribute('data-arch'));
      row.style.display = show ? '' : 'none';
      const next = row.nextElementSibling;
      if (next && next.classList.contains('spk-detail-row')) {{
        if (!show) {{ next.classList.remove('visible'); next.style.display = 'none'; }}
        else {{ next.style.display = ''; }}
      }}
    }});

    // 한눈에 보기 winner 블록 전환
    const wa = document.getElementById('overview-winners-all');
    const wl = document.getElementById('overview-winners-llm');
    if (wa) wa.style.display = arch === 'all' ? '' : 'none';
    if (wl) wl.style.display = arch === 'LLM' ? '' : 'none';

    // 순위 재계산 (모든 언어 탭의 테이블 대상)
    document.querySelectorAll('.rankings-table').forEach(table => updateRanks(table));

    // 전체 모델 목록 매트릭스 번호 재계산
    const matrixTable = document.querySelector('.ov-table tbody');
    if (matrixTable) {{
      let n = 1;
      Array.from(matrixTable.querySelectorAll('tr[data-arch]')).forEach(row => {{
        const numCell = row.querySelector('td:first-child');
        if (!numCell) return;
        if (row.style.display === 'none') return;
        numCell.textContent = n++;
      }});
    }}
  }}

  function sortTable(th) {{
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const allRows = Array.from(tbody.children).filter(r => r.tagName === 'TR');

    // 메인 행과 상세 행 분리
    const mainRows = allRows.filter(r => !r.classList.contains('spk-detail-row'));
    const detailMap = new Map();
    mainRows.forEach(r => {{
      const next = r.nextElementSibling;
      if (next && next.classList.contains('spk-detail-row')) detailMap.set(r, next);
    }});

    const idx = Array.from(th.parentNode.children).indexOf(th);
    const asc = (th.dataset.asc !== 'true');

    // 헤더 정렬 표시 업데이트
    Array.from(th.parentNode.children).forEach(h => {{
      h.dataset.asc = '';
      h.classList.remove('sort-asc', 'sort-desc');
    }});
    th.dataset.asc = String(asc);
    th.classList.add(asc ? 'sort-asc' : 'sort-desc');

    mainRows.sort((a, b) => {{
      let av = a.children[idx]?.dataset.val ?? a.children[idx]?.textContent.trim() ?? '';
      let bv = b.children[idx]?.dataset.val ?? b.children[idx]?.textContent.trim() ?? '';
      const af = parseFloat(av);
      const bf = parseFloat(bv);
      if (!isNaN(af) && !isNaN(bf)) {{
        // -1 = 데이터 없음 → 항상 맨 아래
        const na = af < 0, nb = bf < 0;
        if (na && nb) return 0;
        if (na) return 1;
        if (nb) return -1;
        return asc ? af - bf : bf - af;
      }}
      if (av === bv) return 0;
      return asc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
    }});

    // 재배치 + 순위 업데이트
    mainRows.forEach((r, i) => {{
      tbody.appendChild(r);
      const d = detailMap.get(r);
      if (d) tbody.appendChild(d);
    }});

    // 가시적인 행 기준으로 순위 재계산
    updateRanks(table);
  }}

  document.addEventListener('DOMContentLoaded', () => {{
    // 초기 로딩 시 텍스트 표시
    document.querySelectorAll('.text-display').forEach(el => {{
       const lang = el.id.replace('text-display-', '');
       const pivotKey = (lang === 'ko' ? 'ko_medium' : lang + '_medium');
       el.textContent = TEXT_CONTENT[pivotKey] || '';
    }});
    filterArch('LLM');
  }});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"보고서 저장: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="다국어 TTS 벤치마크 HTML 보고서 생성")
    parser.add_argument("--results-dir", default=os.path.join(BENCH_DIR, "results", "results_multilingual"))
    parser.add_argument("--output", default=None, help="출력 HTML 경로 (기본: results-dir/report.html)")
    args = parser.parse_args()

    out = args.output or os.path.join(args.results_dir, "benchmark_multilingual.html")
    generate_html(args.results_dir, out)


if __name__ == "__main__":
    main()
