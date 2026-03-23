# TTS 모델 벤치마크 계획서

## 목적
동일한 한국어 문장을 공개된 LLM 기반 TTS 모델 14종에 입력하여,
**음성 품질 · 추론 속도 · 자원 사용량**을 객관적인 수치로 비교한다.

---

## 테스트 텍스트 (고정)
```
안녕하세요. 인공지능 음성 합성 기술이 놀라울 정도로 발전했습니다.
한국어 발음의 정확성과 자연스러움을 평가하고 있습니다.
```

## 측정 항목
| 항목 | 설명 |
|------|------|
| 로드 시간 | 모델을 메모리에 올리는 데 걸린 시간 (s) |
| TTFA | 첫 번째 오디오 청크가 나올 때까지 시간 (ms) |
| 추론 시간 | 전체 음성 생성 완료까지 시간 (s) |
| 오디오 길이 | 생성된 WAV 길이 (s) |
| RTF | 추론 시간 / 오디오 길이 (낮을수록 빠름) |
| VRAM | 모델 로드 후 GPU 메모리 사용량 (MB) |

---

## 진행 현황

| 순서 | 모델 | 파라미터 | 한국어 | 상태 | 비고 |
|------|------|----------|--------|------|------|
| 1 | CosyVoice 2.0 | 500M | ✅ | ✅ 완료 | TTFA 27,669ms / RTF 0.653 (비정상, 재확인 필요) |
| 2 | CosyVoice 3.0 | 500M | ✅ | ✅ 완료 | TTFA 4,104ms / RTF 0.619 / VRAM 3,285MB |
| 3 | Qwen3-TTS 0.6B | 600M | ✅ | ✅ 완료 | TTFA 29,046ms / RTF 3.331 / VRAM 2,070MB |
| 4 | Qwen3-TTS 1.7B | 1.7B | ✅ | ✅ 완료 | TTFA 32,275ms / RTF 3.635 / VRAM 4,004MB |
| 5 | Spark-TTS 0.5B | 500M | △ | ✅ 완료 | TTFA 10,492ms / RTF 1.064 / VRAM 3,740MB |
| 6 | Bark | ~900M | △ | ✅ 완료 | TTFA 19,095ms / RTF 1.721 / VRAM 4,234MB (로드 640s) |
| 7 | Fish Speech 1.5 | ~500M | △ | ✅ 완료 | TTFA 11,981ms / RTF 1.303 / VRAM 1,562MB |
| 8 | Orpheus-TTS | 3B | ❌ | ❌ 스킵 | gated 모델 - 접근 권한 없음 |
| 9 | MioTTS 1.7B | 1.7B | △ | ✅ 완료 | TTFA 11,680ms / RTF 0.933 / VRAM 4,206MB (영/일, en_medium) |
| 10 | MioTTS 2.6B | 2.6B | △ | ✅ 완료 | TTFA 7,686ms / RTF 0.709 / VRAM 5,820MB (영/일, en_medium) |
| 11 | IndexTTS-2 | ~1B | △ | ✅ 완료 | TTFA 335,826ms / RTF 19.387 / VRAM 6,285MB (FP16, 중/영/일) |
| 12 | GPT-SoVITS | ~200M | ✅ | ✅ 완료 | TTFA 4,486ms / RTF 0.442 / VRAM 1,925MB (V3+BigVGAN, 한국어) |
| 13 | GLM-TTS | ~1.5B | ❌ | ✅ 완료 | TTFA 14,117ms / RTF 0.654 / VRAM 5,500MB (중국어 텍스트, 한국어 불가) |
| 14 | LLMVoX | 30M | ❌ | ❌ 스킵 | LLM 애드온 (Llama 8B 필요) - 독립 실행 불가 |
| - | VALL-E | - | - | ❌ 스킵 | 비공개 |
| - | Voicebox | - | - | ❌ 스킵 | 비공개 |

---

## 파일 구조

```
d:\tts-benchmark\
│
├── BENCHMARK_PLAN.md          ← 이 문서
│
├── run_benchmark.py           ← 메인 실행 CLI
│                                 (python run_benchmark.py --models cosyvoice3 qwen3_1.7b ...)
│
├── adapters/
│   ├── run_model.py           ← 각 모델 실제 실행 로직 (서브프로세스 진입점)
│   └── __init__.py
│
├── benchmark/
│   ├── core.py                ← BenchmarkResult 데이터 구조 + 로거 (JSON/CSV/MD 저장)
│   ├── base_adapter.py        ← 공통 추상 클래스
│   └── __init__.py
│
├── results/
│   ├── 20260226_153201_results.json    ← 원시 데이터 (모든 실행 누적)
│   ├── 20260226_153201_results.csv     ← 스프레드시트용
│   ├── 20260226_153201_report.md       ← 읽기 좋은 마크다운 리포트
│   └── wav/
│       ├── cosyvoice2/CosyVoice2-0.5B.wav
│       ├── cosyvoice3/CosyVoice3-0.5B.wav
│       └── ...                         ← 모델별 생성 음성 파일
│
└── cosyvoice/                 ← CosyVoice 레포 (패치 적용됨)
    └── .venv/                 ← CosyVoice 전용 Python 3.10 venv

d:\models\                     ← 모델 가중치 저장소
├── CosyVoice2-0.5B/
├── CosyVoice3-0.5B/
├── Qwen3-TTS-0.6B/
├── Qwen3-TTS-1.7B/
├── fish-speech-1.5/
└── ...
```

---

## 실행 방법

```bash
cd d:\tts-benchmark

# 특정 모델 실행
python run_benchmark.py --models cosyvoice3 --runs 1

# 여러 모델 한 번에
python run_benchmark.py --models cosyvoice2 cosyvoice3 qwen3_0.6b --runs 3

# 전체 실행
python run_benchmark.py --all --runs 1
```

---

## 설계 원칙

- **venv 격리**: 모델마다 의존성 충돌 방지를 위해 별도 Python 가상환경 사용
- **서브프로세스 실행**: `run_model.py`를 각 모델의 Python으로 독립 실행 후 JSON stdout으로 결과 수집
- **누적 저장**: 실행할 때마다 타임스탬프 파일로 저장 (덮어쓰지 않음)
- **WAV 보관**: 같은 텍스트로 생성한 음성을 모델별로 보관해 귀로도 비교 가능
