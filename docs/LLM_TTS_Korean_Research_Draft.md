# LLM 기반 한국어 TTS 모델 조사 보고서 (2026-03-24 업데이트)

> 목적: 한국어/다국어 TTS 모델 현황 조사, 상용 가능 모델 선별, 벤치마크 계획 확정
> 버전: 2026-03-24 (JA 재테스트 완료, CER 정규화 버그 수정 반영)

---
## 1. 조사 기준 (5대 전제)
1) 한국어 공식 지원 또는 공인 예제 존재
2) 가중치·코드 공개 + 상업 가능 라이선스(우선: Apache2.0/MIT)
3) G2P/phoneme 전처리 없이 텍스트 직접 입력
4) 실시간성: RTF < 1.0을 우선, 0.7 미만이면 가산점
5) 멀티스피커 zero-shot 3~10초 참조로 동작

## 2. 측정 지표
- RTF, TTFA, 추론 시간, 오디오 길이, VRAM peak
- 품질: CER/WER(Whisper), 주관 품질 코멘트, SIM(화자 유사도, WavLM-SV)
- 참고: 스트리밍 가능 여부, 인라인 컨트롤(tag) 지원 여부

## 3. 현재 벤치마크 결과 요약

### 3-A. 한국어(KO) ? RTF < 1.0 모델
| 순위 | 모델 | KO RTF | KO CER | VRAM | 라이선스 | 비고 |
|---|---|---|---|---|---|---|
| 1 | ZipVoice-FT | 0.003 | 5.7% | N/A(REST) | GPL-3.0 | Docker REST API, 내부 엔진 |
| 2 | Hierspeech++ | 0.035 | ? | ~4.9GB | Apache 2.0 | KO 특화, CER 미측정 |
| 3 | MeloTTS-KO | 0.047 | 5.3% | ~0.9GB | MIT | 고속, 고품질 |
| 4 | OpenVoice-v2 | 0.116 | 5.4% | ~1.3GB | MIT | MeloTTS 기반 |
| 5 | GPT-SoVITS V4 | 0.402 | 17.3% | ~2.0GB | MIT | ref prefix 제거 재측정 |
| 6 | XTTS-v2 | 0.455 | 42.2% | ~2.0GB | CPML | 17언어, 상업 제한 |
| 7 | GPT-SoVITS V3 | 0.546 | 17.6% | ~2.2GB | MIT | ref prefix 제거 재측정 |
| 8 | CosyVoice3-0.5B | 0.558 | 95.3% | ~3.6GB | Apache 2.0 | ZH 특화, KO 품질 낮음 |
| 9 | CosyVoice2-0.5B | 0.676 | 102.2% | ~4.6GB | Apache 2.0 | ZH 특화, KO 품질 낮음 |

> KO CER 기준: Whisper large-v3 / 텍스트 10종 평균. MeloTTS·ZipVoice·OpenVoice가 CER 5~6%로 우수.

### 3-B. 영어(EN) ? 주요 모델
| 모델 | EN RTF | EN CER | VRAM | 라이선스 | 비고 |
|---|---|---|---|---|---|
| Kokoro-82M | 0.011 | 0.9% | ~0.8GB | Apache 2.0 | EN 최고속·최고품질 |
| MeloTTS-KO | 0.015 | 1.5% | ~1.1GB | MIT | KO/EN 겸용 |
| OpenVoice-v2 | 0.099 | 2.8% | ~1.4GB | MIT | KO/EN 겸용 |
| XTTS-v2 | 0.300 | 1.9% | ~2.0GB | CPML | 17언어 |
| StyleTTS2 | 0.306 | 7.2% | CPU | MIT | EN 전용 |
| F5-TTS | 0.302 | 66.2% | ~0.8GB | CC-BY-NC | 비상업 |
| GPT-SoVITS V4 | 0.327 | 39.7% | ~2.2GB | MIT | en_female ref 오염→en_male ref 재측정 |
| GPT-SoVITS V3 | 0.377 | 6.7% | ~2.3GB | MIT | EN 품질 우수 |
| MioTTS-2.6B | 0.524 | 0.6% | ~5.9GB | Apache 2.0 | EN 고품질 |
| Qwen3-TTS 1.7B | 1.580 | 4.4% | ~4.7GB | Apache 2.0 | 다국어, EN도 우수 |

### 3-C. 일본어(JA) ? 재테스트 완료 (2026-03-24)
> 참조음성 오염(YouTube 아웃트로) 수정 후 재측정. CER 정규화 버그(일본어 문자 제거) 수정 포함.

| 모델 | JA RTF | JA CER | VRAM | 비고 |
|---|---|---|---|---|
| Kokoro-82M | 0.009 | ? | ~1.1GB | CER 미측정 |
| CosyVoice3-0.5B | 0.683 | 97.6% | ~3.6GB | ZH 특화, JA 품질 낮음 |
| MioTTS-2.6B | 0.597 | ? | ~5.9GB | EN/JA 지원 |
| MioTTS-1.7B | 0.725 | ? | ~4.3GB | EN/JA 지원 |
| GPT-SoVITS-V3 | 0.828 | 111.4% | ~2.1GB | Chinese G2P 우회(pyopenjtalk 미설치) |
| CosyVoice2-0.5B | 0.645 | 168.9% | ~3.5GB | ZH 특화, JA 품질 매우 낮음 |
| Qwen3-TTS-0.6B | 2.540 | 65.4% | ~2.8GB | RTF 느리나 품질↑ |
| Qwen3-TTS-1.7B | 3.540 | 53.4% | ~4.7GB | JA 최고 품질(CER 기준) |

### 3-D. 중국어(ZH) ? 주요 모델
| 모델 | ZH RTF | ZH CER | VRAM | 비고 |
|---|---|---|---|---|
| Kokoro-82M | 0.010 | 98.9% | ~1.0GB | ZH 실질 미지원(발음 불안정) |
| GPT-SoVITS V3 | 0.358 | 85.5% | ~2.2GB | ZH 음소 처리 한계 |
| F5-TTS | 0.435 | 74.9% | ~0.8GB | CC-BY-NC |
| CosyVoice2-0.5B | 0.557 | 119.2% | ~4.0GB | ZH 지원하나 CER 높음 |
| CosyVoice3-0.5B | 0.560 | 99.2% | ~3.6GB | |
| Kani-TTS-370M | 0.768 | ? | ~1.7GB | |
| IndexTTS-2 | 0.946 | 0.0% | ~7.6GB | ZH 특화 최고품질 |
| MaskGCT | 1.303 | 18.5% | ~9.7GB | CC-BY-NC |
| Qwen3-TTS-1.7B | 1.819 | 19.9% | ~4.7GB | |
| Qwen3-TTS-0.6B | 1.963 | 54.3% | ~2.8GB | |
| Spark-TTS-0.5B | 1.073 | 5.8% | ~4.3GB | ZH 우수 |

## 4. 즉시 테스트/추가 후보 (2026-03)
- Supertonic v2 (Supertone): MIT+OpenRAIL-M, KO/EN/ES/PT/FR, **공식 RTF 미공개/모델 비공개 배포** → 내부 측정 필요
- Chatterbox Multilingual (Resemble AI): MIT, 23언어, TTS Arena V2 1위(ELO 1501, 2025.09 기준), 마케팅 기준 지연 <200ms(실측 필요) ?cite?turn0search12?
- Higgs Audio V2.5 (Boson AI): Apache 2.0, KO 1위 주장(약 10만 시간 데이터), **가중치 미공개·서비스형**, RTF 미공개
- Fish Audio S2 (Fish Audio): HF 모델카드 기본 Research License(비상업), 일부 문서에 Apache 2.0 병기 → 상업 시 별도 계약; 공식 블로그에 실시간 지향 수치 제시(내부 검증 필요); Dual-AR(4B+400M)+Firefly-GAN; 80+언어; 인라인 emotion/화자 태그; SGLang 스트리밍; 벤치 미실행
- GPT-SoVITS V4 (RVC-Boss): MIT, 48kHz, multi-lingual, V3 개선판; **벤치마크 완료** ? KO RTF=0.402, EN RTF=0.327, KO CER=17.3% (ref prefix 제거 재측정), EN CER=39.7% (en_male ref 기준; en_female ref 오염→hallucination 확인됨)

## 5. 라이선스 주의 / 상업 불가
- Fish-Speech 1.5: CC-BY-NC-SA 4.0 (비상업)
- ChatTTS, F5-TTS: CC-BY-NC (비상업)
- SeamlessM4T v2: CC-BY-NC
- XTTS-v2: CPML (상업 제한)
## 6. 한국어 미지원/조건부 모델 (참고용)
- SparkTTS, IndexTTS-2/2.5: ZH/EN 위주, KO 공식 미지원
- Dia-1.6B, StyleTTS2, Parler-TTS, MetaVoice-1B: EN 위주
- Kyutai Pocket TTS, Kokoro v0.x: EN/JA/ZH 중심, KO 미지원

## 6-1. 내부 벤치마크 완료 모델 (전체 결과)

> 측정 환경: RTX 5080 16GB, CUDA 12.9, Python 3.x. RTF = 추론시간/오디오길이. CER = Whisper large-v3 기준.

| 모델 | 지원 언어 | 라이선스 | KO RTF | KO CER | EN RTF | EN CER | JA RTF | JA CER | ZH RTF | ZH CER | VRAM | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ZipVoice-FT | KO 전용 | GPL-3.0 | **0.003** | 5.7% | ? | ? | ? | ? | ? | ? | N/A | REST API, 내부 운용 중 |
| Hierspeech++ | KO/EN | Apache-2.0 | **0.035** | ? | 0.142 | ? | ? | ? | ? | ? | ~4.9GB | KO 특화, 실시간 가능 |
| MeloTTS-KO | KO/EN | MIT | **0.047** | 5.3% | **0.015** | 1.5% | ? | ? | ? | ? | ~0.9GB | 고속·저자원 |
| OpenVoice-v2 | KO/EN | MIT | 0.116 | 5.4% | 0.099 | 2.8% | ? | ? | ? | ? | ~1.3GB | MeloTTS 기반 TCC |
| GPT-SoVITS V4 | KO/EN/JA/ZH | MIT | 0.402 | 17.3% | 0.327 | 39.7% | ? | ? | ? | ? | ~2.0GB | KO: ref prefix 제거 재측정; EN: en_male ref 기준 |
| XTTS-v2 | KO/EN 외 15개 | CPML | 0.455 | 42.2% | 0.300 | 1.9% | ? | ? | ? | ? | ~2.0GB | 상업 제한 |
| GPT-SoVITS V3 | KO/EN/JA/ZH | MIT | 0.546 | 17.6% | 0.377 | 6.7% | 0.828 | 111.4%† | 0.358 | 23.5% | ~2.2GB | KO/ZH: ref prefix 제거 재측정; †JA: Chinese G2P 우회 |
| CosyVoice3-0.5B | KO/EN/JA/ZH | Apache-2.0 | 0.558 | 95.3% | 0.532 | 100.7% | 0.683 | 97.6% | 0.560 | 99.2% | ~3.6GB | ZH 특화 |
| CosyVoice2-0.5B | KO/EN/JA/ZH | Apache-2.0 | 0.676 | 102.2% | 0.568 | 97.9% | 0.645 | 168.9% | 0.557 | 119.2% | ~4.6GB | ZH 특화 |
| MaskGCT | KO/EN/ZH | CC-BY-NC-4.0 | 1.099 | 101.3% | 0.872 | 2.6% | ? | ? | 1.303 | 18.5% | ~9.7GB | 비상업 |
| VoiceCraft-X | KO/EN | CPML | 2.635 | 96.0% | 1.939 | 98.1% | ? | ? | ? | ? | ~4.0GB | 실시간 불가, 가중치 CPML·코드 CC BY-NC-SA |
| Qwen3-TTS-0.6B | KO/EN/JA/ZH 외 | Apache-2.0 | 2.639 | 3.6% | 1.543 | 4.6% | 2.540 | 65.4% | 1.963 | 54.3% | ~2.7GB | KO 품질 우수, RTF 느림 |
| Qwen3-TTS-1.7B | KO/EN/JA/ZH 외 | Apache-2.0 | 1.989 | 3.7% | 1.580 | 4.4% | 3.540 | **53.4%** | 1.819 | 19.9% | ~4.7GB | 다국어 최고품질 |
| OuteTTS-1.0-0.6B | KO/EN/JA/ZH | CC-BY-NC-SA-4.0 | 3.234 | 9.7% | 3.555 | 2.2% | 3.283 | ? | 3.219 | 8.6% | ~4.0GB | v0.3 비상업(Emilia 포함), 실시간 불가·RTF 편차 큼 |
| Kani-TTS-370M | KO/EN/JA/ZH | Apache-2.0 | 4.925 | ? | 0.904 | ? | ? | ? | 0.768 | ? | ~1.8GB | KO 실시간 불가 |
| Kokoro-82M | EN/JA/ZH 외 | Apache-2.0 | ? | ? | **0.011** | **0.9%** | 0.009 | ? | 0.010 | 98.9% | ~0.8GB | EN 최속, KO 미지원 |
| MioTTS-1.7B | EN/JA | Apache-2.0 | ? | ? | 0.853 | 1.5% | 0.725 | ? | ? | ? | ~4.3GB | 공식 벤치 부족, 커뮤니티 WER 의존 |
| MioTTS-2.6B | EN/JA | Apache-2.0 | ? | ? | 0.524 | **0.6%** | 0.597 | ? | ? | ? | ~5.9GB | 공식 벤치 부족, 커뮤니티 WER 의존 |
| Fish-Speech-1.5 | EN/JA/ZH | CC-BY-NC-SA | ? | ? | 1.081 | 25.3% | 1.343 | ? | 1.482 | 42.4% | ~1.7GB | 비상업 |
| IndexTTS-2 | ZH/EN | Apache-2.0 | ? | ? | 0.832 | ? | 1.012 | ? | 0.946 | **0.0%** | ~7.6GB | ZH 최고품질 |
| Spark-TTS-0.5B | ZH/EN | Apache-2.0 | ? | ? | 0.996 | 0.9% | ? | ? | 1.073 | 5.8% | ~4.3GB | |
| F5-TTS | ZH/EN | CC-BY-NC | ? | ? | 0.302 | 66.2% | ? | ? | 0.435 | 74.9% | ~0.8GB | 비상업 |
| StyleTTS2 | EN | MIT | ? | ? | 0.306 | 7.2% | ? | ? | ? | ? | CPU | EN 전용 |
| Parler-TTS-mini | EN | Apache-2.0 | ? | ? | 1.184 | 9.0% | ? | ? | ? | ? | ~5.1GB | v1(mini) 공식 리포트 부족, 설명 기반 제어 |
| Chatterbox-TTS | EN | MIT | ? | ? | 0.639 | 2.0% | ? | ? | ? | ? | ~3.9GB | EN 전용 |
| MARS5-TTS | EN | AGPL-3.0 | ? | ? | 2.704 | 63.8% | ? | ? | ? | ? | ~6.8GB | 카피레프트 |
| Bark | EN | MIT | ? | ? | 1.798 | 7.2% | ? | ? | ? | ? | ~5.1GB | |
| Dia-1.6B | EN | Apache-2.0 | ? | ? | 4.449 | 190.7% | ? | ? | ? | ? | ~6.9GB | 대화형 포맷 |
| Piper-TTS | EN | MPL-2.0 | ? | ? | ? | ? | ? | ? | ? | ? | ? | 오프라인, `--output-raw` stdout 기반 부분 스트리밍 |

## 6-2. 스트리밍 지원 여부 및 지원 방식 (검증일: 2026-03-25)

> 판정 기준  
> `지원`: 공식 문서/코드에 스트리밍 모드 또는 스트리밍 API가 명시됨  
> `부분 지원`: stdout/raw chunk 등 제한적 방식(전용 실시간 API와 구분)  
> `미확인`: 내부 자료에는 있으나 공식 문서에서 방식 확인이 어려움  
> `미지원(미명시)`: 공식 문서에서 스트리밍 모드 언급 없음(배치 합성 중심)

### A) 공식 근거로 방식 확인된 모델
| 모델 | 상태 | 지원 방식 | 공식 근거 |
|---|---|---|---|
| CosyVoice2 / CosyVoice3 | 지원 | Bi-Streaming(text-in + audio-out), 스트리밍 추론 모드(EOS/청크 기반), 저지연 지향 | https://github.com/FunAudioLLM/CosyVoice |
| Qwen3-TTS-0.6B / 1.7B | 지원 | Dual-Track 하이브리드 구조로 스트리밍/비스트리밍 동시 지원, first packet 저지연 명시 | https://github.com/QwenLM/Qwen3-TTS |
| GPT-SoVITS V3 / V4 | 지원 | `api_v2.py`에서 `streaming_mode`(0~3), `return_fragment`, `StreamingResponse`로 chunk 응답 | https://github.com/RVC-Boss/GPT-SoVITS/blob/main/api_v2.py |
| XTTS-v2 | 지원 | `inference_stream()` API 및 Streaming manually 예제 제공, <200ms 지연 문구 명시 | https://docs.coqui.ai/en/latest/models/xtts.html |
| LLMVoX | 지원 | `streaming_server.py`, multi-queue streaming, chunk 파라미터(`initial_dump_size`) 제공 | https://github.com/mbzuai-oryx/LLMVoX |
| GLM-TTS | 지원 | README에 streaming inference 명시, `flow/flow.py` 실시간 생성 경로 설명 | https://github.com/zai-org/GLM-TTS |
| Fish Audio S2 Pro | 지원 | SGLang 기반 streaming inference engine 명시, TTFA/RTF 등 프로덕션 스트리밍 수치 제시 | https://huggingface.co/fishaudio/s2-pro |
| VoXtream2 | 지원 | Full-stream 지원(`--full-stream`), Python `generate_stream()` API, FPL/RTF 벤치 제공 | https://github.com/herimor/voxtream |
| Piper-TTS | 부분 지원 | `--output-raw`로 생성 중 PCM을 stdout으로 연속 출력(플레이어 파이프 연결 방식) | https://github.com/rhasspy/piper |

### B) 내부 표기와 공식 문서 간 간극(추가 검증 필요)
| 모델 | 현재 판정 | 확인 결과 |
|---|---|---|
| ZipVoice-FT | 부분 지원(내부) | 내부 REST 운용에서는 스트리밍 형태 사용 가능하나, 원본 ZipVoice 공식 문서에서 범용 streaming API는 명시 약함 |
| Fish-Speech-1.5 | 미확인 | 공식 리포는 S2 문서/서버 링크를 제공하지만, 1.5 자체의 표준 스트리밍 인터페이스 명시는 제한적 |
| MioTTS-1.7B / 2.6B | 미확인 | 공식 GitHub/HF에서 스트리밍 인터페이스 방식(엔드포인트/청크 프로토콜) 확인이 어려움 |
| Supertonic-v2 / Higgs Audio V2.5 | 미확인 | 서비스형/비공개 가중치 성격이 강해 오픈 문서 기준의 스트리밍 구현 방식 확인이 제한됨 |

### C) 미지원(또는 공식 미명시) 모델
- Hierspeech++, MeloTTS, OpenVoice-v2, OuteTTS, Kani-TTS, Kokoro, Bark, StyleTTS2, Parler-TTS-mini, Chatterbox-TTS, Dia-1.6B, MARS5-TTS, VoiceCraft-X, MaskGCT, IndexTTS-2, Spark-TTS, F5-TTS
- 위 모델들은 현재 공개 문서 기준으로는 배치 합성 중심이며, 표준 스트리밍 API/모드가 명확히 문서화되어 있지 않음
- 향후 공식 서버 문서가 추가되면 상태를 `미명시`에서 `지원`으로 재분류 필요

## 7. 제외 사유: 가중치 비공개
- Seed-TTS, NaturalSpeech3, VALL-E 2/3, BASE TTS, Voicebox, SoundStorm, E2-TTS 등

## 8. 다음 액션
- ~~Phase 1 (즉시): Fish Audio S2 → 라이선스(Research) 문제로 제외 확정. Supertonic v2 → zero-shot 클론 미지원으로 제외 확정. Chatterbox-ML → KO 품질 퇴보 사용자 보고로 제외 확정.~~
- ~~Phase 2: Higgs Audio V2.5 → V2.5 가중치 미공개/V2 VRAM 부족으로 제외 확정. GPT-SoVITS V4 → ? 완료 (KO RTF=0.402, EN RTF=0.327)~~
- **JA 재테스트 ? 완료 (2026-03-24)**: 참조음성 오염(YouTube 아웃트로 ja_female.wav) 수정 후 cosyvoice2/3/gpt_sovits/qwen3×2 재측정. CER 정규화 버그(일본어 문자 제거) 수정 포함.
- **미완료**: Hierspeech++ KO/JA CER 미측정, MioTTS JA CER 미측정, VoiceCraft-X/MaskGCT/Kani CER 미측정
- **추가 검토 대상**: VoXtream2 (herimor) → KO 실측 필요 (eSpeak NG 내장 KO 음소, Ubuntu 환경), RTF 0.173/TTFA 74ms
- **장기 계획**: SIM(화자 유사도, WavLM-SV) 지표 추가, 스트리밍 벤치(SGLang 등)

## 9. 참고 링크
- Fish Audio S2: https://huggingface.co/fishaudio/s2-pro , https://fish.audio/blog/fish-audio-open-sources-s2 , 라이선스: https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE
- Supertonic v2: https://github.com/supertone-inc/supertonic
- Chatterbox Multilingual: https://github.com/resemble-ai/chatterbox
- Higgs Audio V2.5: https://github.com/boson-ai/higgs-audio
- GPT-SoVITS V4: https://github.com/RVC-Boss/GPT-SoVITS
- VoXtream2: https://github.com/herimor/voxtream , HF: https://huggingface.co/herimor/voxtream2 , arXiv: 2603.13518
- ZipVoice-FT: https://github.com/k2-fsa/ZipVoice , 데모: https://zipvoice.github.io/ , HF: https://huggingface.co/k2-fsa/ZipVoice
- Hierspeech++: https://github.com/sh-lee-prml/HierSpeechpp , 데모: https://sh-lee-prml.github.io/HierSpeechpp-demo/
- MeloTTS: https://github.com/myshell-ai/MeloTTS
- Parler-TTS(-mini): https://github.com/huggingface/parler-tts
- Bark: https://github.com/suno-ai/bark
- Kani-TTS: https://github.com/nineninesix-ai/kani-tts , HF 카드 예: https://huggingface.co/nineninesix/kani-tts-370m
- OpenVoice-v2 파생 예시: https://github.com/Nyan-SouthKorea/RealTime_zeroshot_TTS_ko
- OuteTTS 데모: https://github.com/OuteAI/OuteTTS-0.3-1B-Demo

### 공식 리포지토리 및 가중치 주소
- ZipVoice-FT (ONNX/REST): GitHub https://github.com/thewh1teagle/zipvoice-onnx , HF https://huggingface.co/thewh1teagle/zipvoice-onnx
- Hierspeech++ (KO): GitHub https://github.com/sh0317/Hierspeechpp , HF https://huggingface.co/sh0317/Hierspeechpp
- MeloTTS: GitHub https://github.com/myshell-ai/MeloTTS , HF https://huggingface.co/myshell-ai/MeloTTS-Korean
- OpenVoice-v2: GitHub https://github.com/myshell-ai/OpenVoice , HF https://huggingface.co/myshell-ai/OpenVoiceV2
- GPT-SoVITS (V3/V4): GitHub https://github.com/RVC-Boss/GPT-SoVITS , HF https://huggingface.co/RVC-Boss/GPT-SoVITS
- XTTS-v2: GitHub https://github.com/coqui-ai/TTS , HF https://huggingface.co/coqui/XTTS-v2
- CosyVoice 2/3: GitHub https://github.com/FunAudioLLM/CosyVoice , HF https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
- Kokoro-82M: GitHub https://github.com/hexgrad/kokoro , HF https://huggingface.co/hexgrad/Kokoro-82M
- StyleTTS2: GitHub https://github.com/yl4579/StyleTTS2 , HF https://huggingface.co/yl4579/StyleTTS2-LJSpeech
- F5-TTS: GitHub https://github.com/SWivid/F5-TTS , HF https://huggingface.co/SWivid/F5-TTS
- MioTTS (1.7B/2.6B): GitHub https://github.com/Aratako/MioTTS-Inference , HF https://huggingface.co/Aratako/MioTTS-1.7B / https://huggingface.co/Aratako/MioTTS-2.6B
- Qwen3-TTS (0.6B/1.7B): GitHub https://github.com/QwenLM/Qwen3-TTS , HF https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base (0.6B: Qwen3-TTS-12Hz-0.6B-Base)
- IndexTTS-2: GitHub https://github.com/index-tts/index-tts , HF https://huggingface.co/IndexTeam/IndexTTS-2
- MaskGCT (Amphion): GitHub https://github.com/open-mmlab/Amphion , HF https://huggingface.co/amphion/MaskGCT
- Spark-TTS: GitHub https://github.com/SparkAudio/Spark-TTS , HF https://huggingface.co/SparkAudio/Spark-TTS-0.5B


## 10. 외부 벤치마크/리더보드 요약 (검증일: 2026-03-25)

### 10-1. Artificial Analysis Speech Leaderboard (공식 페이지 기준)
- 제공 지표: Quality ELO, 가격(USD / 1M chars), 생성속도(Characters/s), 생성시간
- FAQ 표시값(접속 시점):
  - #1 Inworld TTS 1.5 Max: ELO 1238
  - #2 Eleven v3: ELO 1197
  - #3 Inworld TTS 1 Max: ELO 1183
  - #4 Inworld TTS 1.5 Mini: ELO 1182
  - #5 Speech 2.8 HD: ELO 1175
- 오픈가중치 관련 표시값(접속 시점):
  - 최고 오픈가중치: Kokoro 82M v1.0 (ELO 1072)
  - 최저가 모델: Kokoro 82M v1.0 ($0.65 / 1M chars)
- 방법론 핵심: 22.05kHz 표준화, Arena ELO(블라인드 선호), 가격/생성시간 병행 비교
- 출처:
  - https://artificialanalysis.ai/text-to-speech/leaderboard
  - https://artificialanalysis.ai/text-to-speech/methodology

### 10-2. Hugging Face TTS Arena V2 (공개 리더보드 스냅샷)
- 공개 보드(Mar 2026 선택 기준) 상위권:
  - #1 Vocu V3.0: ELO 1581
  - #2 Inworld TTS: ELO 1577
  - #3 Inworld TTS MAX: ELO 1575
  - #4 CastleFlow v1.0: ELO 1574
  - #5 Hume Octave: ELO 1565
  - #6 Papla P1: ELO 1561
- 오픈 모델 참고:
  - Kokoro v1.0: #17, ELO 1500
  - StyleTTS 2: #24, ELO 1369
  - CosyVoice 2.0: #25, ELO 1358
  - Spark TTS: #26, ELO 1342
- 출처:
  - https://tts-agi-tts-arena-v2.hf.space/leaderboard
  - https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2

### 10-3. Inworld 2026 벤치 리포트 (벤더 리포트, 2026-01-22 발행)
- 실시간 대화형 관점 요약:
  - Best overall: Inworld TTS-1.5-Max (ELO 1,160 인용, sub-250ms)
  - Fastest TTFA: Cartesia Sonic 3 (40ms 인용)
  - 초저지연 옵션: Inworld TTS-1.5-Mini (P90 TTFA <130ms)
  - 가격: Inworld 1.5 Mini $5 / 1M chars, 1.5 Max $10 / 1M chars
- 주의: 본 문서는 벤더 발행 자료이며, 일부 수치는 외부 리더보드(AA)와 시점 차이로 달라질 수 있음
- 출처:
  - https://inworld.ai/resources/best-voice-ai-tts-apis-for-real-time-voice-agents-2026-benchmarks
  - https://docs.cartesia.ai/build-with-cartesia/tts-models/latest
  - https://aws.amazon.com/about-aws/whats-new/2026/02/cartesia-sonic-3-on-sagemaker-jumpstart/

### 10-4. 실무 적용 가이드 (우리 벤치마크와 결합)
- 종합 비교(품질·가격·속도): Artificial Analysis 우선 참고
- 사람 선호 기반 음질: HF TTS Arena V2 병행 확인
- 실시간 에이전트 지연(TTFA) 설계: Inworld 리포트 + Cartesia 공식 문서 교차 검증
- 권장 원칙: 외부 리더보드 수치는 "후보 축소" 용도로만 사용하고, 최종 채택은 내부 KO/EN/JA/ZH 동일 프롬프트 재측정으로 확정

