# LLM 기반 한국어 TTS 모델 조사 보고서 (2026-03-23 재작성)

> 목적: 한국어/다국어 TTS 모델 현황 조사, 상용 가능 모델 선별, 벤치마크 계획 확정
> 버전: 2026-03-23 (이전 파일 손상으로 재작성)

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

## 3. 현재 벤치마크 통과 모델 (RTF 기준 상위)
| 순위 | 모델 | KO RTF | EN RTF | VRAM | 라이선스 | 비고 |
|---|---|---|---|---|---|---|
| 1 | GPT-SoVITS V3 | 0.546 | 0.377 | ~2.1GB | MIT | KO/EN 품질 우수, ref 필수 |
| 2 | Qwen3-TTS 0.6B | 0.547 | 0.501 | ~2.7GB | Apache 2.0 | 10언어, AR codec |
| 3 | Qwen3-TTS 1.7B | 0.556 | 0.500 | ~4.3GB | Apache 2.0 | 품질↑, 자원↑ |
| 4 | CosyVoice3-0.5B | 0.558 | 0.532 | ~3.6GB | Apache 2.0 | EOS 트리거, zero-shot |
| 5 | CosyVoice2-0.5B | 0.676 | 0.568 | ~4.1GB | Apache 2.0 | 품질 양호, RTF 느림 |

## 4. 즉시 테스트/추가 후보 (2026-03)
- Supertonic v2 (Supertone): MIT+OpenRAIL-M, KO/EN/ES/PT/FR, RTF ~0.001 보고
- Chatterbox Multilingual (Resemble AI): MIT, 23언어, TTS Arena V2 1위(ELO 1501, 2025.09 기준)
- Higgs Audio V2.5 (Boson AI): Apache 2.0, KO 1위 주장(약 10만 시간 데이터), RTF 미공개
- Fish Audio S2 (Fish Audio): HF 모델카드 기본은 Research License(비상업), 일부 문서에 Apache 2.0 병기 → 상업 시 별도 계약; H200 RTF≈0.195·TTFA<100ms; Dual-AR(4B+400M)+Firefly-GAN; 80+언어; 인라인 emotion/화자 태그; SGLang 스트리밍; 벤치 미실행(보고서 미포함)
- GPT-SoVITS V4 (RVC-Boss): MIT, 48kHz, multi-lingual, V3 개선판(미측정)

## 5. 라이선스 주의 / 상업 불가
- Fish-Speech 1.5: CC-BY-NC-SA 4.0 (비상업)
- ChatTTS, F5-TTS: CC-BY-NC (비상업)
- SeamlessM4T v2: CC-BY-NC
- XTTS-v2: CPML (상업 제한)

## 6. 한국어 미지원/조건부 모델 (참고용)
- SparkTTS, IndexTTS-2/2.5: ZH/EN 위주, KO 공식 미지원
- Dia-1.6B, StyleTTS2, Parler-TTS, MetaVoice-1B: EN 위주
- Kyutai Pocket TTS, Kokoro v0.x: EN/JA/ZH 중심, KO 미지원

## 7. 제외 사유: 가중치 비공개
- Seed-TTS, NaturalSpeech3, VALL-E 2/3, BASE TTS, Voicebox, SoundStorm, E2-TTS 등

## 8. 다음 액션
- Phase 1 (즉시): Fish Audio S2, Supertonic v2, Chatterbox Multi → KO RTF/TTFA 측정, CER/SIM 추가
- Phase 2: Higgs Audio V2.5, GPT-SoVITS V4 → 라이선스/RTF 확인 후 실행
- Phase 3: 스트리밍 벤치(SGLang 등) 및 MOS 라인업 준비

## 9. 참고 링크
- Fish Audio S2: https://huggingface.co/fishaudio/s2-pro , https://fish.audio/blog/fish-audio-open-sources-s2 , 라이선스: https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE
- Supertonic v2: https://github.com/supertone-inc/supertonic
- Chatterbox Multilingual: https://github.com/resemble-ai/chatterbox
- Higgs Audio V2.5: https://github.com/boson-ai/higgs-audio
- GPT-SoVITS V4: https://github.com/RVC-Boss/GPT-SoVITS
