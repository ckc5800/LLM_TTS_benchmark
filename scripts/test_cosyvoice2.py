"""
CosyVoice 2.0 Quick Inference Test
모델: FunAudioLLM/CosyVoice2-0.5B
테스트: 한국어 zero-shot voice clone
"""
import sys
import os

# CosyVoice repo path setup
COSYVOICE_DIR = os.path.join(os.path.dirname(__file__), 'cosyvoice')
sys.path.insert(0, COSYVOICE_DIR)
sys.path.insert(0, os.path.join(COSYVOICE_DIR, 'third_party', 'Matcha-TTS'))

# Model will be downloaded here
MODEL_DIR = r'D:\models\CosyVoice2-0.5B'
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'cosyvoice2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_model():
    """HuggingFace에서 모델 다운로드"""
    if os.path.exists(os.path.join(MODEL_DIR, 'cosyvoice.yaml')):
        print(f"[SKIP] 모델 이미 존재: {MODEL_DIR}")
        return
    print(f"[DOWNLOAD] CosyVoice2-0.5B → {MODEL_DIR}")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id='FunAudioLLM/CosyVoice2-0.5B',
        local_dir=MODEL_DIR,
        ignore_patterns=['*.gitattributes'],
    )
    print("[DOWNLOAD] 완료")

def test_sft(cosyvoice):
    """SFT 내장 화자 테스트 (중국어 텍스트 사용)"""
    import torchaudio
    print("\n[TEST 1] SFT 내장 화자 목록:")
    spks = cosyvoice.list_available_spks()
    print(f"  사용 가능한 화자: {spks}")

    # SFT 화자는 중국어 기반
    test_text = "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
    spk = spks[0]
    print(f"\n[TEST 1] SFT 화자 '{spk}' 로 생성 중...")
    for i, result in enumerate(cosyvoice.inference_sft(test_text, spk, stream=False)):
        out_path = os.path.join(OUTPUT_DIR, f'sft_{spk.replace("/","_")}_{i}.wav')
        torchaudio.save(out_path, result['tts_speech'], cosyvoice.sample_rate)
        print(f"  → 저장: {out_path}")

def test_korean_cross_lingual(cosyvoice, ref_wav_path):
    """한국어 cross-lingual 테스트"""
    import torchaudio
    ko_text = "<|ko|>안녕하세요. 저는 코시보이스 이점영 TTS 모델입니다. 한국어 음성 합성 품질을 테스트하고 있습니다."
    print(f"\n[TEST 2] 한국어 cross-lingual: {ko_text[:40]}...")
    for i, result in enumerate(cosyvoice.inference_cross_lingual(ko_text, ref_wav_path, stream=False)):
        out_path = os.path.join(OUTPUT_DIR, f'korean_cross_lingual_{i}.wav')
        torchaudio.save(out_path, result['tts_speech'], cosyvoice.sample_rate)
        print(f"  → 저장: {out_path}")

def test_zero_shot_with_reference(cosyvoice, ref_wav_path, ref_text):
    """Zero-shot voice clone 테스트 (레퍼런스 오디오 필요)"""
    import torchaudio

    ko_text = "안녕하세요. 이것은 제로샷 음성 복제 테스트입니다. 한국어 음성 합성 품질을 확인하고 있습니다."
    print(f"\n[TEST 2] Zero-shot voice clone")
    print(f"  ref: {ref_wav_path}")
    print(f"  text: {ko_text[:30]}...")

    for i, result in enumerate(cosyvoice.inference_zero_shot(
        ko_text, ref_text, ref_wav_path, stream=False
    )):
        out_path = os.path.join(OUTPUT_DIR, f'zero_shot_{i}.wav')
        torchaudio.save(out_path, result['tts_speech'], cosyvoice.sample_rate)
        print(f"  → 저장: {out_path}")

def main():
    import time

    print("=" * 60)
    print("CosyVoice 2.0 추론 테스트")
    print("=" * 60)

    # 1. 모델 다운로드
    download_model()

    # 2. 모델 로드
    print(f"\n[LOAD] 모델 로딩 중: {MODEL_DIR}")
    t0 = time.time()
    from cosyvoice.cli.cosyvoice import AutoModel
    cosyvoice = AutoModel(
        model_dir=MODEL_DIR,
        load_jit=False,
        load_trt=False,
    )
    print(f"[LOAD] 완료 ({time.time()-t0:.1f}s), sample_rate={cosyvoice.sample_rate}")

    # 3. SFT 테스트 (화자 있을 때만)
    spks = cosyvoice.list_available_spks()
    if spks:
        test_sft(cosyvoice)
    else:
        print("\n[TEST 1] SFT 화자 없음 (zero-shot 전용 모델) - 스킵")

    # 4. 한국어 cross-lingual (레퍼런스 오디오 사용)
    ref_wav = os.path.join(COSYVOICE_DIR, 'asset', 'zero_shot_prompt.wav')
    if os.path.exists(ref_wav):
        test_korean_cross_lingual(cosyvoice, ref_wav)

    # 5. Zero-shot voice clone
    if os.path.exists(ref_wav):
        test_zero_shot_with_reference(
            cosyvoice,
            ref_wav_path=ref_wav,
            ref_text='希望你以后能够做的比我还好呦。'
        )

    print("\n[DONE] 테스트 완료!")
    print(f"출력 파일: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
