"""
다국어 참조 음성 다운로드 스크립트 (v3 - FLEURS 기반)
대상: EN(여/남), JA(여/남), ZH(여/남)
저장: D:\tts-benchmark\references\{key}.wav + {key}.txt
소스: google/fleurs (gender=0 남성, gender=1 여성)
"""

import os
import numpy as np
import soundfile as sf

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
REF_DIR = os.path.join(ROOT_DIR, "references")
os.makedirs(REF_DIR, exist_ok=True)
TARGET_SEC = 12.0  # 최대 길이


def save_ref(key, audio, sr, text):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.array(audio, dtype=np.float32)
    audio = audio[:int(TARGET_SEC * sr)]
    sf.write(os.path.join(REF_DIR, f"{key}.wav"), audio, sr, subtype="PCM_16")
    with open(os.path.join(REF_DIR, f"{key}.txt"), "w", encoding="utf-8") as f:
        f.write(text.strip())
    dur = len(audio) / sr
    preview = text[:70] + ("..." if len(text) > 70 else "")
    print(f"  OK [{key}] {dur:.1f}s | {preview}")


def skip(key):
    if os.path.exists(os.path.join(REF_DIR, f"{key}.wav")):
        print(f"  [스킵] {key} 이미 존재")
        return True
    return False


def download_fleurs(lang_code, gender_int, key, split="test",
                    min_dur=7.0, max_dur=18.0):
    """FLEURS에서 특정 언어·성별의 샘플 1개 다운로드.
    gender_int: 0=남성, 1=여성
    """
    if skip(key):
        return True
    print(f"\n[{key}] FLEURS '{lang_code}' gender={'여성' if gender_int == 1 else '남성'}...")
    from datasets import load_dataset
    try:
        ds = load_dataset(
            "google/fleurs", lang_code,
            split=split, streaming=True, trust_remote_code=True,
        )
        for sample in ds:
            if sample.get("gender") != gender_int:
                continue
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            dur = len(audio) / sr
            text = sample.get("transcription", "").strip()
            if not text or not (min_dur <= dur <= max_dur):
                continue
            save_ref(key, audio, sr, text)
            return True
        # min_dur 기준 완화해서 재시도
        ds2 = load_dataset(
            "google/fleurs", lang_code,
            split=split, streaming=True, trust_remote_code=True,
        )
        for sample in ds2:
            if sample.get("gender") != gender_int:
                continue
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            dur = len(audio) / sr
            text = sample.get("transcription", "").strip()
            if not text or dur < 3.0:
                continue
            save_ref(key, audio, sr, text)
            return True
        print(f"  [실패] 조건에 맞는 샘플 없음")
        return False
    except Exception as e:
        print(f"  [오류] {e}")
        return False


# FLEURS 언어 코드
# EN: en_us, JA: ja_jp, ZH: cmn_hans_cn
TARGETS = [
    # (lang_code,      gender, key,         gender_int)
    ("en_us",          1,      "en_female"),
    ("en_us",          0,      "en_male"),
    ("ja_jp",          1,      "ja_female"),
    ("ja_jp",          0,      "ja_male"),
    ("cmn_hans_cn",    1,      "zh_female"),
    ("cmn_hans_cn",    0,      "zh_male"),
]

if __name__ == "__main__":
    print("=" * 60)
    print("다국어 참조 음성 다운로드 (FLEURS)")
    print(f"저장 위치: {REF_DIR}")
    print("=" * 60)

    try:
        import datasets
        print(f"datasets 버전: {datasets.__version__}")
    except ImportError:
        print("pip install datasets 필요")
        raise

    for lang_code, gender_int, key in TARGETS:
        download_fleurs(lang_code, gender_int, key)

    print("\n" + "=" * 60)
    print("결과:")
    for f in sorted(os.listdir(REF_DIR)):
        if not f.endswith(".wav"):
            continue
        k = f[:-4]
        wav_p = os.path.join(REF_DIR, f)
        txt_p = os.path.join(REF_DIR, k + ".txt")
        try:
            info = sf.info(wav_p)
            txt = open(txt_p, encoding="utf-8").read()[:60] if os.path.exists(txt_p) else "?"
            print(f"  {k:20s} {info.duration:.1f}s  '{txt}'")
        except Exception:
            print(f"  {k}")
    print("=" * 60)
