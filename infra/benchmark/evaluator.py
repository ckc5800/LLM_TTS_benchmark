
import os
import re
import difflib

try:
    from faster_whisper import WhisperModel
    _BACKEND = "faster_whisper"
except ImportError:
    try:
        import whisper as _whisper_module
        _BACKEND = "openai_whisper"
    except ImportError:
        _BACKEND = None


class TTSQualityEvaluator:
    def __init__(self, model_size="base", device="cuda", compute_type="float16"):
        print(f"품질 평가기 로드 중 (backend={_BACKEND}, model={model_size})...")
        if _BACKEND == "faster_whisper":
            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        elif _BACKEND == "openai_whisper":
            import torch
            dev = device if torch.cuda.is_available() else "cpu"
            self._model = _whisper_module.load_model(model_size, device=dev)
        else:
            raise ImportError("faster-whisper 또는 openai-whisper가 필요합니다.")
        self._backend = _BACKEND

    def transcribe(self, audio_path: str, language: str = "ko") -> str:
        """오디오를 텍스트로 변환 (STT)"""
        if not os.path.exists(audio_path):
            return ""
        if self._backend == "faster_whisper":
            segments, info = self._model.transcribe(audio_path, language=language, beam_size=5)
            return "".join(seg.text for seg in segments).strip()
        else:
            result = self._model.transcribe(audio_path, language=language)
            return result["text"].strip()

    def calculate_metrics(self, reference_text: str, hypothesis_text: str) -> dict:
        """WER, CER 계산"""
        ref_chars = self._normalize(reference_text)
        hyp_chars = self._normalize(hypothesis_text)

        cer = self._error_rate(list(ref_chars), list(hyp_chars))

        ref_words = reference_text.split()
        hyp_words = hypothesis_text.split()
        wer = self._error_rate(ref_words, hyp_words)

        return {
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "stt_text": hypothesis_text,
        }

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^가-힣a-zA-Z0-9]', '', text)
        return text

    def _error_rate(self, ref, hyp):
        if not ref:
            return 1.0 if hyp else 0.0
        sm = difflib.SequenceMatcher(None, ref, hyp)
        errors = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'replace':
                errors += max(i2 - i1, j2 - j1)
            elif tag == 'insert':
                errors += (j2 - j1)
            elif tag == 'delete':
                errors += (i2 - i1)
        return errors / len(ref)


if __name__ == "__main__":
    evaluator = TTSQualityEvaluator()
    ref = "안녕하세요 오늘 날씨가 정말 좋습니다."
    hyp = "안녕하세요 오늘 날씨는 정말 좋습니다."
    print(evaluator.calculate_metrics(ref, hyp))
