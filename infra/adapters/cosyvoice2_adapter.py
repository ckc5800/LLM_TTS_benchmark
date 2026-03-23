"""
CosyVoice 2.0 (FunAudioLLM/CosyVoice2-0.5B) 어댑터
- 한국어: cross-lingual (<|ko|> prefix) 지원
- Voice clone / zero-shot 지원
"""
from __future__ import annotations
import sys
import os
import time

# CosyVoice repo path
COSYVOICE_REPO = os.path.join(os.path.dirname(__file__), '..', 'cosyvoice')
sys.path.insert(0, os.path.abspath(COSYVOICE_REPO))
sys.path.insert(0, os.path.abspath(os.path.join(COSYVOICE_REPO, 'third_party', 'Matcha-TTS')))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmark.base_adapter import BaseTTSAdapter


class CosyVoice2Adapter(BaseTTSAdapter):
    model_name = "CosyVoice2-0.5B"
    model_version = "2.0"
    model_size_params = "500M"
    korean_support = True
    model_hf_id = "FunAudioLLM/CosyVoice2-0.5B"

    # cross-lingual 한국어 테스트용 레퍼런스 오디오 (cosyvoice repo 내장)
    REF_WAV = os.path.join(COSYVOICE_REPO, 'asset', 'zero_shot_prompt.wav')
    NOTES = "cross-lingual <|ko|> 사용, campplus ONNX CPU (no CUDA EP)"

    def _download(self):
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=self.model_hf_id,
            local_dir=self.model_dir,
            ignore_patterns=['*.gitattributes'],
        )

    def _is_downloaded(self) -> bool:
        return os.path.exists(os.path.join(self.model_dir, 'cosyvoice2.yaml'))

    def _load_model(self):
        from cosyvoice.cli.cosyvoice import AutoModel
        self._model = AutoModel(
            model_dir=self.model_dir,
            load_jit=False,
            load_trt=False,
        )

    def _synthesize(self, text: str) -> tuple:
        import torch
        # cross-lingual으로 한국어 생성
        ko_text = f"<|ko|>{text}"
        ref_wav = self.REF_WAV if os.path.exists(self.REF_WAV) else None

        if ref_wav is None:
            raise RuntimeError("Reference WAV not found for cross-lingual test")

        chunks = []
        ttfa_ms = -1.0
        t0 = time.perf_counter()

        for i, result in enumerate(self._model.inference_cross_lingual(
            ko_text, ref_wav, stream=False
        )):
            chunk = result['tts_speech']
            if i == 0:
                ttfa_ms = (time.perf_counter() - t0) * 1000
            chunks.append(chunk)

        if not chunks:
            raise RuntimeError("No audio generated")

        audio = torch.cat(chunks, dim=-1)
        return audio, self._model.sample_rate, ttfa_ms

    def _unload_model(self):
        import gc
        import torch
        del self._model
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
