"""
CosyVoice 3.0 (FunAudioLLM/Fun-CosyVoice3-0.5B-2512) 어댑터
- 한국어: cross-lingual 지원
- 9개 언어 + 18개+ 중국어 방언
"""
from __future__ import annotations
import sys
import os
import time

COSYVOICE_REPO = os.path.join(os.path.dirname(__file__), '..', 'cosyvoice')
sys.path.insert(0, os.path.abspath(COSYVOICE_REPO))
sys.path.insert(0, os.path.abspath(os.path.join(COSYVOICE_REPO, 'third_party', 'Matcha-TTS')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmark.base_adapter import BaseTTSAdapter


class CosyVoice3Adapter(BaseTTSAdapter):
    model_name = "CosyVoice3-0.5B"
    model_version = "3.0"
    model_size_params = "500M"
    korean_support = True
    model_hf_id = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

    REF_WAV = os.path.join(COSYVOICE_REPO, 'asset', 'zero_shot_prompt.wav')
    REF_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

    def _download(self):
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=self.model_hf_id,
            local_dir=self.model_dir,
            ignore_patterns=['*.gitattributes'],
        )

    def _is_downloaded(self) -> bool:
        return os.path.exists(os.path.join(self.model_dir, 'cosyvoice3.yaml'))

    def _load_model(self):
        from cosyvoice.cli.cosyvoice import AutoModel
        self._model = AutoModel(
            model_dir=self.model_dir,
            load_jit=False,
            load_trt=False,
        )

    def _synthesize(self, text: str) -> tuple:
        import torch
        ko_text = f"<|ko|>{text}"
        ref_wav = self.REF_WAV
        ref_text = self.REF_TEXT

        if not os.path.exists(ref_wav):
            raise RuntimeError("Reference WAV not found")

        chunks = []
        ttfa_ms = -1.0
        t0 = time.perf_counter()

        for i, result in enumerate(self._model.inference_zero_shot(
            ko_text, ref_text, ref_wav, stream=False
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
        import gc, torch
        del self._model
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
