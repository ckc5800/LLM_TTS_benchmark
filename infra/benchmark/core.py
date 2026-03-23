"""
TTS Benchmark Core - 공통 데이터 구조 및 유틸리티
"""
from __future__ import annotations
import time
import json
import csv
import os
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from datetime import datetime


# ─── 테스트 문장 ────────────────────────────────────────────────────────────────
TEST_TEXTS = {
    "ko_short": "안녕하세요. 오늘 날씨가 정말 좋네요.",
    "ko_medium": (
        "안녕하세요. 인공지능 음성 합성 기술이 놀라울 정도로 발전했습니다. "
        "한국어 발음의 정확성과 자연스러움을 평가하고 있습니다."
    ),
    "ko_long": (
        "안녕하세요. 오늘은 여러 가지 오픈소스 TTS 모델들을 벤치마크 테스트합니다. "
        "인공지능 음성 합성 기술은 최근 몇 년 사이에 놀라울 정도로 발전했으며, "
        "자연스러운 한국어 발음과 억양을 재현하는 능력이 크게 향상되었습니다. "
        "각 모델의 추론 속도와 음성 품질을 비교하겠습니다."
    ),
    "zh_medium": (
        "人工智能语音合成技术取得了令人瞩目的进展。"
        "自然语言处理和深度学习的结合使得语音合成更加自然流畅。"
    ),
    "en_medium": (
        "Artificial intelligence speech synthesis technology has made remarkable progress. "
        "The combination of natural language processing and deep learning "
        "makes speech synthesis more natural and fluent."
    ),
    "ko_numbers": (
        "2026년 3월 2일 오후 3시 45분입니다. "
        "서울 기온은 영하 2.5도이며, 내일 강수 확률은 80퍼센트입니다. "
        "문의 전화번호는 공일공 일이삼사 오육칠팔입니다."
    ),
    "ko_mixed": (
        "AI와 머신러닝 기술을 활용한 TTS 시스템이 발전하고 있습니다. "
        "GPT와 Claude 같은 대형 언어 모델들도 음성 기능을 지원합니다. "
        "RTF 0.5 이하면 실시간 서비스에 적합합니다."
    ),
    "en_numbers": (
        "The meeting is scheduled for March 2nd, 2026 at 3:45 PM. "
        "The temperature is minus 2.5 degrees Celsius, and there is an 80 percent chance of rain. "
        "Please call us at 1-800-555-0123 for more information."
    ),
    # ── 신규 KO (5개) ────────────────────────────────────────────────────────
    "ko_formal": (
        "존경하는 여러분, 오늘 이 자리에 함께해 주셔서 감사드립니다. "
        "저희 회사는 지난 10년간 꾸준한 성장을 거듭하며 "
        "글로벌 시장에서 경쟁력을 갖춘 기업으로 발전해 왔습니다. "
        "앞으로도 최선을 다해 여러분의 기대에 부응하겠습니다."
    ),
    "ko_conversational": (
        "야, 오늘 점심 뭐 먹을래? 나는 김치찌개가 당기는데. "
        "아니면 짜장면도 괜찮고. 요즘 날씨도 추운데 따뜻한 국물이 최고 아니겠어? "
        "네 생각은 어때?"
    ),
    "ko_technical": (
        "딥러닝 모델의 학습 과정에서 배치 정규화와 드롭아웃을 적용하면 "
        "과적합을 효과적으로 방지할 수 있습니다. "
        "트랜스포머 아키텍처의 어텐션 메커니즘은 "
        "시퀀스 데이터를 병렬로 처리하여 훈련 효율을 높입니다."
    ),
    "ko_question": (
        "정말요? 믿을 수가 없네요! "
        "그게 사실이라면 정말 대단한 일이 아닐 수 없어요. "
        "어떻게 그런 생각을 하게 되셨나요? "
        "혹시 더 자세히 설명해 주실 수 있을까요?"
    ),
    "ko_proverb": (
        "가는 말이 고와야 오는 말이 곱다고 했습니다. "
        "천 리 길도 한 걸음부터 시작되니, 포기하지 말고 꾸준히 노력해 보세요. "
        "세 살 버릇 여든까지 간다는 말처럼, 좋은 습관을 지금 시작하는 것이 중요합니다."
    ),
    # ── 신규 EN (8개) ────────────────────────────────────────────────────────
    "en_short": (
        "Welcome to our service. How can I help you today?"
    ),
    "en_long": (
        "Artificial intelligence is transforming industries across the globe at an unprecedented pace. "
        "From healthcare diagnostics to autonomous vehicles, machine learning algorithms "
        "are solving problems that were once thought impossible. "
        "Researchers continue to push the boundaries of what these systems can achieve, "
        "developing models that can understand, generate, and reason about complex information."
    ),
    "en_technical": (
        "The transformer architecture uses multi-head self-attention mechanisms "
        "to process sequential data in parallel, achieving state-of-the-art performance. "
        "Gradient checkpointing reduces GPU memory usage during backpropagation "
        "by recomputing activations on the backward pass."
    ),
    "en_conversational": (
        "Hey, what's up? Are you free this weekend? "
        "I was thinking we could grab some coffee and catch up. "
        "It's been ages since we last hung out, and I've got so much to tell you!"
    ),
    "en_formal": (
        "Ladies and gentlemen, we are pleased to announce the launch of our newest product line. "
        "Following extensive research and development spanning three years, "
        "this innovation represents a significant advancement in the field. "
        "We invite you to join us in celebrating this milestone achievement."
    ),
    "en_punctuation": (
        "Wait, are you serious? I can't believe it! "
        "First, you forgot the meeting; then, you missed the deadline... "
        "and now this? Well, I suppose we'll have to start over again."
    ),
    "en_names": (
        "Apple CEO Tim Cook met with Microsoft founder Bill Gates in New York City last Tuesday. "
        "Their discussion focused on artificial intelligence developments at both companies, "
        "with particular emphasis on OpenAI's recent breakthroughs and their impact on Silicon Valley."
    ),
    "en_emotional": (
        "I'm so incredibly excited to share this news with you! "
        "After years of hard work, countless setbacks, and unwavering dedication, "
        "we finally achieved what seemed impossible. "
        "This moment truly means everything to me, and I'm so grateful for your support."
    ),
    # ── 일본語 (5개) ─────────────────────────────────────────────────────────
    "ja_short": "こんにちは。今日はとても良い天気ですね。",
    "ja_medium": (
        "人工知能音声合成技術は目覚ましい進歩を遂げています。"
        "自然言語処理と深層学習の組み合わせにより、"
        "より自然で流暢な音声合成が実現されています。"
    ),
    "ja_formal": (
        "皆様、本日はお集まりいただきありがとうございます。"
        "弊社はこの十年間、継続的な成長を遂げてまいりました。"
        "今後ともご支援ご協力のほど、よろしくお願い申し上げます。"
    ),
    "ja_conversational": (
        "ねえ、今日の昼ご飯どうする？ラーメンかカレーか迷ってるんだけど。"
        "最近寒いし、温かいものが食べたいよね。あなたはどう思う？"
    ),
    "ja_technical": (
        "トランスフォーマーアーキテクチャは、マルチヘッド自己注意機構を用いて"
        "シーケンスデータを並列処理します。"
        "バッチ正規化とドロップアウトを組み合わせることで、"
        "モデルの汎化性能を大幅に向上させることができます。"
    ),
    # ── 中文 (5개) ───────────────────────────────────────────────────────────
    "zh_short": "你好，今天天气真不错。",
    "zh_medium": (
        "人工智能语音合成技术取得了令人瞩目的进展。"
        "自然语言处理和深度学习的结合使得语音合成更加自然流畅。"
    ),
    "zh_formal": (
        "尊敬的各位来宾，感谢您出席本次活动。"
        "我公司在过去十年中取得了长足的发展，"
        "在全球市场上建立了强大的竞争优势。"
        "未来，我们将继续努力，不负各方期望。"
    ),
    "zh_conversational": (
        "哎，你今天中午吃什么？我想吃火锅，最近天气冷，"
        "吃点热乎的最舒服了。你有什么想法吗？"
    ),
    "zh_technical": (
        "深度学习模型在训练过程中，通过批量归一化和随机丢弃来防止过拟合。"
        "Transformer架构的注意力机制能够并行处理序列数据，"
        "显著提高了训练效率和模型性能。"
    ),
    # ── Otros (Spanish, French, German, Dutch, Italian, Portuguese, Polish) ──
    "es_short": "Hola. ¿Cómo estás hoy? El clima es muy agradable.",
    "fr_short": "Bonjour. Comment allez-vous aujourd'hui? Il fait beau.",
    "de_short": "Hallo. Wie geht es dir heute? Das Wetter ist sehr schön.",
    "nl_short": "Hallo. Hoe gaat het met je vandaag? Het weer is erg mooi.",
    "it_short": "Ciao. Come stai oggi? Il tempo è molto bello.",
    "pt_short": "Olá. Como você está hoje? O tempo está muito bonito.",
    "pl_short": "Cześć. Jak się dzisiaj masz? Pogoda jest bardzo ładna.",
}
DEFAULT_TEST_TEXT = TEST_TEXTS["ko_medium"]

# ─── 텍스트 스위트 (KO 10 + EN 10 + JA 5 + ZH 5 + others) ────────────────────
TEXT_SUITES: dict[str, list[str]] = {
    "ko": [
        "ko_short", "ko_medium", "ko_long", "ko_numbers", "ko_mixed",
        "ko_formal", "ko_conversational", "ko_technical", "ko_question", "ko_proverb",
    ],
    "en": [
        "en_short", "en_medium", "en_long", "en_numbers", "en_technical",
        "en_conversational", "en_formal", "en_punctuation", "en_names", "en_emotional",
    ],
    "ja": [
        "ja_short", "ja_medium", "ja_formal", "ja_conversational", "ja_technical",
    ],
    "zh": [
        "zh_short", "zh_medium", "zh_formal", "zh_conversational", "zh_technical",
    ],
    "es": ["es_short"],
    "fr": ["fr_short"],
    "de": ["de_short"],
    "nl": ["nl_short"],
    "it": ["it_short"],
    "pt": ["pt_short"],
    "pl": ["pl_short"],
}


# ─── 결과 데이터 구조 ─────────────────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    # 모델 정보
    model_name: str
    model_version: str = ""
    model_size_params: str = ""          # e.g. "500M", "1.7B"
    korean_support: bool = True

    # 테스트 설정
    test_text: str = DEFAULT_TEST_TEXT
    test_type: str = "TTS"               # "TTS" | "VoiceClone"
    run_index: int = 0                   # 반복 실행 인덱스

    # 타이밍
    load_time_s: float = -1.0            # 모델 로드 시간
    ttfa_ms: float = -1.0               # Time To First Audio (ms)
    inference_time_s: float = -1.0      # 전체 추론 시간
    audio_duration_s: float = -1.0      # 생성된 오디오 길이
    rtf: float = -1.0                   # Real-Time Factor

    # 리소스
    vram_before_mb: float = -1.0
    vram_after_mb: float = -1.0
    vram_used_mb: float = -1.0          # after - before
    vram_peak_mb: float = -1.0          # 추론 중 최대 VRAM (max_memory_allocated)

    # 출력
    sample_rate: int = -1
    output_wav: str = ""

    # 상태
    success: bool = False
    error: str = ""
    notes: str = ""

    # 타임스탬프
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def compute_derived(self):
        """rtf, vram_used 등 파생 지표 계산"""
        if self.audio_duration_s > 0 and self.inference_time_s > 0:
            self.rtf = self.inference_time_s / self.audio_duration_s
        if self.vram_before_mb >= 0 and self.vram_after_mb >= 0:
            self.vram_used_mb = self.vram_after_mb - self.vram_before_mb

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        if not self.success:
            return f"  [{self.model_name}] FAILED: {self.error}"
        return (
            f"  [{self.model_name}] "
            f"load={self.load_time_s:.1f}s | "
            f"TTFA={self.ttfa_ms:.0f}ms | "
            f"total={self.inference_time_s:.2f}s | "
            f"audio={self.audio_duration_s:.2f}s | "
            f"RTF={self.rtf:.3f} | "
            f"VRAM={self.vram_used_mb:.0f}MB"
        )


# ─── VRAM 측정 유틸 ──────────────────────────────────────────────────────────
def get_vram_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return -1.0


def get_vram_peak_mb() -> float:
    """추론 중 최대 VRAM 사용량 (reset_vram_peak() 이후의 max)"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return -1.0


def reset_vram_peak():
    """peak VRAM 카운터 리셋 (추론 직전에 호출)"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def clear_vram():
    try:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ─── 오디오 길이 측정 ──────────────────────────────────────────────────────────
def get_audio_duration(wav_path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        return info.duration
    except Exception:
        return -1.0


def tensor_duration(tensor, sample_rate: int) -> float:
    """torch tensor에서 오디오 길이 계산"""
    try:
        if tensor.ndim == 1:
            return tensor.shape[0] / sample_rate
        return tensor.shape[-1] / sample_rate
    except Exception:
        return -1.0


# ─── 결과 저장 ────────────────────────────────────────────────────────────────
class BenchmarkLogger:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add(self, result: BenchmarkResult):
        result.compute_derived()
        self.results.append(result)
        print(result.summary_line())
        self._auto_save()

    def _auto_save(self):
        """실행할 때마다 자동 저장 (crash 대비)"""
        self.save_json()
        self.save_csv()

    def save_json(self):
        path = os.path.join(self.results_dir, f"{self.session_id}_results.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in self.results], f, ensure_ascii=False, indent=2)
        return path

    def save_csv(self):
        if not self.results:
            return
        path = os.path.join(self.results_dir, f"{self.session_id}_results.csv")
        fields = list(asdict(self.results[0]).keys())
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows([r.to_dict() for r in self.results])
        return path

    def save_markdown(self):
        path = os.path.join(self.results_dir, f"{self.session_id}_report.md")
        lines = [
            f"# TTS Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Test Configuration",
            f"- Text: `{DEFAULT_TEST_TEXT}`",
            f"- GPU: {_get_gpu_name()}",
            f"",
            f"## Results",
            f"",
            f"| Model | Size | Korean | TTFA(ms) | Inference(s) | Audio(s) | RTF | VRAM(MB) | Load(s) | Success |",
            f"|-------|------|--------|----------|-------------|---------|-----|----------|---------|---------|",
        ]
        for r in self.results:
            if r.run_index > 0:
                continue  # 첫 번째 실행만 표에 포함 (반복 실행 평균은 별도)
            ko = "✔" if r.korean_support else "✘"
            status = "✔" if r.success else f"✘ {r.error[:30]}"
            ttfa = f"{r.ttfa_ms:.0f}" if r.ttfa_ms >= 0 else "-"
            inf = f"{r.inference_time_s:.2f}" if r.inference_time_s >= 0 else "-"
            dur = f"{r.audio_duration_s:.2f}" if r.audio_duration_s >= 0 else "-"
            rtf = f"{r.rtf:.3f}" if r.rtf >= 0 else "-"
            vram = f"{r.vram_used_mb:.0f}" if r.vram_used_mb >= 0 else "-"
            load = f"{r.load_time_s:.1f}" if r.load_time_s >= 0 else "-"
            lines.append(
                f"| {r.model_name} | {r.model_size_params} | {ko} | {ttfa} | {inf} | {dur} | {rtf} | {vram} | {load} | {status} |"
            )

        lines += ["", "## Notes", ""]
        for r in self.results:
            if r.notes:
                lines.append(f"- **{r.model_name}**: {r.notes}")

        with open(path, 'w', encoding='utf-8-sig') as f:
            f.write('\n'.join(lines))
        return path

    def finalize(self):
        json_path = self.save_json()
        csv_path = self.save_csv()
        md_path = self.save_markdown()
        print(f"\n{'='*60}")
        print(f"벤치마크 완료 - {len(self.results)}개 결과")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
        print(f"  MD:   {md_path}")
        print(f"{'='*60}")


def _get_gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "Unknown"
