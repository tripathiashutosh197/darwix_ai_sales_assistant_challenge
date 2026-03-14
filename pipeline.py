"""
pipeline.py
------------
The Empathy Engine core pipeline.

Wires together:
  1. Emotion detector  (Transformer — multi-label 15-dim EmotionVector)
  2. Voice mapper      (hybrid: pitch analytical→Hz, rate analytical+30%, volume neural)
  3. TTS engine        (gTTS + librosa)
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PipelineResult:
    text:           str
    emotion:        str         # dominant emotion label
    secondary:      str | None  # second highest emotion
    emotion_vector: dict        # full 15-dim scores
    intensity:      float       # overall arousal 0.0-1.0
    rate_percent:   float
    pitch_st:       float
    pitch_hz:       float
    volume_db:      float
    emphasis:       str
    audio_path:     str
    engine_used:    str
    latency_ms:     int


class EmpathyPipeline:
    """
    End-to-end pipeline: text → emotion vector → voice params → audio file.
    """

    def __init__(
        self,
        emotion_model: str = None,
        tts_engine:    str = None,
        output_dir:    str = "outputs",
    ):
        from emotion import get_detector
        from tts import get_engine
        from tts.voice_mapper import VoiceMapper

        print("[Pipeline] Initializing emotion detector...")
        self.detector = get_detector(emotion_model)

        print("[Pipeline] Initializing TTS engine...")
        self.tts = get_engine(tts_engine, output_dir=output_dir)

        self.mapper     = VoiceMapper()
        self.output_dir = output_dir
        self._tts_name  = tts_engine or os.getenv("TTS_ENGINE", "gtts")

        os.makedirs(output_dir, exist_ok=True)
        print("[Pipeline] Ready.\n")

    def run(self, text: str, filename: str = None) -> PipelineResult:
        """Full pipeline: text → audio file."""
        t0 = time.time()

        # ── Step 1: Detect emotion vector ─────────────────────────────────
        emotion_result = self.detector.detect(text)

        # Print full vector (top emotions above threshold)
        top3    = sorted(emotion_result.scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = "  ".join(f"{e}={s:.2f}" for e, s in top3)
        print(f"[Pipeline] Emotions:  {top_str}")
        print(f"[Pipeline] Dominant:  {emotion_result.dominant}  "
              f"intensity={emotion_result.intensity:.2f}  "
              f"compound={emotion_result.compound:+.3f}")

        # ── Step 2: Map to voice parameters ───────────────────────────────
        voice_params = self.mapper.map_vector(emotion_result)

        print(f"[Pipeline] Voice:   "
              f"rate={voice_params.rate_percent:.0f}%  "
              f"pitch={voice_params.pitch_hz:.0f}Hz({voice_params.pitch_st:+.2f}st)  "
              f"vol={voice_params.volume_db:+.1f}dB  "
              f"emphasis={voice_params.emphasis}")

        # ── Step 3: Synthesize audio ───────────────────────────────────────
        audio_path = self.tts.synthesize(text, voice_params, filename=filename)

        latency_ms = int((time.time() - t0) * 1000)
        print(f"[Pipeline] Done in {latency_ms}ms → {audio_path}\n")

        return PipelineResult(
            text           = text,
            emotion        = emotion_result.dominant,
            secondary      = emotion_result.secondary,
            emotion_vector = emotion_result.scores,
            intensity      = emotion_result.intensity,
            rate_percent   = voice_params.rate_percent,
            pitch_st       = voice_params.pitch_st,
            pitch_hz       = voice_params.pitch_hz,
            volume_db      = voice_params.volume_db,
            emphasis       = voice_params.emphasis,
            audio_path     = audio_path,
            engine_used    = self._tts_name,
            latency_ms     = latency_ms,
        )

    def batch(self, texts: list[str]) -> list[PipelineResult]:
        """Run the pipeline on a list of texts."""
        return [self.run(t) for t in texts]
