"""
tts/engines/pyttsx3_engine.py
-------------------------------
Offline TTS using pyttsx3. No API key, no internet required.

Limitations:
  - No SSML support (parameters set directly on the engine)
  - Voice quality is robotic compared to cloud engines
  - Pitch control is limited on some platforms (macOS works best)

Best for: rapid local prototyping, offline demos.
"""

import os
import uuid
import pyttsx3
from tts.voice_mapper import VoiceParams


class Pyttsx3Engine:
    """
    Synthesizes speech with pyttsx3, applying rate/volume from VoiceParams.

    Note: pyttsx3 rate is words-per-minute. We map our percent-of-baseline
    to WPM using a 175 WPM default speaking rate.
    """

    DEFAULT_WPM = 175  # typical conversational rate

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._engine = pyttsx3.init()

    def synthesize(self, text: str, params: VoiceParams, filename: str = None) -> str:
        """
        Synthesize text with the given vocal parameters.

        Returns the path to the generated .wav file.
        """
        if filename is None:
            filename = f"{params.emotion}_{uuid.uuid4().hex[:8]}.wav"

        output_path = os.path.join(self.output_dir, filename)

        # Rate: convert percent → WPM
        wpm = int(self.DEFAULT_WPM * (params.rate_percent / 100))
        wpm = max(80, min(300, wpm))

        # Volume: pyttsx3 uses 0.0–1.0; convert dB change to linear multiplier
        # 0 dB = 1.0, +6 dB ≈ 2.0, -6 dB ≈ 0.5
        current_vol = self._engine.getProperty("volume") or 1.0
        db = params.volume_db
        vol_multiplier = 10 ** (db / 20)
        new_vol = max(0.1, min(1.0, current_vol * vol_multiplier))

        self._engine.setProperty("rate",   wpm)
        self._engine.setProperty("volume", new_vol)

        # pyttsx3 does not support pitch — log a note
        if params.pitch_st != 0:
            print(f"[pyttsx3] Note: pitch={params.pitch_st:+.1f}st requested but pyttsx3 has no pitch control.")

        self._engine.save_to_file(text, output_path)
        self._engine.runAndWait()

        print(f"[pyttsx3] Saved: {output_path}  (rate={wpm}wpm  vol={new_vol:.2f})")
        return output_path

    def list_voices(self):
        """Print available voices on this system."""
        for v in self._engine.getProperty("voices"):
            print(f"  id={v.id}  name={v.name}")


# ── quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tts.voice_mapper import VoiceMapper
    mapper = VoiceMapper()
    engine = Pyttsx3Engine(output_dir="outputs")

    cases = [
        ("This is absolutely wonderful! I couldn't be happier!", "joy",     0.9),
        ("I am completely fed up with this service.",            "anger",   0.85),
        ("I just feel so lost and alone right now.",             "sadness", 0.75),
        ("Your account balance is one hundred dollars.",         "neutral", 0.0),
    ]
    for text, emotion, intensity in cases:
        p = mapper.map(emotion, intensity)
        path = engine.synthesize(text, p)
        print(f"  → {path}")
