"""
tts/engines/elevenlabs_engine.py
----------------------------------
Premium TTS using ElevenLabs API.

ElevenLabs has built-in voice cloning and emotional expressiveness.
We control expressiveness via:
  - stability      : 0.0 (very expressive/variable) → 1.0 (consistent/flat)
  - similarity_boost: voice similarity to original (0.0–1.0)
  - style           : style exaggeration 0.0–1.0 (newer models only)
  - speed           : 0.7–1.2 (maps to our rate_percent)

We DON'T use SSML here — ElevenLabs has its own prosody model.
Instead we map emotion intensity directly to its native parameters.

Setup:
  pip install elevenlabs
  Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env

Free tier: 10,000 chars/month
"""

import os
import uuid
from tts.voice_mapper import VoiceParams


# ElevenLabs emotional parameter profiles
# stability: lower = more expressive/variable delivery
# style:     higher = more exaggerated style (v2 models only)
_EL_PROFILES = {
    #             stability  similarity  style  speed_range
    "joy":      (0.25,       0.80,       0.70,  (1.05, 1.25)),
    "surprise": (0.20,       0.75,       0.75,  (1.00, 1.20)),
    "anger":    (0.15,       0.85,       0.80,  (1.05, 1.30)),
    "disgust":  (0.30,       0.80,       0.50,  (0.90, 1.05)),
    "fear":     (0.25,       0.80,       0.60,  (1.00, 1.15)),
    "sadness":  (0.55,       0.80,       0.40,  (0.75, 0.90)),
    "neutral":  (0.70,       0.80,       0.10,  (0.95, 1.00)),
}


class ElevenLabsEngine:
    """
    Maps VoiceParams → ElevenLabs voice settings for expressive synthesis.
    """

    def __init__(self, output_dir: str = "outputs"):
        from elevenlabs.client import ElevenLabs  # lazy import

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        api_key  = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise EnvironmentError("ELEVENLABS_API_KEY not set in environment.")

        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.client   = ElevenLabs(api_key=api_key)

    def synthesize(self, text: str, params: VoiceParams, filename: str = None) -> str:
        """
        Synthesize with ElevenLabs and return path to .mp3 file.
        """
        if filename is None:
            uid = uuid.uuid4().hex[:8]
            filename = f"{params.emotion}_{uid}.mp3"

        output_path = os.path.join(self.output_dir, filename)

        profile = _EL_PROFILES.get(params.emotion, _EL_PROFILES["neutral"])
        stability, similarity, style, speed_range = profile

        # Scale stability inversely with intensity (more intense = less stable = more expressive)
        effective_stability  = stability  + (1 - stability)  * (1 - params.intensity) * 0.5
        effective_style      = style      * params.intensity
        speed_min, speed_max = speed_range
        effective_speed      = speed_min  + (speed_max - speed_min) * params.intensity

        from elevenlabs import VoiceSettings

        print(f"[ElevenLabs] emotion={params.emotion} intensity={params.intensity:.2f} | "
              f"stability={effective_stability:.2f}  style={effective_style:.2f}  speed={effective_speed:.2f}")

        audio_generator = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=round(effective_stability, 3),
                similarity_boost=similarity,
                style=round(effective_style, 3),
                use_speaker_boost=True,
            ),
        )

        # Stream chunks to file
        with open(output_path, "wb") as f:
            for chunk in audio_generator:
                if chunk:
                    f.write(chunk)

        print(f"[ElevenLabs] Saved: {output_path}")
        return output_path


# ── quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tts.voice_mapper import VoiceMapper
    mapper = VoiceMapper()
    engine = ElevenLabsEngine(output_dir="outputs")

    cases = [
        ("This is absolutely the best news I've heard all year!", "joy",     0.95),
        ("I've told you three times and nothing has been fixed.", "anger",   0.80),
        ("I understand. Take all the time you need.",             "sadness", 0.60),
    ]
    for text, emotion, intensity in cases:
        p = mapper.map(emotion, intensity)
        path = engine.synthesize(text, p)
        print(f"  → {path}\n")
