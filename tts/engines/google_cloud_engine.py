"""
tts/engines/google_cloud_engine.py
------------------------------------
High-quality TTS using Google Cloud Text-to-Speech API with full SSML support.

Setup:
  1. Create a GCP project at https://console.cloud.google.com
  2. Enable "Cloud Text-to-Speech API"
  3. Create a service account → download JSON key
  4. Set env var: GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
  5. pip install google-cloud-texttospeech

Voices used:
  - en-US-Neural2-F  (female, warm/natural)
  - en-US-Neural2-D  (male, authoritative)
  - en-US-Studio-O   (Studio quality, higher cost)

SSML support: full <prosody rate pitch volume> + <emphasis> + <break>
"""

import os
import uuid
from tts.voice_mapper import VoiceParams
from tts.ssml_builder import SSMLBuilder


class GoogleCloudTTSEngine:
    """
    Uses Google Cloud TTS with SSML for precise prosody control.

    Cost: ~$4 per 1 million characters (Neural2 voices).
          Studio voices are ~$16/M chars but sound excellent.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        voice_name: str = "en-US-Neural2-F",
        language_code: str = "en-US",
    ):
        from google.cloud import texttospeech  # lazy import

        self.output_dir = output_dir
        self.voice_name = voice_name
        self.language_code = language_code
        self.client = texttospeech.TextToSpeechClient()
        self.builder = SSMLBuilder()
        os.makedirs(output_dir, exist_ok=True)

        # Import enums once
        self._tts = texttospeech

    def synthesize(self, text: str, params: VoiceParams, filename: str = None) -> str:
        """
        Synthesize with SSML prosody and return the path to the .mp3 file.
        """
        if filename is None:
            uid = uuid.uuid4().hex[:8]
            filename = f"{params.emotion}_{uid}.mp3"

        output_path = os.path.join(self.output_dir, filename)

        # Build SSML
        ssml = self.builder.build(text, params)
        print(f"[GoogleTTS] SSML:\n{ssml}\n")

        # Voice selection
        voice = self._tts.VoiceSelectionParams(
            language_code=self.language_code,
            name=self.voice_name,
        )

        # Synthesis input
        synthesis_input = self._tts.SynthesisInput(ssml=ssml)

        # Audio config
        audio_config = self._tts.AudioConfig(
            audio_encoding=self._tts.AudioEncoding.MP3,
            sample_rate_hertz=24000,
        )

        # API call
        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        with open(output_path, "wb") as f:
            f.write(response.audio_content)

        print(f"[GoogleTTS] Saved: {output_path}")
        return output_path

    def list_voices(self, language_code: str = "en-US"):
        """Print all available voices for a language."""
        response = self.client.list_voices(language_code=language_code)
        for voice in response.voices:
            print(f"  {voice.name}  gender={voice.ssml_gender.name}  "
                  f"rates={voice.natural_sample_rate_hertz}")


# ── quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tts.voice_mapper import VoiceMapper
    mapper = VoiceMapper()
    engine = GoogleCloudTTSEngine(output_dir="outputs")

    cases = [
        ("I am absolutely delighted to hear that! Wonderful news!", "joy",     0.92),
        ("This is unacceptable. I demand to speak to a manager.",   "anger",   0.88),
        ("I understand. I'm so sorry to hear about your loss.",     "sadness", 0.70),
        ("Your ticket number is 4-7-2-1. Please hold.",            "neutral", 0.0),
    ]
    for text, emotion, intensity in cases:
        p = mapper.map(emotion, intensity)
        path = engine.synthesize(text, p)
        print(f"  → {path}\n")
