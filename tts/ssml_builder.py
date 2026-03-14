"""
tts/ssml_builder.py
---------------------
Builds SSML (Speech Synthesis Markup Language) strings from VoiceParams.

SSML output is compatible with:
  - Google Cloud Text-to-Speech
  - Amazon Polly
  - Microsoft Azure TTS
  - Any W3C SSML-compliant engine

Key tags used:
  <speak>             root element
  <prosody>           controls rate, pitch, volume
  <emphasis>          stresses words/phrases
  <break>             injects silence (for sadness pause, dramatic effect)
  <amazon:auto-breaths> (Polly only, skipped for others)
"""

from tts.voice_mapper import VoiceParams


class SSMLBuilder:
    """
    Converts raw text + VoiceParams → SSML string.

    Example output for joy at intensity=0.9:
        <speak>
          <prosody rate="130%" pitch="+3st" volume="+4dB">
            <emphasis level="strong">
              This is absolutely wonderful news!
            </emphasis>
          </prosody>
        </speak>
    """

    def build(self, text: str, params: VoiceParams) -> str:
        text = self._escape_xml(text)

        rate_str   = f"{params.rate_percent:.0f}%"
        pitch_str  = f"{params.pitch_st:+.0f}st"
        volume_str = f"{params.volume_db:+.0f}dB"

        # Build inner content
        content = text

        # Wrap in <emphasis> if needed
        if params.emphasis in ("moderate", "strong"):
            content = f'<emphasis level="{params.emphasis}">{content}</emphasis>'

        # Wrap in <prosody>
        prosody_attrs = []
        if params.rate_percent != 100:
            prosody_attrs.append(f'rate="{rate_str}"')
        if params.pitch_st != 0:
            prosody_attrs.append(f'pitch="{pitch_str}"')
        if params.volume_db != 0:
            prosody_attrs.append(f'volume="{volume_str}"')

        if prosody_attrs:
            attrs = " ".join(prosody_attrs)
            content = f'<prosody {attrs}>{content}</prosody>'

        # Add leading pause (sadness, surprise)
        if params.pause_before_ms > 0:
            pause_tag = f'<break time="{params.pause_before_ms}ms"/>'
            content = pause_tag + content

        return f"<speak>{content}</speak>"

    def build_plain(self, text: str) -> str:
        """Neutral SSML — no modulation, just the wrapper."""
        return f"<speak>{self._escape_xml(text)}</speak>"

    @staticmethod
    def _escape_xml(text: str) -> str:
        """Escape characters that would break SSML XML."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )


# ── quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tts.voice_mapper import VoiceMapper

    mapper  = VoiceMapper()
    builder = SSMLBuilder()

    samples = [
        ("I am so excited about this incredible opportunity!", "joy",     0.95),
        ("I can't believe this happened again. I'm furious.",  "anger",   0.85),
        ("I miss you so much. Everything feels empty.",        "sadness", 0.80),
        ("Please hold while I check your account.",            "neutral", 0.0),
        ("Wait — that's actually real? That's unbelievable!",  "surprise",0.88),
    ]

    for text, emotion, intensity in samples:
        params = mapper.map(emotion, intensity)
        ssml   = builder.build(text, params)
        print(f"\n[{emotion} @ {intensity}]")
        print(ssml)
