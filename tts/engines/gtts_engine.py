"""
tts/engines/gtts_engine.py
----------------------------
gTTS for synthesis + librosa for audio modulation.

Punctuation handling:
  Splits text at punctuation boundaries, synthesizes each chunk
  separately, inserts calibrated silence between chunks, then
  concatenates all chunks into a single audio file.

  Pause durations:
    ,   →  180ms  (brief pause)
    ;   →  250ms  (medium pause)
    :   →  250ms  (medium pause)
    .   →  400ms  (sentence end)
    !   →  350ms  (exclamation)
    ?   →  380ms  (question — slight rise implied by pitch)
    ... →  500ms  (ellipsis / trailing off)
    —   →  300ms  (em dash)

This makes "Wait, are you serious?" sound like natural speech
with a genuine pause after "Wait" rather than running together.

Requires: ffmpeg  (sudo apt install ffmpeg)
"""

import os
import sys
import re
import uuid
import numpy as np
import librosa
import soundfile as sf
from gtts import gTTS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tts.voice_mapper import VoiceParams


# ── Pause durations in seconds ────────────────────────────────────────────────
PAUSE_MAP = {
    ",":   0.18,
    ";":   0.25,
    ":":   0.25,
    ".":   0.40,
    "!":   0.35,
    "?":   0.38,
    "...": 0.50,
    "—":   0.30,
    "-":   0.15,
}

# Regex to split text at punctuation while keeping the punctuation mark
SPLIT_PATTERN = re.compile(
    r'(\.{3}|[,;:!?\.—])'
)


def _make_silence(duration_seconds: float, sr: int) -> np.ndarray:
    """Create a numpy array of silence at the given sample rate."""
    samples = int(sr * duration_seconds)
    return np.zeros(samples, dtype=np.float32)


def _split_into_chunks(text: str) -> list[tuple[str, str]]:
    """
    Split text into (chunk_text, following_punctuation) pairs.

    Example:
      "Wait, are you serious? No way!"
      → [("Wait", ","), ("are you serious", "?"), ("No way", "!")]

    Handles:
      - Multiple punctuation marks
      - Ellipsis (...)
      - Em dash (—)
      - Trailing text with no punctuation
    """
    # Normalize ellipsis first
    text = text.replace("...", "…")

    parts    = SPLIT_PATTERN.split(text)
    chunks   = []
    i        = 0

    while i < len(parts):
        chunk = parts[i].strip()
        punct = ""

        if i + 1 < len(parts) and SPLIT_PATTERN.match(parts[i + 1]):
            punct = parts[i + 1].replace("…", "...")
            i    += 2
        else:
            i += 1

        if chunk:
            chunks.append((chunk, punct))
        elif punct and chunks:
            # Punctuation with no text — attach to previous chunk
            prev_text, prev_punct = chunks[-1]
            chunks[-1] = (prev_text, prev_punct + punct)

    return chunks


def _synthesize_chunk(text: str, lang: str, tmp_dir: str) -> tuple[np.ndarray, int]:
    """Synthesize a single text chunk with gTTS and return (audio_array, sample_rate)."""
    tmp_path = os.path.join(tmp_dir, f"_chunk_{uuid.uuid4().hex[:8]}.mp3")
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(tmp_path)
        y, sr = librosa.load(tmp_path, sr=None)
        return y, sr
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class GTTSEngine:

    def __init__(self, output_dir: str = "outputs", lang: str = "en"):
        self.output_dir = output_dir
        self.lang       = lang
        os.makedirs(output_dir, exist_ok=True)

    def synthesize(self, text: str, params: VoiceParams, filename: str = None) -> str:
        if filename is None:
            uid      = uuid.uuid4().hex[:8]
            filename = f"{params.emotion}_{uid}.mp3"

        final_path = os.path.join(self.output_dir, filename)

        # ── Step 1: Split text at punctuation boundaries ──────────────────
        chunks = _split_into_chunks(text)

        if not chunks:
            chunks = [(text, "")]

        # ── Step 2: Synthesize each chunk and build audio with pauses ─────
        # ── Step 2: Synthesize each chunk and build audio with pauses ─────
        segments = []
        sr       = None
        FADE_MS  = 8    # 8ms crossfade at chunk boundaries — eliminates clicks

        for chunk_text, punct in chunks:
            if not chunk_text.strip():
                continue

            y_chunk, chunk_sr = _synthesize_chunk(chunk_text, self.lang, self.output_dir)

            if sr is None:
                sr = chunk_sr

            if chunk_sr != sr:
                y_chunk = librosa.resample(
                    y_chunk, orig_sr=chunk_sr, target_sr=sr,
                    res_type="soxr_hq"
                )

            # Apply short fade-in and fade-out to each chunk
            fade_samples = int(sr * FADE_MS / 1000)
            if len(y_chunk) > fade_samples * 2:
                # Fade in
                y_chunk[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
                # Fade out
                y_chunk[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)

            segments.append(y_chunk)

            pause_duration = 0.0
            for p, duration in PAUSE_MAP.items():
                if p in punct:
                    pause_duration = max(pause_duration, duration)

            if pause_duration > 0:
                # Use a very short fade into silence rather than hard cut
                silence = _make_silence(pause_duration, sr)
                segments.append(silence)
        # ── Step 3: Concatenate all segments ─────────────────────────────
        y = np.concatenate(segments)

        # ── Step 4: Apply rate (time-stretch, minimum perceptible) ───────
        speed_factor = params.rate_percent / 100.0
        if speed_factor > 1.0:
            speed_factor = max(speed_factor, 1.15)
        elif speed_factor < 1.0:
            speed_factor = min(speed_factor, 0.82)
        if abs(speed_factor - 1.0) > 0.01:
            y = librosa.effects.time_stretch(
                y,
                rate=speed_factor,
                n_fft=2048,          # larger window = smoother transitions
                hop_length=512,      # overlap between frames
                window="hann",       # hann window reduces metallic artifacts
            )

        # ── Step 5: Apply pitch shift independently ───────────────────────
        if params.pitch_st != 0:
            y = librosa.effects.pitch_shift(
                y, sr=sr,
                n_steps=params.pitch_st,
                bins_per_octave=24,
                res_type="soxr_hq",   # highest quality resampler
                n_fft=2048,
                hop_length=512,
                window="hann",
            )

        # ── Step 6: Apply volume ──────────────────────────────────────────
        if params.volume_db != 0:
            gain = 10 ** (params.volume_db / 20.0)
            y    = y * gain

        # Always normalize to 90% of max to avoid any clipping artifacts
        max_v = np.max(np.abs(y))
        if max_v > 0.001:
            y = y * (0.90 / max_v) * min(max_v, 1.0)
            
        # ── Step 7: Leading pause for emotion (sadness/empathy/relief) ───
        if params.pause_before_ms > 0:
            pause_samples = int(sr * params.pause_before_ms / 1000)
            y = np.concatenate([np.zeros(pause_samples, dtype=np.float32), y])

        # ── Step 8: Save ──────────────────────────────────────────────────
        wav_path = os.path.join(self.output_dir, f"_tmp_{uuid.uuid4().hex[:6]}.wav")
        sf.write(wav_path, y, sr)

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(wav_path)
            audio.export(final_path, format="mp3")
            os.remove(wav_path)
        except Exception:
            final_path = final_path.replace(".mp3", ".wav")
            os.rename(wav_path, final_path)

        print(f"[gTTS+librosa] {params.emotion} | "
              f"rate={params.rate_percent:.0f}%  "
              f"pitch={params.pitch_hz:.0f}Hz({params.pitch_st:+.2f}st)  "
              f"vol={params.volume_db:+.1f}dB  "
              f"chunks={len(chunks)}  "
              f"pause={params.pause_before_ms}ms")
        print(f"[gTTS+librosa] Saved: {final_path}")
        return final_path


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tts.voice_mapper import VoiceMapper
    mapper = VoiceMapper()
    engine = GTTSEngine(output_dir="outputs")

    cases = [
        ("Wait, are you serious? That sounds too good to be true!",         "surprise",     0.85),
        ("I cannot believe this happened again. Every single time!",        "frustration",  0.90),
        ("Oh wow, that is amazing! I love this deal, sign me up!",          "excitement",   0.92),
        ("I understand. That must have been really difficult for you.",      "empathy",      0.75),
        ("Finally! I am so relieved. Thank you so much for sorting this.",  "relief",       0.88),
        ("I need this fixed immediately. It is absolutely critical.",        "urgency",      0.95),
        ("Your account balance is one hundred and forty-two dollars.",       "neutral",      0.0),
    ]

    for text, emotion, intensity in cases:
        p    = mapper.map(emotion, intensity)
        path = engine.synthesize(text, p)
        print(f"  → {path}\n")
