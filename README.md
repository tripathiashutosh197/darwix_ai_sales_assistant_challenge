# ⚡ Empathy Engine

> Emotionally-aware Text-to-Speech that dynamically modulates vocal parameters based on detected emotion in the source text.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Emotion → Voice Mapping Logic](#emotion--voice-mapping-logic)
- [TTS Engine Comparison](#tts-engine-comparison)
- [Fine-Tuning the Emotion Model](#fine-tuning-the-emotion-model)
- [API Reference](#api-reference)
- [Design Decisions](#design-decisions)

---

## Overview

Standard TTS systems produce flat, monotonic speech. The Empathy Engine adds an emotional layer:

1. **Detects** the emotion in input text (joy, anger, sadness, fear, surprise, disgust, neutral)
2. **Measures** its intensity on a 0–1 scale
3. **Maps** both to vocal parameters (rate, pitch, volume, emphasis)
4. **Synthesizes** expressive audio via your chosen TTS engine

**Example:**
```
Input:  "I can't believe how amazing this turned out!"
Emotion: joy @ intensity 0.92
Voice:   rate=126%  pitch=+3.2st  volume=+3.7dB  emphasis=strong
Output:  joy_a3f8b2c1.mp3
```

---

## Architecture

```
Text Input
    │
    ▼
Emotion Detector ──────────────────► EmotionResult
(VADER or Transformer)               label + intensity + scores
    │
    ▼
Voice Mapper ──────────────────────► VoiceParams
(intensity-scaled rules)             rate / pitch / volume / emphasis
    │
    ▼
SSML Builder ──────────────────────► SSML string
(<prosody> + <emphasis> + <break>)   (for cloud engines)
    │
    ▼
TTS Engine ────────────────────────► audio file
(pyttsx3 / gTTS / Google / ElevenLabs)
```

---

## Project Structure

```
empathy_engine/
│
├── cli.py                          # Command-line interface
├── pipeline.py                     # Core pipeline (wires all blocks)
├── requirements.txt
├── .env.example                    # Copy to .env and configure
│
├── emotion/
│   ├── __init__.py                 # Factory: get_detector()
│   ├── vader_detector.py           # Fast offline detector (VADER)
│   └── transformer_detector.py     # Accurate detector (HuggingFace)
│                                   # Also contains fine_tune() function
│
├── tts/
│   ├── __init__.py                 # Factory: get_engine()
│   ├── voice_mapper.py             # Emotion → VoiceParams
│   ├── ssml_builder.py             # VoiceParams → SSML string
│   └── engines/
│       ├── pyttsx3_engine.py       # Offline (no API key)
│       ├── gtts_engine.py          # Google free TTS + pydub
│       ├── google_cloud_engine.py  # Google Cloud TTS (SSML)
│       └── elevenlabs_engine.py    # ElevenLabs (best quality)
│
├── web/
│   ├── app.py                      # FastAPI server
│   └── templates/
│       └── index.html              # Web UI with audio player
│
└── utils/
    ├── prepare_training_data.py    # Dataset prep for fine-tuning
    └── evaluate.py                 # Accuracy / F1 evaluation
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/yourname/empathy-engine.git
cd empathy-engine
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On Ubuntu/Debian, pyttsx3 needs espeak:
> ```bash
> sudo apt install espeak ffmpeg
> ```
> On macOS: `brew install espeak ffmpeg`

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Emotion model: "vader" (fast, offline) or "transformer" (accurate)
EMOTION_MODEL=transformer

# TTS engine: "pyttsx3" | "gtts" | "google_cloud" | "elevenlabs"
TTS_ENGINE=gtts

# Only needed for google_cloud:
# GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json

# Only needed for elevenlabs:
# ELEVENLABS_API_KEY=your_key_here
# ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

### 4. (First run only) Download the transformer model

The HuggingFace model (~82MB) downloads automatically on first use and is cached at `~/.cache/huggingface/`. No manual step needed.

---

## Running the App

### CLI — single input

```bash
python cli.py "I can't believe how amazing this is!"
```

### CLI — demo mode (runs 7 preset emotional sentences)

```bash
python cli.py --demo
```

### CLI — batch from file

```bash
python cli.py --file sentences.txt --tts gtts --emotion transformer
```

### Web interface

```bash
uvicorn web.app:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

### Run individual modules directly (for testing)

```bash
# Test VADER detector
python emotion/vader_detector.py

# Test Transformer detector
python emotion/transformer_detector.py

# Test voice mapper
python tts/voice_mapper.py

# Test SSML builder
python tts/ssml_builder.py

# Test gTTS engine
python tts/engines/gtts_engine.py
```

---

## Emotion → Voice Mapping Logic

### Parameter table (at intensity = 1.0)

| Emotion  | Rate    | Pitch   | Volume | Emphasis |
|----------|---------|---------|--------|----------|
| joy      | +30%    | +3.5 st | +4 dB  | strong   |
| surprise | +20%    | +4.0 st | +3 dB  | strong   |
| anger    | +25%    | +2.0 st | +6 dB  | strong   |
| disgust  | −10%    | −1.5 st | +2 dB  | moderate |
| fear     | +15%    | +2.5 st | −2 dB  | moderate |
| sadness  | −25%    | −3.5 st | −3 dB  | none     |
| neutral  | 0%      | 0 st    | 0 dB   | none     |

### Intensity scaling

Every parameter is multiplied by the detected intensity score (0–1):

```
rate_percent  = 100 + rate_delta  × intensity
pitch_st      = base_pitch_st     × intensity
volume_db     = base_volume_db    × intensity
```

This means "This is good" (joy @ 0.45) produces subtle modulation,
while "THIS IS THE BEST DAY EVER!" (joy @ 0.97) produces dramatic modulation.

### Emphasis downgrade

The `<emphasis>` level is automatically reduced for low-intensity detections:
- intensity < 0.35 → `none`
- intensity < 0.65 → `moderate` (even if template says `strong`)
- intensity ≥ 0.65 → use template value

### Design rationale

- **High arousal emotions** (anger, joy, surprise) get faster speech because research shows
  humans naturally speak faster when emotionally activated.
- **Low arousal emotions** (sadness) get slower speech + leading pause to simulate the
  hesitation and effort of speaking while distressed.
- **Fear** is fast but quieter — mimicking hushed, anxious speech.
- **Disgust** is slightly slower and lower — the "ugh" vocal quality.

---

## TTS Engine Comparison

| Engine         | Quality | SSML | Cost          | Offline | Setup difficulty |
|----------------|---------|------|---------------|---------|-----------------|
| pyttsx3        | ★★☆☆☆  | ✗    | Free          | ✓       | None            |
| gTTS           | ★★★☆☆  | ✗    | Free          | ✗       | None            |
| Google Cloud   | ★★★★☆  | ✓    | $4/1M chars   | ✗       | Service account |
| ElevenLabs     | ★★★★★  | ✗    | $5/mo (30K)   | ✗       | API key         |

**Recommendation:**
- Start with `gtts` (zero setup, good quality)
- Switch to `google_cloud` for SSML-driven prosody (best accuracy)
- Use `elevenlabs` for the most human-sounding output

---

## Fine-Tuning the Emotion Model

You only need this if you want to train on your own domain-specific data (e.g. customer service transcripts). The pretrained model works well out of the box.

### Step 1: Prepare dataset

Download GoEmotions from https://github.com/google-research/google-research/tree/master/goemotions/data

```bash
python utils/prepare_training_data.py \
    --goemotions_dir path/to/goemotions/data \
    --output data/
```

Or create `data/train.csv` manually (one row per sample):
```
"I love this product!",joy
"This is terrible.",anger
"Please hold.",neutral
```

### Step 2: Fine-tune

```python
from emotion.transformer_detector import fine_tune

fine_tune(
    train_csv  = "data/train.csv",
    val_csv    = "data/val.csv",
    output_dir = "models/emotion_finetuned",
    epochs     = 4,
    batch_size = 16,
)
```

### Step 3: Use your fine-tuned model

In `.env`:
```env
EMOTION_MODEL=transformer
```

In `emotion/transformer_detector.py`, change:
```python
MODEL_ID = "models/emotion_finetuned"
```

### Step 4: Evaluate

```bash
python utils/evaluate.py --csv data/val.csv --model transformer
```

---

## API Reference

### POST /synthesize

**Request:**
```json
{ "text": "I can't believe how amazing this is!" }
```

**Response:**
```json
{
  "emotion":      "joy",
  "intensity":    0.91,
  "secondary":    "surprise",
  "rate_percent": 127.3,
  "pitch_st":     3.19,
  "volume_db":    3.64,
  "emphasis":     "strong",
  "audio_url":    "/audio/joy_a3f8b2c1.mp3",
  "latency_ms":   843
}
```

### GET /audio/{filename}

Returns the generated audio file (MP3 or WAV).

### GET /health

Returns `{"status": "ok", "pipeline_loaded": true}`.

---

## Design Decisions

**Why DistilRoBERTa over BERT?**
DistilRoBERTa is 40% smaller and 60% faster than RoBERTa-base while retaining ~97% of its accuracy. For a real-time service this latency reduction matters significantly.

**Why intensity scaling instead of binary thresholds?**
A binary "happy/not happy" switch produces jarring, unnatural transitions. The continuous intensity score (derived from the model's confidence) creates smooth, proportional modulation — the same way human vocal expressiveness scales with emotional arousal.

**Why pydub for gTTS post-processing?**
gTTS produces natural-sounding speech but has no prosody API. pydub's frame-rate resampling trick for speed/pitch is acoustically equivalent to time-domain stretching for short clips, and requires no additional dependencies beyond ffmpeg.

**Why SSML for Google Cloud?**
Server-side SSML processing produces significantly more natural results than client-side audio manipulation. The synthesis model can apply prosody changes at the phoneme level rather than stretching/resampling the final audio waveform.

**Why is sadness slower with a leading pause?**
This follows the Brunswik Lens Model of vocal emotion expression: sadness is characterized by low speech rate, low pitch, low intensity, and longer inter-utterance pauses. The 600ms leading break before sad speech mimics the effortful, reluctant quality of grieving speech.
