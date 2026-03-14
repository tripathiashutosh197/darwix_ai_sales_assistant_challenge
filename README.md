# Empathy Engine
### Installation & Operations Guide
*Emotionally-aware TTS with Multi-Label Emotion Vectors*

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Project Setup](#4-project-setup)
5. [Download Training Data](#5-download-training-data-optional-but-recommended)
6. [Testing Each Module](#6-testing-each-module)
7. [Running the Application](#7-running-the-application)
8. [Switching Between Detectors](#8-switching-between-detectors)
9. [Troubleshooting](#9-troubleshooting)
10. [Complete Environment Summary](#10-complete-environment-summary)

---

## 1. Overview

The Empathy Engine is a text-to-speech pipeline that detects emotion in input text and dynamically modulates vocal parameters to produce expressive, human-sounding audio output. Unlike standard TTS systems that produce flat monotonic speech, the Empathy Engine detects 15 simultaneous emotion dimensions and maps them to a calibrated combination of speech rate, pitch, and volume.

**Key capabilities:**

- 15-dimensional emotion vector covering base emotions and sales-specific states
- Multi-label detection — a speaker can be surprised and angry simultaneously
- Pitch constrained to 175–225 Hz for speaker identity consistency
- Rate modulation with perceptible minimum thresholds via librosa
- Punctuation-aware synthesis with calibrated pauses at commas, periods, question marks
- Neural voice predictor that learns from usage data over time

---

## 2. System Requirements

| Component | Requirement |
|---|---|
| Operating System | Ubuntu 20.04+ / Debian / macOS / WSL2 |
| Python | 3.11 (required — librosa incompatible with 3.14) |
| Conda | Miniconda or Anaconda |
| ffmpeg | Required for audio format conversion |
| espeak | Required for pyttsx3 (optional engine) |
| RAM | 4 GB minimum, 8 GB recommended |
| Disk | 2 GB for model cache + audio outputs |
| Internet | Required for gTTS synthesis and model download |

---

## 3. Installation

### 3.1 Install System Dependencies

**Ubuntu / Debian / WSL:**
```bash
sudo apt update
sudo apt install ffmpeg espeak espeak-data git -y
```

**macOS:**
```bash
brew install ffmpeg espeak
```

### 3.2 Create Conda Environment

> **Important:** Always use Python 3.11. librosa and numba are incompatible with Python 3.14.

```bash
conda create -n empathy python=3.11 -y
conda activate empathy
```

### 3.3 Install Python Packages

```bash
pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install librosa soundfile soxr gTTS pydub
pip install fastapi uvicorn python-multipart jinja2
pip install python-dotenv numpy scipy requests platformdirs pooch
pip install vaderSentiment
```

### 3.4 Verify Installation

```bash
python3 -c "import transformers, torch, librosa, soundfile, gtts, soxr; print('All OK')"
```

---

## 4. Project Setup

### 4.1 Folder Structure

Create the following structure and place each file in the correct location:

```
empathy_engine/
├── cli.py
├── pipeline.py
├── requirements.txt
├── .env
├── emotion/
│   ├── __init__.py
│   ├── transformer_detector.py
│   └── vader_detector.py
├── tts/
│   ├── __init__.py
│   ├── voice_mapper.py
│   ├── ssml_builder.py
│   └── engines/
│       ├── __init__.py
│       ├── gtts_engine.py
│       ├── pyttsx3_engine.py
│       ├── google_cloud_engine.py
│       └── elevenlabs_engine.py
├── web/
│   ├── __init__.py
│   ├── app.py
│   └── templates/
│       └── index.html
├── utils/
│   ├── __init__.py
│   ├── generate_training_data.py
│   └── evaluate.py
├── outputs/
└── data/
```

### 4.2 Create `__init__.py` Files

These files must exist for Python to recognise each folder as a package:

```bash
cd ~/Documents/vlm/empathy_engine

touch __init__.py
touch emotion/__init__.py
touch tts/__init__.py
touch tts/engines/__init__.py
touch web/__init__.py
touch utils/__init__.py

mkdir -p outputs data
```

### 4.3 Configure `.env`

Create a file named `.env` in the project root:

```env
EMOTION_MODEL=transformer
TTS_ENGINE=gtts
AUDIO_OUTPUT_DIR=outputs/
```

To switch to the VADER detector (faster, offline):

```env
EMOTION_MODEL=vader
```

---

## 5. Download Training Data (Optional but Recommended)

This step downloads the GoEmotions and ISEAR datasets to pre-train the neural voice predictor. Without this step the system uses the analytical fallback mapper, which still works well.

```bash
mkdir -p data

wget -P data https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
wget -P data https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
wget -P data https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
wget -P data https://raw.githubusercontent.com/niderhoff/nlp-datasets/master/data/ISEAR/isear.csv
```

Generate training data from downloaded datasets:

```bash
python3 utils/generate_training_data.py
```

Train the neural voice predictor:

```bash
python3 -c "from tts.voice_mapper import train_voice_predictor; train_voice_predictor(epochs=3000)"
```

---

## 6. Testing Each Module

Run these in order to verify each component works before running the full pipeline.

| Module | Test Command |
|---|---|
| Voice mapper | `python3 -m tts.voice_mapper` |
| SSML builder | `python3 -m tts.ssml_builder` |
| Transformer detector | `python3 -m emotion.transformer_detector` |
| VADER detector | `python3 -m emotion.vader_detector` |
| gTTS engine | `python3 -m tts.engines.gtts_engine` |

---

## 7. Running the Application

### 7.1 CLI — Single Sentence

```bash
conda activate empathy
cd ~/Documents/vlm/empathy_engine
python3 cli.py "I cannot believe this happened again!"
```

### 7.2 CLI — Demo Mode (All Emotions)

```bash
python3 cli.py --demo
```

### 7.3 CLI — Batch from File

```bash
python3 cli.py --file sentences.txt
```

### 7.4 CLI — Override Detector

```bash
python3 cli.py "text here" --emotion vader
python3 cli.py "text here" --emotion transformer
```

### 7.5 Web Interface

```bash
uvicorn web.app:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser.

### 7.6 Multi-Emotion Pipeline Test

```bash
python3 -c "
from pipeline import EmpathyPipeline
p = EmpathyPipeline()

sentences = [
    ('I cannot believe this happened again, every single time I call nothing gets fixed!!!',  'frustrated_urgent'),
    ('Oh wow that is absolutely incredible, I love this deal, sign me up right now!',         'excited_joy'),
    ('I am so relieved, finally someone sorted this out, thank you so much!',                 'relief_joy'),
    ('Wait, are you serious? That sounds too good to be true, what is the catch?',            'surprised_skeptical'),
    ('I understand that must have been really difficult for you, we will sort this out.',      'empathy_calm'),
    ('I have been on hold for forty minutes and nobody can give me a straight answer!!!',     'frustrated_anger'),
    ('I do not understand what you mean by that, can you explain in simple terms please?',    'confused_neutral'),
    ('I am scared my account has been compromised, I need this checked immediately!',         'fear_urgent'),
    ('I thought this was a premium service but the quality has really let me down.',          'disappointment_sadness'),
    ('This is absolutely disgusting behavior, I want to speak to a manager right now!!!',     'anger_disgust'),
]

for text, label in sentences:
    print('=' * 70)
    r = p.run(text, filename=f'{label}.mp3')
    print(f'TEXT:     {text[:70]}')
    print(f'EMOTIONS:')
    top = sorted(r.emotion_vector.items(), key=lambda x: x[1], reverse=True)
    for emotion, score in top:
        if score < 0.08:
            continue
        bar = chr(9608) * int(score * 20)
        print(f'  {emotion:15s} {score:.2f} {bar}')
    print(f'DOMINANT: {r.emotion}  secondary={r.secondary}  intensity={r.intensity:.2f}')
    print(f'VOICE:    rate={r.rate_percent:.0f}%  pitch={r.pitch_hz:.0f}Hz({r.pitch_st:+.2f}st)  vol={r.volume_db:+.1f}dB')
    print(f'AUDIO:    {r.audio_path}')
    print()
"
```

---

## 8. Switching Between Detectors

Both detectors return the same `EmotionVector` format and are fully interchangeable.

| Setting | Description |
|---|---|
| `EMOTION_MODEL=transformer` | HuggingFace DistilRoBERTa — most accurate, handles negation and context, ~200ms latency |
| `EMOTION_MODEL=vader` | Keyword-based — instant, offline, 100–200 keywords per emotion, great on short texts and heavy punctuation |

---

## 9. Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'transformers'` | `pip install transformers` inside `conda activate empathy` |
| `No module named 'tts'` | Run from project root using `python3 -m tts.module_name` |
| `llvmlite build failed` | Wrong Python version — use `conda create -n empathy python=3.11` |
| `ffmpeg not found` | `sudo apt install ffmpeg` then restart terminal |
| Audio sounds robotic | `pip install soxr` — enables high quality resampler |
| `platformdirs not found` | `conda install platformdirs -c conda-forge -y` |
| Intensity too low | Set `EMOTION_MODEL=transformer` for better intensity scoring |
| Port 8000 in use | `uvicorn web.app:app --port 8001` |
| `No module named 'dotenv'` | `pip install python-dotenv` |
| `Object of type float32 is not JSON serializable` | Use `float(v)` before `json.dumps` in `_log_training_sample` |

---

## 10. Complete Environment Summary

| Package | Purpose |
|---|---|
| `transformers` | HuggingFace emotion classification model |
| `torch` | PyTorch backend for transformer inference |
| `librosa` | Time-stretching, pitch-shifting, audio analysis |
| `soundfile` | WAV file read/write |
| `soxr` | High quality audio resampler (eliminates robotic artifacts) |
| `gTTS` | Google Text-to-Speech free API |
| `pydub` | MP3 export via ffmpeg |
| `fastapi` | Web server for the browser UI |
| `uvicorn` | ASGI server to run FastAPI |
| `python-dotenv` | Load `.env` configuration file |
| `vaderSentiment` | Rule-based sentiment for VADER detector |
| `numpy` | Numerical operations for neural network |
| `scipy` | Signal processing utilities |
| `platformdirs` | Required by librosa dependency pooch |
| `soxr` | High quality resampler for pitch shift |
