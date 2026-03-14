# Empathy Engine
### Technical Architecture & System Report
*Function, Design Rationale, and System Impact of Every Module*

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Emotion Detection Layer](#2-emotion-detection-layer-emotion)
3. [Voice Parameter Mapping](#3-voice-parameter-mapping-ttsvoice_mapperpy)
4. [SSML Builder](#4-ssml-builder-ttsssml_builderpy)
5. [Audio Synthesis Engine](#5-audio-synthesis-engine-ttsenginesgtts_enginepy)
6. [Core Pipeline](#6-core-pipeline-pipelinepy)
7. [Interfaces](#7-interfaces-clipy-and-web)
8. [Training Data Pipeline](#8-training-data-pipeline-utils)
9. [Key Design Decisions](#9-key-design-decisions--their-impact)
10. [Complete File Reference](#10-complete-file-reference)

---

## 1. Architecture Overview

The Empathy Engine is structured as a linear pipeline of four loosely-coupled modules. Each module has a clearly defined input and output contract, making them independently testable and replaceable. The pipeline processes one text string and produces one audio file, with all intermediate values returned in a `PipelineResult` object for full transparency.

| Stage | Module → Output |
|---|---|
| 1. Input | Raw text string |
| 2. Emotion Detection | `emotion/` → `EmotionVector` (15 float scores) |
| 3. Voice Mapping | `tts/voice_mapper.py` → `VoiceParams` (rate, pitch, volume) |
| 4. Synthesis | `tts/engines/gtts_engine.py` → `.mp3` audio file |
| 5. Interface | `cli.py` or `web/app.py` → user-facing output |

The key architectural decision is that emotion is represented as a **vector** rather than a single label. This means "frustrated and urgent" and "surprised and happy" are structurally different inputs that produce structurally different voice parameters — rather than both being collapsed into a single emotion bucket.

---

## 2. Emotion Detection Layer (`emotion/`)

### 2.1 `emotion/__init__.py` — Detector Factory

This file is the single entry point for emotion detection across the entire system. It reads the `EMOTION_MODEL` environment variable and returns either a `VADEREmotionDetector` or a `TransformerEmotionDetector` instance. Both return an `EmotionVector` so the rest of the system never needs to know which detector is active.

> **Technical:** Implements the factory pattern. The pipeline, CLI, and web server all call `get_detector()` and receive a duck-typed object with a single `detect(text)` method.

> **System Impact:** Allows switching between fast offline detection and accurate neural detection with a single `.env` change, with zero code changes in any other file.

---

### 2.2 `emotion/transformer_detector.py` — Neural Emotion Classifier

This is the primary emotion detector. It wraps the `j-hartmann/emotion-english-distilroberta-base` model from HuggingFace, a DistilRoBERTa model fine-tuned on multiple emotion datasets including GoEmotions, ISEAR, and Twitter data. The model returns confidence scores for 7 base emotions simultaneously.

#### Base emotion classification

The model is loaded with `top_k=None` which returns scores for all 7 classes rather than just the highest. This gives a probability distribution across emotions rather than a single label, which is the foundation of the multi-label vector approach.

#### Sales emotion derivation

Eight additional sales-specific emotions are derived analytically from the base 7 scores combined with keyword matching. Each derived emotion is computed as a weighted blend of contributing base emotions plus a keyword boost:

| Sales Emotion | Base Contributors | Keyword Signal |
|---|---|---|
| frustration | anger(0.6) + sadness(0.4) | again, every time, still not fixed |
| urgency | fear(0.5) + anger(0.5) | immediately, asap, critical, deadline |
| confusion | neutral(0.4) + fear(0.6) | don't understand, what do you mean |
| excitement | joy(0.55) + surprise(0.45) | sign me up, can't wait, love this |
| skepticism | disgust(0.5) + neutral(0.5) | too good to be true, are you sure |
| empathy | joy(0.45) + sadness(0.55) | i understand, that must be hard |
| disappointment | sadness(0.65) + disgust(0.35) | expected better, let me down |
| relief | joy(0.7) + neutral(0.3) | finally, relieved, at last, sorted |

#### Intensity scaling

The overall intensity of the `EmotionVector` is computed as the L2 norm of non-neutral scores, then scaled by a factor of 2.5 and clamped to 1.0. This aggressive scaling ensures that even moderate emotional text produces a perceptible intensity value rather than a low score that would result in near-neutral voice modulation.

> **Technical:** L2 norm gives a geometrically meaningful measure of overall emotional activation across all 15 dimensions simultaneously, unlike averaging which would be dragged down by zero-scored emotions.

> **System Impact:** The 2.5× scaling factor directly controls how expressive the final audio sounds. Without it, most text produces intensity scores below 0.4 which maps to near-neutral voice parameters.

---

### 2.3 `emotion/vader_detector.py` — Keyword-Based Detector

The VADER detector is an alternative to the transformer that runs entirely offline with microsecond latency. It uses the VADER sentiment analyzer for compound scoring and augments it with a custom lexicon of 100–200 keywords per emotion, covering all 15 emotion dimensions.

#### Lexicon design

Each emotion's keyword list covers four vocabulary registers: direct expression ("I am furious"), indirect signals ("every single time"), sales-call specific language ("demand a refund, speak to a manager"), and intensity modifiers ("absolutely, beyond"). The lists are long enough to provide good coverage without requiring exact phrase matching.

#### Scoring pipeline

The detector scores each emotion through four stages: VADER compound provides a positive/negative baseline, keyword hits provide per-emotion signal, punctuation and capitalisation provide an intensity boost (each exclamation mark adds 0.08, caps ratio adds up to 0.30), and finally neutral is set inversely proportional to total emotional content.

| Characteristic | VADER | Transformer |
|---|---|---|
| Latency | Microseconds | 100–300ms |
| Short texts | Excellent | Uncertain |
| Punctuation/caps | Excellent | Partially |
| Context/negation | Weak | Excellent |
| Offline | Always | Needs cached model |
| Accuracy overall | ~72% | ~85% |

> **System Impact:** Use VADER for real-time applications where latency matters or internet is unavailable. Use Transformer for highest accuracy on complex multi-clause sentences.

---

## 3. Voice Parameter Mapping (`tts/voice_mapper.py`)

The voice mapper is the mathematical heart of the system. It converts a 15-dimensional emotion vector into three continuous voice parameters: speech rate as a percentage of baseline, pitch in semitones (converted to Hz), and volume in decibels. It implements a hybrid approach where different parameters come from different sources.

### 3.1 Hybrid Parameter Architecture

| Parameter | Source & Rationale |
|---|---|
| Pitch (Hz) | Analytical rules converted to 175–225 Hz band. Constrained to preserve speaker identity. |
| Rate (%) | Analytical rules with 30% higher lower bound (min 65%). Ensures speech always intelligible. |
| Volume (dB) | Neural network if trained, else analytical. Most nuanced parameter benefits from learning. |
| Emphasis | Dominant emotion lookup table, intensity-gated. |
| Pause (ms) | Weighted average across all 15 emotion pauses. Accounts for blended emotional states. |

### 3.2 Analytical Rules

The `_RULES` dictionary defines per-emotion parameter deltas at full intensity (intensity = 1.0). These are psychoacoustically grounded values:

| Emotion | Rate delta | Pitch (st) | Volume (dB) |
|---|---|---|---|
| joy | +45% | +1.5st | +4.0dB |
| surprise | +35% | +2.0st | +3.0dB |
| anger | +40% | +1.0st | +6.0dB |
| sadness | -40% | -1.5st | -3.0dB |
| frustration | +30% | +0.8st | +5.0dB |
| urgency | +50% | +1.0st | +5.5dB |
| excitement | +50% | +1.8st | +4.5dB |
| empathy | -20% | -0.5st | -1.5dB |
| relief | -15% | +0.5st | +1.0dB |
| confusion | -10% | +0.5st | -1.0dB |

### 3.3 Neutral Exclusion

When computing the weighted sum of emotion contributions, the neutral emotion score is explicitly excluded from the calculation. Neutral always scores 0.6–0.9 even in emotional text because the model returns probability distributions that sum to 1. Without exclusion, neutral would dilute every emotion toward zero modulation.

> **Technical:** The `active_vec` excludes `neutral_idx` before computing `total_weight`. This means `rate_delta` is normalised over the sum of actual emotional scores only, not the full 15-dim sum.

> **System Impact:** Without this fix, a text scoring `joy=0.7` but `neutral=0.9` would produce a weighted rate of only +14% instead of +45%. The fix ensures emotional strength is measured relative to other emotions, not relative to neutral.

### 3.4 Pitch Hz Band (175–225 Hz)

Pitch modulation is constrained to a 25 Hz band centred at 200 Hz. This range represents the natural variation of a single human speaker across emotional states. The analytical semitone shift is linearly mapped into this band: +2.0 semitones maximum maps to 225 Hz, -1.5 semitones minimum maps to 175 Hz.

The semitone value is then converted back from Hz for use by librosa's `pitch_shift` function:

```
pitch_st = 12 × log2(pitch_hz / 200.0)
```

> **Technical:** 200 Hz is chosen as baseline because it sits near the midpoint of the average human speaking voice fundamental frequency range (85–255 Hz for mixed gender).

> **System Impact:** Without this constraint, large pitch shifts (e.g. +4 semitones for high-intensity joy) would make the voice sound like a different person. With the band constraint, all emotions sound like the same speaker in different moods.

### 3.5 Neural Voice Predictor

The neural predictor is a 3-layer MLP (multilayer perceptron) with architecture:

```
15 inputs → 64 hidden (tanh) → 64 hidden (tanh) → 3 outputs
```

It is trained entirely in NumPy using vanilla backpropagation, with no PyTorch dependency for inference. Training data is collected automatically every time the analytical mapper runs, logged to `tts/training_data.jsonl`.

After training on 60,000+ samples from GoEmotions and ISEAR, the network captures non-obvious combinations that the analytical rules cannot express — such as the difference between frustrated+urgent (loud and fast) versus frustrated+empathetic (slower, more measured).

> **Technical:** The network predicts only `volume_db`. Rate and pitch continue to use analytical rules for determinism and speaker consistency. Selective application means the neural component improves expressiveness without risking the speaker identity constraint.

---

## 4. SSML Builder (`tts/ssml_builder.py`)

The SSML builder converts `VoiceParams` into Speech Synthesis Markup Language XML. SSML is supported by Google Cloud TTS, Amazon Polly, and Microsoft Azure TTS. The builder wraps the text in `<prosody>` tags with rate, pitch, and volume attributes, adds `<emphasis>` tags for strong/moderate emphasis levels, and inserts `<break>` tags for leading pauses.

Example output for anger at intensity 0.85:

```xml
<speak>
  <break time="0ms"/>
  <prosody rate="134%" pitch="+0.85st" volume="+5.1dB">
    <emphasis level="strong">
      I demand to speak to a manager immediately.
    </emphasis>
  </prosody>
</speak>
```

> **System Impact:** SSML is used when switching to Google Cloud TTS engine. Server-side prosody processing produces more natural results than post-processing the audio waveform because the TTS model can apply parameter changes at the phoneme level.

---

## 5. Audio Synthesis Engine (`tts/engines/gtts_engine.py`)

The gTTS engine is the primary synthesis component. It combines Google's free Text-to-Speech API for natural voice quality with librosa for high-quality audio manipulation. The engine implements punctuation-aware chunked synthesis to produce natural speech rhythm.

### 5.1 Punctuation-Aware Chunked Synthesis

Text is split at punctuation boundaries using a regex that captures commas, semicolons, colons, periods, exclamation marks, question marks, ellipsis, and em dashes. Each chunk is synthesised separately by gTTS, then calibrated silence is inserted between chunks before concatenation.

| Punctuation | Pause Duration | Purpose |
|---|---|---|
| `,` comma | 180ms | Brief breath between clauses |
| `;` semicolon | 250ms | Medium pause between related clauses |
| `:` colon | 250ms | Pause before explanation or list |
| `.` period | 400ms | Full sentence end pause |
| `?` question | 380ms | Question pause with implied rising intonation |
| `!` exclamation | 350ms | Emphatic pause after declaration |
| `...` ellipsis | 500ms | Trailing off, unfinished thought |
| `—` em dash | 300ms | Dramatic interruption or aside |

> **Technical:** Each chunk gets 8ms fade-in and fade-out applied before concatenation. This eliminates click artifacts at chunk boundaries that would otherwise be audible as sharp transients.

> **System Impact:** Without chunked synthesis, "Wait, are you serious?" would be spoken as a continuous stream. With it, there is a genuine 180ms breath after "Wait" and a 380ms pause after "serious?" — matching natural conversational prosody.

### 5.2 librosa Processing Chain

After concatenation, three transformations are applied in sequence:

| Step | Algorithm & Settings |
|---|---|
| Time-stretch (rate) | `librosa.effects.time_stretch` with `n_fft=2048`, `hop_length=512`, `hann` window. Phase vocoder algorithm preserves spectral envelope. |
| Pitch-shift | `librosa.effects.pitch_shift` with `res_type=soxr_hq`, `bins_per_octave=24`, `n_fft=2048`. Independent from rate via two-pass resampling. |
| Volume | Linear gain applied as `10^(dB/20)` with normalisation to 90% of max amplitude to prevent clipping. |

> **Technical:** Time-stretch and pitch-shift are independent operations. Without this separation, speeding up audio also raises pitch (like a tape speed change). The two-pass approach: (1) stretch time via phase vocoder, (2) shift pitch via frequency-domain resampling with soxr_hq.

> **System Impact:** `soxr_hq` (SoX Resampler high quality) eliminates the metallic robotic artifacts produced by librosa's default `kaiser_best` resampler. The `hann` window in the phase vocoder reduces spectral leakage that causes the characteristic buzzing sound of basic time-stretching.

### 5.3 Minimum Perceptible Thresholds

Rate changes below ±15% are not reliably audible in short audio clips. The engine enforces minimum thresholds so every emotional modulation produces a clearly perceptible difference:

- Rate above baseline: enforced minimum 1.15× (15% faster)
- Rate below baseline: enforced maximum 0.82× (18% slower)

> **System Impact:** Without minimum thresholds, a sentence detected as mild joy (intensity 0.35) might produce rate=107% which is acoustically indistinguishable from 100%. The threshold ensures even mild emotions produce a noticeable change.

---

## 6. Core Pipeline (`pipeline.py`)

The pipeline module wires all components together and manages the execution flow. It exposes a single `run(text)` method that returns a `PipelineResult` dataclass containing all intermediate values alongside the final audio path. This design ensures every parameter used to generate the audio is traceable and logged.

| PipelineResult Field | Description |
|---|---|
| `text` | Original input text |
| `emotion` | Dominant emotion label |
| `secondary` | Second highest emotion label |
| `emotion_vector` | Full 15-dim score dict |
| `intensity` | Overall arousal 0.0–1.0 |
| `rate_percent` | Final speech rate applied |
| `pitch_st` | Pitch in semitones applied |
| `pitch_hz` | Pitch in Hz (175–225 band) |
| `volume_db` | Volume adjustment applied |
| `emphasis` | Emphasis level: none / moderate / strong |
| `audio_path` | Path to generated `.mp3` file |
| `engine_used` | Which TTS engine was used |
| `latency_ms` | Total processing time in ms |

> **Technical:** The pipeline uses `hasattr(emotion_result, 'scores')` to detect `EmotionVector` vs legacy single-label results, maintaining backward compatibility if the detector is replaced.

---

## 7. Interfaces (`cli.py` and `web/`)

### 7.1 `cli.py` — Command Line Interface

The CLI provides three operating modes: single sentence synthesis, demo mode that runs 11 preset emotional sentences covering all emotion categories, and batch mode that reads sentences from a file. The `_print_result` function displays the full 15-dim emotion vector as a bar chart in the terminal alongside voice parameters and audio path.

> **System Impact:** Demo mode is the fastest way to evaluate system performance. Running `python3 cli.py --demo` generates 11 audio files and prints full vector output, providing an immediate comparative test of all emotion modulations.

### 7.2 `web/app.py` — FastAPI Server

The web server exposes three endpoints. `POST /synthesize` accepts text and returns a full JSON response including the `emotion_vector`, `pitch_hz`, `secondary` emotion, and `audio_url`. `GET /audio/{filename}` serves the generated audio files. `GET /health` returns pipeline load status.

The pipeline is lazy-loaded on first request rather than at startup. This means the server starts in under 1 second and the 2–3 second model loading time is absorbed by the first synthesis request.

### 7.3 `web/templates/index.html` — Browser UI

The web interface renders the full 15-dimensional emotion vector as colour-coded bars, one per emotion. Each emotion has a distinct colour (amber for joy, red for anger, blue for sadness, etc.). Pitch is displayed in both Hz and semitones. The secondary emotion is shown alongside the dominant emotion in the badge.

---

## 8. Training Data Pipeline (`utils/`)

### 8.1 `utils/generate_training_data.py`

This script processes three data sources into the `training_data.jsonl` format expected by the neural voice predictor.

| Data Source | Samples & Coverage |
|---|---|
| GoEmotions | ~54,000 samples. 27 emotions mapped to 15. Covers nuanced combinations like admiration, caring, realization. |
| ISEAR | ~7,000 samples. 7 core emotions. Strong coverage of fear, grief, shame scenarios. |
| Synthetic | ~400 samples. Generated programmatically to cover all 90 emotion pairs and 20 triples. |
| **Total** | **~61,000 samples for neural predictor training** |

GoEmotions (58,000 Reddit comments with 27 emotion labels) is mapped to the 15-emotion schema using a label mapping that handles multi-label annotations. ISEAR (7,000 self-reported emotional situations) provides diverse real-world emotional text. Synthetic samples cover all pairwise and triple-emotion combinations systematically.

### 8.2 `utils/evaluate.py`

Evaluation script that runs either detector on a labeled CSV test set and reports accuracy, per-class F1 score, and a confusion matrix. Used to measure the quality of a fine-tuned model or to compare VADER vs Transformer on a domain-specific dataset.

```bash
python3 utils/evaluate.py --csv data/val.csv --model transformer
python3 utils/evaluate.py --csv data/val.csv --model vader
```

---

## 9. Key Design Decisions & Their Impact

| Decision | Alternative Considered | Why This Approach |
|---|---|---|
| 15-dim emotion vector | Single label classification | Real speech is emotionally complex. A person can be surprised and angry simultaneously. Single labels force an artificial choice. |
| Pitch band 175–225 Hz | Unconstrained pitch shift | Large pitch shifts make the voice sound like a different person. The band preserves speaker identity while allowing expressive variation. |
| Rate floor +30% | Original 50% floor | Very slow speech (50% rate) is difficult to understand. Raising the floor to 65% ensures sad/empathetic speech remains clearly intelligible. |
| Neutral exclusion from weighted sum | Include neutral in calculation | Neutral always scores 0.6–0.9 in emotional text. Including it dilutes all emotional parameters toward zero modulation. |
| Chunked synthesis at punctuation | Single gTTS call per sentence | gTTS ignores punctuation for pausing. Chunking allows calibrated silence at each punctuation mark for natural speech rhythm. |
| librosa over pydub | pydub frame rate trick | pydub's speed change is a frame rate hack that couples rate and pitch. librosa uses proper phase vocoder and frequency-domain algorithms. |
| soxr_hq resampler | Default kaiser_best | kaiser_best produces metallic artifacts on pitch shift. soxr_hq uses a higher-order filter that preserves voice naturalness. |
| Analytical pitch and rate | Full neural prediction | Neural networks can learn unexpected combinations that sound unnatural. Constraining pitch and rate to analytical rules maintains predictability and speaker identity. |
| Volume from neural | Volume from rules | Volume is the most context-dependent parameter. Frustrated+empathetic should be quieter than frustrated+urgent despite similar anger scores. Neural learning captures this nuance. |
| 8ms chunk crossfade | Hard cut between chunks | Hard cuts create audible clicks at chunk join points. 8ms linear fade eliminates the transient without affecting perceived speech rhythm. |
| Intensity × 2.5 scaling | Raw L2 norm | Raw norm values cluster between 0.15–0.40 for most text. Without scaling, nearly all text produces near-neutral voice output. |
| VADER with 100–200 keywords/emotion | Original VADER rules only | Original VADER has no concept of frustration, urgency, or empathy. Extended lexicon brings it to full 15-emotion parity with the transformer. |

---

## 10. Complete File Reference

| File | Category | Function |
|---|---|---|
| `pipeline.py` | Core | Orchestrates all modules, returns PipelineResult with full traceability |
| `emotion/__init__.py` | Emotion | Factory returning VADER or Transformer detector based on .env |
| `emotion/transformer_detector.py` | Emotion | 15-dim vector via HuggingFace model + keyword-based sales emotion derivation |
| `emotion/vader_detector.py` | Emotion | 15-dim vector via VADER compound + 100–200 keywords per emotion |
| `tts/voice_mapper.py` | Mapping | EmotionVector → VoiceParams with hybrid analytical+neural approach |
| `tts/ssml_builder.py` | Mapping | VoiceParams → SSML XML for cloud TTS engines |
| `tts/engines/gtts_engine.py` | Synthesis | Chunked gTTS + librosa modulation pipeline with soxr_hq |
| `tts/engines/pyttsx3_engine.py` | Synthesis | Offline TTS fallback, no API or internet required |
| `tts/engines/google_cloud_engine.py` | Synthesis | Google Cloud TTS with full SSML prosody support |
| `tts/engines/elevenlabs_engine.py` | Synthesis | ElevenLabs premium voice with native emotion parameters |
| `cli.py` | Interface | CLI with single/demo/batch modes and full vector bar display |
| `web/app.py` | Interface | FastAPI server with /synthesize, /audio, /health endpoints |
| `web/templates/index.html` | Interface | Browser UI with 15-emotion bars, pitch Hz display, audio player |
| `utils/generate_training_data.py` | Training | GoEmotions + ISEAR + synthetic → training_data.jsonl |
| `utils/evaluate.py` | Training | Accuracy + F1 + confusion matrix evaluation against labeled CSV |
| `tts/voice_predictor.npz` | Model | Trained MLP weights saved after running train_voice_predictor() |
| `tts/training_data.jsonl` | Data | Auto-collected training samples logged on every pipeline run |
