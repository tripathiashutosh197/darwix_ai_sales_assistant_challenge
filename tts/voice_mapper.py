"""
tts/voice_mapper.py
--------------------
Hybrid voice parameter predictor.

Parameter sources:
  pitch   → old analytical rules, converted to Hz and clamped to 175-225 Hz
             (keeps speaker identity consistent — same person, different moods)
  rate    → old analytical rules, lower bound raised by 30% (min 65% speed)
  volume  → neural network if trained, else analytical vector approach
  emphasis, pause → emotion vector weighted approach

The 175-225 Hz pitch band matches natural human speaker variation.
Staying within this band means pitch changes sound like emotional
inflection rather than a different voice.
"""

from __future__ import annotations
import os
import json
import math
import numpy as np
from dataclasses import dataclass

EMOTIONS = [
    "joy", "surprise", "anger", "sadness", "fear", "disgust", "neutral",
    "frustration", "urgency", "confusion", "excitement",
    "skepticism", "empathy", "disappointment", "relief",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "voice_predictor.npz")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "training_data.jsonl")

# ── Pitch constants ───────────────────────────────────────────────────────────
PITCH_HZ_MIN = 175.0
PITCH_HZ_MAX = 225.0
PITCH_HZ_MID = (PITCH_HZ_MIN + PITCH_HZ_MAX) / 2.0   # 200 Hz baseline

# ── Rate constants ────────────────────────────────────────────────────────────
RATE_LOWER_BOUND = 50.0 * 1.3    # original 50% raised by 30% → 65%
RATE_UPPER_BOUND = 200.0

# ── Analytical rules (pitch + rate source) ───────────────────────────────────
_RULES = {
    #                rate_delta  pitch_st  volume_db  pause_ms
    "joy":          (+45,        +1.5,     +4.0,      0),
    "surprise":     (+35,        +2.0,     +3.0,      300),
    "anger":        (+40,        +1.0,     +6.0,      0),
    "sadness":      (-40,        -1.5,     -3.0,      600),
    "fear":         (+30,        +1.2,     -2.0,      0),
    "disgust":      (-20,        -0.5,     +2.0,      0),
    "neutral":      (0,           0.0,      0.0,      0),
    "frustration":  (+30,        +0.8,     +5.0,      0),
    "urgency":      (+50,        +1.0,     +5.5,      0),
    "confusion":    (-10,        +0.5,     -1.0,      200),
    "excitement":   (+50,        +1.8,     +4.5,      0),
    "skepticism":   (-15,        -0.3,     -0.5,      300),
    "empathy":      (-20,        -0.5,     -1.5,      400),
    "disappointment":(-25,       -1.0,     -2.0,      500),
    "relief":       (-15,        +0.5,     +1.0,      300),
}

_EMPHASIS_MAP = {
    "joy": "strong", "surprise": "strong", "anger": "strong",
    "excitement": "strong", "urgency": "strong", "frustration": "strong",
    "fear": "moderate", "disgust": "moderate", "confusion": "moderate",
    "skepticism": "moderate", "empathy": "none",
    "sadness": "none", "disappointment": "none", "relief": "none", "neutral": "none",
}


@dataclass
class VoiceParams:
    rate_percent:    float    # speech rate as % of baseline
    pitch_st:        float    # pitch in semitones (for librosa)
    pitch_hz:        float    # pitch in Hz (175-225 band)
    volume_db:       float    # volume in dB
    emphasis:        str      # none | moderate | strong
    pause_before_ms: int      # leading silence in ms
    emotion:         str      # dominant emotion label
    intensity:       float    # overall intensity 0.0-1.0


class VoiceMapper:
    """
    Hybrid mapper:
      - pitch  : analytical → Hz band 175-225
      - rate   : analytical → lower bound +30%
      - volume : neural if available, else analytical
    """

    def __init__(self):
        self._net = self._load_network()
        if self._net:
            print("[VoiceMapper] Neural predictor loaded.")
        else:
            print("[VoiceMapper] No trained model found — using analytical mapper.")
            print("[VoiceMapper] Training data being collected → tts/training_data.jsonl")

    # ── Public interface ──────────────────────────────────────────────────────

    def map(self, label: str, intensity: float) -> VoiceParams:
        """Compatibility shim — accepts simple label+intensity."""
        scores = {e: 0.0 for e in EMOTIONS}
        scores[label]      = intensity
        scores["neutral"]  = max(0.0, 1.0 - intensity)

        class _FakeVector:
            dominant  = label
            secondary = "neutral"
        _FakeVector.scores    = scores
        _FakeVector.intensity = intensity

        return self.map_vector(_FakeVector)

    def map_vector(self, ev) -> VoiceParams:
        """
        Hybrid parameter selection:
          pitch   → analytical semitones → converted to 175-225 Hz band
          rate    → analytical with lower bound raised 30% (min 65%)
          volume  → neural if trained, else analytical
          emphasis, pause → emotion vector weighted
        """
        vec = np.array([ev.scores.get(e, 0.0) for e in EMOTIONS], dtype=np.float32)

        # ── Pitch + Rate: always from analytical approach ─────────────────
        rate_analytical, pitch_st_analytical, volume_analytical = self._analytical(vec, ev)

        # ── Volume: neural if available, else analytical ──────────────────
        if self._net:
            _, _, volume_db = self._predict(vec)
        else:
            volume_db = volume_analytical

        # Always log training sample
        self._log_training_sample(vec, rate_analytical, pitch_st_analytical, volume_db)

        # ── Pitch: convert analytical semitones → Hz in 175-225 band ─────
        pitch_hz = self._semitones_to_hz(pitch_st_analytical)
        # Convert Hz back to semitones relative to 200 Hz for librosa
        pitch_st = round(12 * math.log2(pitch_hz / PITCH_HZ_MID), 2) if pitch_hz != PITCH_HZ_MID else 0.0

        # ── Rate: analytical with 30% higher lower bound ──────────────────
        rate_percent = max(RATE_LOWER_BOUND, min(RATE_UPPER_BOUND, rate_analytical))

        # ── Volume: clamp ─────────────────────────────────────────────────
        volume_db = max(-8.0, min(10.0, volume_db))

        # ── Pause: weighted average of per-emotion pauses ─────────────────
        pause_ms = int(sum(
            ev.scores.get(e, 0) * _RULES.get(e, (0, 0, 0, 0))[3]
            for e in EMOTIONS
        ))

        # ── Emphasis: dominant emotion, downgraded at low intensity ───────
        emphasis = _EMPHASIS_MAP.get(ev.dominant, "moderate")
        if ev.intensity < 0.35:
            emphasis = "none"
        elif ev.intensity < 0.55 and emphasis == "strong":
            emphasis = "moderate"

        return VoiceParams(
            rate_percent    = round(rate_percent, 1),
            pitch_st        = pitch_st,
            pitch_hz        = pitch_hz,
            volume_db       = round(volume_db, 2),
            emphasis        = emphasis,
            pause_before_ms = min(pause_ms, 800),
            emotion         = ev.dominant,
            intensity       = round(ev.intensity, 3),
        )

    def describe(self, params: VoiceParams) -> str:
        return (
            f"emotion={params.emotion} intensity={params.intensity:.2f} | "
            f"rate={params.rate_percent:.0f}%  "
            f"pitch={params.pitch_hz:.0f}Hz({params.pitch_st:+.2f}st)  "
            f"vol={params.volume_db:+.1f}dB  "
            f"emphasis={params.emphasis}  "
            f"pause={params.pause_before_ms}ms"
        )

    # ── Pitch conversion ──────────────────────────────────────────────────────

    def _semitones_to_hz(self, semitones: float) -> float:
        """
        Map analytical semitone shift to Hz within the 175-225 Hz band.

        Baseline = 200 Hz (midpoint).
        +2.0 st max → 225 Hz
        -1.5 st max → 175 Hz
        Linear interpolation within band.
        """
        if semitones >= 0:
            hz = PITCH_HZ_MID + (semitones / 2.0) * (PITCH_HZ_MAX - PITCH_HZ_MID)
        else:
            hz = PITCH_HZ_MID + (semitones / 1.5) * (PITCH_HZ_MID - PITCH_HZ_MIN)
        return round(max(PITCH_HZ_MIN, min(PITCH_HZ_MAX, hz)), 1)

    # ── Analytical fallback ───────────────────────────────────────────────────

    def _analytical(self, vec: np.ndarray, ev) -> tuple[float, float, float]:
        """
        Weighted sum of per-emotion rules.
        Neutral excluded so it does not dilute emotional modulation.
        """
        rate_delta = 0.0
        pitch_st   = 0.0
        volume_db  = 0.0

        neutral_idx  = EMOTIONS.index("neutral")
        active_vec   = vec.copy()
        active_vec[neutral_idx] = 0.0
        total_weight = active_vec.sum()

        if total_weight < 1e-6:
            return 100.0, 0.0, 0.0

        for i, emotion in enumerate(EMOTIONS):
            if emotion == "neutral":
                continue
            w = active_vec[i]
            if w < 0.05:
                continue
            rd, ps, vd, _ = _RULES.get(emotion, (0, 0, 0, 0))
            rate_delta += w * rd
            pitch_st   += w * ps
            volume_db  += w * vd

        rate_delta /= total_weight
        pitch_st   /= total_weight
        volume_db  /= total_weight

        # Use max active score as intensity floor to prevent
        # strong individual emotions being washed out by low overall intensity
        active_intensity    = float(active_vec.max())
        effective_intensity = max(ev.intensity, active_intensity * 0.85)

        return (
            100.0 + rate_delta * effective_intensity,
            pitch_st  * effective_intensity,
            volume_db * effective_intensity,
        )

    # ── Training data logger ──────────────────────────────────────────────────

    def _log_training_sample(self, vec, rate, pitch, volume):
        """Save sample as training data for future neural training."""
        sample = {
            "x": [float(v) for v in vec],
            "y": [float(rate), float(pitch), float(volume)],
        }
        with open(DATA_PATH, "a") as f:
            f.write(json.dumps(sample) + "\n")

    # ── Neural network ────────────────────────────────────────────────────────

    def _load_network(self):
        """Load trained weights if they exist."""
        if not os.path.exists(MODEL_PATH):
            return None
        try:
            data = np.load(MODEL_PATH)
            return {
                "W1": data["W1"], "b1": data["b1"],
                "W2": data["W2"], "b2": data["b2"],
                "W3": data["W3"], "b3": data["b3"],
                "x_mean": data["x_mean"], "x_std": data["x_std"],
                "y_mean": data["y_mean"], "y_std": data["y_std"],
            }
        except Exception as e:
            print(f"[VoiceMapper] Could not load neural model: {e}")
            return None

    def _predict(self, vec: np.ndarray) -> tuple[float, float, float]:
        """Forward pass through the 3-layer MLP."""
        net = self._net
        x  = (vec - net["x_mean"]) / (net["x_std"] + 1e-8)
        h1 = np.tanh(x  @ net["W1"] + net["b1"])
        h2 = np.tanh(h1 @ net["W2"] + net["b2"])
        y  = h2 @ net["W3"] + net["b3"]
        y  = y * net["y_std"] + net["y_mean"]
        return float(y[0]), float(y[1]), float(y[2])


# ── Neural network trainer ────────────────────────────────────────────────────

def train_voice_predictor(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
    epochs: int = 2000,
    lr: float = 0.001,
    hidden: int = 64,
):
    """
    Train the neural voice predictor on collected data.

    Usage:
        python3 -c "from tts.voice_mapper import train_voice_predictor; train_voice_predictor()"
    """
    if not os.path.exists(data_path):
        print(f"No training data found at {data_path}")
        print("Run: python3 utils/generate_training_data.py")
        return

    X, Y = [], []
    with open(data_path) as f:
        for line in f:
            s = json.loads(line)
            X.append(s["x"])
            Y.append(s["y"])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    print(f"Loaded {len(X)} training samples.")

    if len(X) < 100:
        print("Warning: fewer than 100 samples. Collect more data first.")

    x_mean, x_std = X.mean(0), X.std(0)
    y_mean, y_std = Y.mean(0), Y.std(0)
    Xn = (X - x_mean) / (x_std + 1e-8)
    Yn = (Y - y_mean) / (y_std + 1e-8)

    n_in, n_out = X.shape[1], Y.shape[1]
    rng = np.random.default_rng(42)

    W1 = rng.standard_normal((n_in,  hidden)).astype(np.float32) * 0.1
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.1
    b2 = np.zeros(hidden, dtype=np.float32)
    W3 = rng.standard_normal((hidden, n_out)).astype(np.float32) * 0.1
    b3 = np.zeros(n_out,  dtype=np.float32)

    def forward(x):
        h1 = np.tanh(x  @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        return h2 @ W3 + b3, h1, h2

    for epoch in range(epochs):
        pred, h1, h2 = forward(Xn)
        loss = ((pred - Yn) ** 2).mean()

        d_out = 2 * (pred - Yn) / len(Xn)
        dW3 = h2.T @ d_out;  db3 = d_out.sum(0)
        d_h2 = d_out @ W3.T * (1 - h2**2)
        dW2 = h1.T @ d_h2;   db2 = d_h2.sum(0)
        d_h1 = d_h2 @ W2.T * (1 - h1**2)
        dW1 = Xn.T @ d_h1;   db1 = d_h1.sum(0)

        W3 -= lr * dW3;  b3 -= lr * db3
        W2 -= lr * dW2;  b2 -= lr * db2
        W1 -= lr * dW1;  b1 -= lr * db1

        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={loss:.6f}")

    np.savez(
        model_path,
        W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
        x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std,
    )
    print(f"Model saved to {model_path}")


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mapper = VoiceMapper()

    class FakeVector:
        dominant  = "frustration"
        secondary = "urgency"
        intensity = 0.85
        scores    = {e: 0.0 for e in EMOTIONS}

    cases = [
        ("joy",          "excitement",  {"joy": 0.82, "excitement": 0.74, "surprise": 0.30}),
        ("frustration",  "urgency",     {"frustration": 0.78, "urgency": 0.65, "anger": 0.40}),
        ("surprise",     "anger",       {"surprise": 0.70, "anger": 0.60, "frustration": 0.35}),
        ("surprise",     "joy",         {"surprise": 0.65, "joy": 0.72, "excitement": 0.50}),
        ("empathy",      "sadness",     {"empathy": 0.80, "sadness": 0.45, "disappointment": 0.30}),
        ("confusion",    "frustration", {"confusion": 0.70, "frustration": 0.55}),
        ("relief",       "joy",         {"relief": 0.88, "joy": 0.60}),
        ("neutral",      "neutral",     {"neutral": 0.95}),
    ]

    for dominant, secondary, score_override in cases:
        ev = FakeVector()
        ev.dominant  = dominant
        ev.secondary = secondary
        ev.scores    = {e: 0.0 for e in EMOTIONS}
        ev.scores.update(score_override)
        ev.intensity = max(score_override.values())

        p = mapper.map_vector(ev)
        print(f"\n[{dominant:15s} + {secondary:15s}]")
        print(f"  {mapper.describe(p)}")
