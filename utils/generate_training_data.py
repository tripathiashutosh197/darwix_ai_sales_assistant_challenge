"""
utils/generate_training_data.py
---------------------------------
Converts public emotion datasets into the training_data.jsonl format
expected by the neural voice predictor in tts/voice_mapper.py

Each output line:
  {"x": [15 emotion scores], "y": [rate_percent, pitch_st, volume_db]}

Run:
  python3 utils/generate_training_data.py
"""

import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts.voice_mapper import VoiceMapper, EMOTIONS, _RULES

OUTPUT_PATH = "tts/training_data.jsonl"
mapper = VoiceMapper()

# ── GoEmotions label mapping (27 → our 15) ───────────────────────────────────
GOEMOTIONS_MAP = {
    "admiration":     {"joy": 0.7, "excitement": 0.5},
    "amusement":      {"joy": 0.8, "surprise": 0.4},
    "anger":          {"anger": 0.9, "frustration": 0.6},
    "annoyance":      {"frustration": 0.8, "anger": 0.5},
    "approval":       {"joy": 0.6, "relief": 0.4},
    "caring":         {"empathy": 0.9, "joy": 0.3},
    "confusion":      {"confusion": 0.9, "fear": 0.2},
    "curiosity":      {"confusion": 0.5, "surprise": 0.5},
    "desire":         {"excitement": 0.7, "joy": 0.5},
    "disappointment": {"disappointment": 0.9, "sadness": 0.5},
    "disapproval":    {"frustration": 0.7, "disgust": 0.5},
    "disgust":        {"disgust": 0.9, "frustration": 0.4},
    "embarrassment":  {"fear": 0.6, "sadness": 0.5},
    "excitement":     {"excitement": 0.9, "joy": 0.7, "surprise": 0.4},
    "fear":           {"fear": 0.9, "urgency": 0.5},
    "gratitude":      {"joy": 0.7, "relief": 0.6},
    "grief":          {"sadness": 0.9, "disappointment": 0.5},
    "joy":            {"joy": 0.95, "excitement": 0.5},
    "love":           {"joy": 0.8, "empathy": 0.6},
    "nervousness":    {"fear": 0.8, "urgency": 0.4},
    "neutral":        {"neutral": 0.95},
    "optimism":       {"joy": 0.6, "excitement": 0.5, "relief": 0.3},
    "pride":          {"joy": 0.7, "excitement": 0.6},
    "realization":    {"surprise": 0.7, "confusion": 0.3},
    "relief":         {"relief": 0.9, "joy": 0.5},
    "remorse":        {"sadness": 0.7, "disappointment": 0.6},
    "sadness":        {"sadness": 0.9, "disappointment": 0.4},
    "surprise":       {"surprise": 0.9, "excitement": 0.3},
    # sales-specific compounds
    "anger+fear":     {"anger": 0.6, "fear": 0.5, "urgency": 0.7},
    "anger+confusion":{"anger": 0.6, "confusion": 0.6, "frustration": 0.7},
}

# ── ISEAR label mapping ───────────────────────────────────────────────────────
ISEAR_MAP = {
    "1": {"joy": 0.9, "excitement": 0.5},
    "2": {"fear": 0.9, "urgency": 0.4},
    "3": {"anger": 0.9, "frustration": 0.6},
    "4": {"sadness": 0.9, "disappointment": 0.5},
    "5": {"disgust": 0.9, "frustration": 0.4},
    "6": {"sadness": 0.7, "fear": 0.5},
    "7": {"sadness": 0.6, "disappointment": 0.6},
}


def scores_to_vector(scores: dict) -> np.ndarray:
    """Convert a partial scores dict to a full 15-dim numpy vector."""
    vec = np.zeros(len(EMOTIONS), dtype=np.float32)
    total = sum(scores.values())
    for i, emotion in enumerate(EMOTIONS):
        vec[i] = scores.get(emotion, 0.0)
    # Add small neutral component if nothing else is strong
    if total < 0.5:
        neutral_idx = EMOTIONS.index("neutral")
        vec[neutral_idx] = max(vec[neutral_idx], 0.3)
    return vec


def vector_to_params(vec: np.ndarray) -> tuple[float, float, float]:
    """Use analytical mapper to get ground truth params for this vector."""

    class FakeVector:
        scores    = {e: float(vec[i]) for i, e in enumerate(EMOTIONS)}
        dominant  = EMOTIONS[int(np.argmax(vec))]
        secondary = EMOTIONS[int(np.argsort(vec)[-2])]
        intensity = float(np.sqrt(np.sum(vec**2) / len(vec)) * 1.4)
        intensity = min(intensity, 1.0)

    rate, pitch, volume = mapper._analytical(vec, FakeVector)
    return rate, pitch, volume


def save_sample(vec: np.ndarray, rate: float, pitch: float, volume: float, f):
    sample = {
        "x": [float(v) for v in vec],
        "y": [float(rate), float(pitch), float(volume)],
    }
    f.write(json.dumps(sample) + "\n")


def process_goemotions(paths: list[str], f) -> int:
    count = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"  Skipping {path} — not found")
            continue
        with open(path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                text = row.get("text", "").strip()
                if not text:
                    continue
                # Find all active emotion columns (value == "1")
                active = []
                for col, val in row.items():
                    if col in GOEMOTIONS_MAP and val.strip() == "1":
                        active.append(col)
                if not active:
                    continue
                # Merge scores from all active emotions
                merged: dict[str, float] = {}
                for label in active:
                    for emotion, score in GOEMOTIONS_MAP[label].items():
                        merged[emotion] = min(1.0, merged.get(emotion, 0.0) + score * (1.0 / len(active)))
                vec = scores_to_vector(merged)
                rate, pitch, volume = vector_to_params(vec)
                save_sample(vec, rate, pitch, volume, f)
                count += 1
    return count


def process_isear(path: str, f) -> int:
    count = 0
    if not os.path.exists(path):
        print(f"  Skipping {path} — not found")
        return 0
    with open(path, newline="", encoding="latin-1") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            emot = row.get("EMOT", "").strip()
            if emot not in ISEAR_MAP:
                continue
            scores = ISEAR_MAP[emot]
            vec    = scores_to_vector(scores)
            rate, pitch, volume = vector_to_params(vec)
            save_sample(vec, rate, pitch, volume, f)
            count += 1
    return count


def generate_synthetic(f) -> int:
    """
    Generate synthetic samples covering all emotion combinations
    including sales-specific ones. Covers edge cases that datasets miss.
    """
    import itertools
    count = 0

    # Single emotion at varying intensities
    for emotion in EMOTIONS:
        for intensity in [0.3, 0.5, 0.7, 0.9, 1.0]:
            scores = {emotion: intensity}
            if emotion != "neutral":
                scores["neutral"] = max(0.0, 0.3 * (1 - intensity))
            vec = scores_to_vector(scores)
            rate, pitch, volume = vector_to_params(vec)
            save_sample(vec, rate, pitch, volume, f)
            count += 1

    # Sales-relevant pairs at varying ratios
    sales_pairs = [
        ("frustration", "urgency"),
        ("surprise",    "anger"),
        ("surprise",    "joy"),
        ("confusion",   "frustration"),
        ("empathy",     "sadness"),
        ("excitement",  "joy"),
        ("skepticism",  "confusion"),
        ("disappointment", "anger"),
        ("relief",      "joy"),
        ("urgency",     "fear"),
        ("anger",       "disappointment"),
        ("excitement",  "surprise"),
    ]

    ratios = [
        (0.9, 0.3), (0.7, 0.5), (0.6, 0.6),
        (0.5, 0.7), (0.3, 0.9), (0.8, 0.8),
    ]

    for (e1, e2), (s1, s2) in itertools.product(sales_pairs, ratios):
        scores = {e1: s1, e2: s2}
        vec    = scores_to_vector(scores)
        rate, pitch, volume = vector_to_params(vec)
        save_sample(vec, rate, pitch, volume, f)
        count += 1

    # Three-emotion combinations (common in sales calls)
    triples = [
        ("frustration", "urgency",    "anger",      0.7, 0.6, 0.5),
        ("surprise",    "joy",        "excitement", 0.8, 0.7, 0.6),
        ("confusion",   "fear",       "urgency",    0.6, 0.5, 0.7),
        ("empathy",     "sadness",    "disappointment", 0.8, 0.5, 0.4),
        ("skepticism",  "confusion",  "frustration",0.6, 0.5, 0.4),
        ("relief",      "joy",        "surprise",   0.9, 0.7, 0.5),
        ("anger",       "frustration","disappointment", 0.8, 0.7, 0.6),
        ("excitement",  "joy",        "surprise",   0.9, 0.8, 0.6),
    ]

    for e1, e2, e3, s1, s2, s3 in triples:
        scores = {e1: s1, e2: s2, e3: s3}
        vec    = scores_to_vector(scores)
        rate, pitch, volume = vector_to_params(vec)
        save_sample(vec, rate, pitch, volume, f)
        count += 1

    return count


def main():
    os.makedirs("tts", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("Generating training data...\n")
    total = 0

    with open(OUTPUT_PATH, "w") as f:

        # GoEmotions
        print("Processing GoEmotions...")
        goe_paths = [
            "data/goemotions_1.csv",
            "data/goemotions_2.csv",
            "data/goemotions_3.csv",
        ]
        n = process_goemotions(goe_paths, f)
        print(f"  GoEmotions: {n} samples")
        total += n

        # ISEAR
        print("Processing ISEAR...")
        n = process_isear("data/isear.csv", f)
        print(f"  ISEAR: {n} samples")
        total += n

        # Synthetic combinations
        print("Generating synthetic combinations...")
        n = generate_synthetic(f)
        print(f"  Synthetic: {n} samples")
        total += n

    print(f"\nTotal samples written: {total}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nNow train the neural predictor:")
    print("  python3 -c \"from tts.voice_mapper import train_voice_predictor; train_voice_predictor()\"")


if __name__ == "__main__":
    main()
