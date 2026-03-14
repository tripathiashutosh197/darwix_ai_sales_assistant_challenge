"""
emotion/transformer_detector.py
---------------------------------
Multi-label emotion vector detector for sales call audio.

Returns a continuous 15-dimensional vector of emotion scores
instead of a single label. Uses a two-model approach:
  1. j-hartmann/emotion-english-distilroberta-base (base 7 emotions)
  2. Rule-based augmentation for 8 sales-specific emotions derived
     from the base vector + linguistic keyword signals.

No training required.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

# ── All 15 emotion dimensions ────────────────────────────────────────────────
EMOTIONS = [
    # Base emotions from transformer model
    "joy", "surprise", "anger", "sadness", "fear", "disgust", "neutral",
    # Sales-specific derived emotions
    "frustration", "urgency", "confusion", "excitement",
    "skepticism", "empathy", "disappointment", "relief",
]

# Sales-specific keyword signals
_KEYWORDS = {
    "frustration":    ["again", "still", "keep", "every time", "always", "never works",
                       "fed up", "tired of", "over and over", "third time", "keeps happening"],
    "urgency":        ["immediately", "right now", "asap", "urgent", "emergency",
                       "can't wait", "need this now", "deadline", "today", "critical"],
    "confusion":      ["don't understand", "what do you mean", "confused", "unclear",
                       "makes no sense", "what is", "how does", "explain", "lost", "huh"],
    "excitement":     ["can't wait", "love this", "amazing deal", "perfect",
                       "exactly what", "sign me up", "let's do it", "when can we start"],
    "skepticism":     ["really", "are you sure", "sounds too good", "prove it",
                       "i doubt", "not convinced", "hard to believe", "catch", "but"],
    "empathy":        ["understand", "sorry to hear", "that must be", "i see",
                       "appreciate", "hear you", "makes sense", "of course"],
    "disappointment": ["expected", "thought it would", "not what i", "let down",
                       "hoped", "promised", "supposed to", "should have"],
    "relief":         ["finally", "thank goodness", "glad that", "relieved",
                       "at last", "great to hear", "so happy that's resolved"],
}

POSITIVE_EMOTIONS = {"joy", "surprise", "excitement", "relief", "empathy"}
NEGATIVE_EMOTIONS = {"anger", "sadness", "fear", "disgust",
                     "frustration", "urgency", "confusion",
                     "skepticism", "disappointment"}


@dataclass
class EmotionVector:
    scores:    dict        # full 15-dim vector, all 0.0–1.0
    dominant:  str         # highest scoring emotion
    secondary: str         # second highest emotion
    intensity: float       # overall arousal magnitude 0.0–1.0
    compound:  float       # synthetic pos-neg score -1 to +1

    def __getitem__(self, key: str) -> float:
        return self.scores.get(key, 0.0)

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)[:n]

    def __repr__(self):
        top3 = ", ".join(f"{e}={s:.2f}" for e, s in self.top(3))
        return f"EmotionVector({top3}, intensity={self.intensity:.2f})"


class TransformerEmotionDetector:
    """
    Multi-label emotion vector detector.

    Step 1: Run j-hartmann model → 7 base emotion scores
    Step 2: Derive 8 sales-specific scores from base + keywords
    Step 3: Return full 15-dim EmotionVector with aggressive intensity scaling
    """

    MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(self, device: int = -1):
        from transformers import pipeline
        self._pipe = pipeline(
            task="text-classification",
            model=self.MODEL_ID,
            top_k=None,
            device=device,
            truncation=True,
            max_length=512,
        )
        print("Emotion model loaded.")

    def detect(self, text: str) -> EmotionVector:
        # ── Step 1: Base 7 emotions from transformer ──────────────────────
        raw  = self._pipe(text)[0]
        base = {r["label"]: round(r["score"], 4) for r in raw}

        # ── Step 2: Derive sales-specific emotions ────────────────────────
        text_lower = text.lower()
        sales = {}

        sales["frustration"]   = self._derive(base, ["anger", "sadness"],   text_lower, "frustration",
                                               base_weights=[0.6, 0.4], keyword_boost=0.35)
        sales["urgency"]       = self._derive(base, ["fear", "anger"],      text_lower, "urgency",
                                               base_weights=[0.5, 0.5], keyword_boost=0.40)
        sales["confusion"]     = self._derive(base, ["neutral", "fear"],    text_lower, "confusion",
                                               base_weights=[0.4, 0.6], keyword_boost=0.45)
        sales["excitement"]    = self._derive(base, ["joy", "surprise"],    text_lower, "excitement",
                                               base_weights=[0.55, 0.45], keyword_boost=0.35)
        sales["skepticism"]    = self._derive(base, ["disgust", "neutral"], text_lower, "skepticism",
                                               base_weights=[0.5, 0.5], keyword_boost=0.40)
        sales["empathy"]       = self._derive(base, ["joy", "sadness"],     text_lower, "empathy",
                                               base_weights=[0.45, 0.55], keyword_boost=0.50)
        sales["disappointment"]= self._derive(base, ["sadness", "disgust"], text_lower, "disappointment",
                                               base_weights=[0.65, 0.35], keyword_boost=0.35)
        sales["relief"]        = self._derive(base, ["joy", "neutral"],     text_lower, "relief",
                                               base_weights=[0.7, 0.3], keyword_boost=0.45)

        # ── Step 3: Combine into full 15-dim vector ───────────────────────
        full = {**base, **sales}

        # Intensity: L2 norm of non-neutral scores, aggressively scaled
        non_neutral = [v for k, v in full.items() if k != "neutral"]
        intensity   = float(np.sqrt(sum(x**2 for x in non_neutral)) / np.sqrt(len(non_neutral)))
        intensity   = round(min(intensity * 2.5, 1.0), 3)   # aggressive scaling

        # Synthetic compound
        pos      = sum(full.get(e, 0) for e in POSITIVE_EMOTIONS)
        neg      = sum(full.get(e, 0) for e in NEGATIVE_EMOTIONS)
        compound = round((pos - neg) / max(pos + neg, 1e-6), 4)

        ranked    = sorted(full.items(), key=lambda x: x[1], reverse=True)
        dominant  = ranked[0][0]
        secondary = ranked[1][0]

        return EmotionVector(
            scores=full,
            dominant=dominant,
            secondary=secondary,
            intensity=intensity,
            compound=compound,
        )

    def _derive(
        self,
        base: dict,
        source_emotions: list[str],
        text: str,
        target: str,
        base_weights: list[float],
        keyword_boost: float,
    ) -> float:
        """Derive a sales emotion score from weighted base emotions + keyword hits."""
        score = sum(
            base.get(e, 0) * w
            for e, w in zip(source_emotions, base_weights)
        )
        keywords = _KEYWORDS.get(target, [])
        hits     = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            boost = keyword_boost * min(hits / 2.0, 1.0)
            score = score + boost * (1 - score)
        return round(min(score, 1.0), 4)


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = TransformerEmotionDetector()
    tests = [
        "I cannot believe this happened again! Every single time!",
        "Oh wow, that is actually amazing! I love this deal!",
        "I need this resolved immediately, it is absolutely critical.",
        "Wait, are you sure about that? That sounds too good to be true.",
        "I understand, that must have been really frustrating for you.",
        "Finally! I am so relieved this is sorted out at last.",
        "I don't understand what you mean by that at all.",
    ]
    for t in tests:
        ev = detector.detect(t)
        print(f"\n{t[:70]}")
        for emotion, score in ev.top(4):
            bar = "█" * int(score * 20)
            print(f"  {emotion:15s} {score:.2f} {bar}")
        print(f"  dominant={ev.dominant}  intensity={ev.intensity:.2f}")
