"""
emotion/vader_detector.py
--------------------------
Fast offline emotion detection using VADER sentiment analysis.
Returns emotion label + intensity score (0.0 – 1.0).

Emotions detected:
  joy, anger, sadness, fear, surprise, neutral

No model training needed — VADER is rule-based and works out of the box.
"""

from dataclasses import dataclass
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class EmotionResult:
    label: str          # primary emotion name
    intensity: float    # 0.0 (mild) → 1.0 (extreme)
    scores: dict        # raw per-emotion scores
    compound: float     # VADER compound score -1 to +1


class VADEREmotionDetector:
    """
    Rule-based emotion detector built on VADER compound + heuristic lexicon.

    Thresholds (tunable):
        compound ≥  0.6  → joy       (high positive)
        compound ≥  0.2  → positive  (mild positive)
        compound ≤ -0.6  → anger/sadness (high negative)
        compound ≤ -0.2  → concerned (mild negative)
        else             → neutral

    Intensity = abs(compound) mapped to [0.3, 1.0] so we never
    apply zero modulation on even mild emotions.
    """

    # Extra lexicon hints for emotions VADER's compound misses
    FEAR_KEYWORDS    = {"afraid","scared","terrified","horror","panic","dread","anxious","nervous","worried","fear","frightened"}
    SURPRISE_KEYWORDS= {"wow","omg","unbelievable","incredible","amazing","shocking","unexpected","whoa","seriously","really","wait"}
    ANGER_KEYWORDS   = {"hate","angry","furious","outraged","disgusting","ridiculous","stupid","idiot","awful","terrible","worst"}
    SADNESS_KEYWORDS = {"sad","depressed","miserable","heartbroken","crying","tears","lonely","hopeless","devastated","grief","sorry"}

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def detect(self, text: str) -> EmotionResult:
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]
        words = set(text.lower().split())

        # Check keyword overrides first for nuanced emotions
        fear_hit     = len(words & self.FEAR_KEYWORDS)
        surprise_hit = len(words & self.SURPRISE_KEYWORDS)
        anger_hit    = len(words & self.ANGER_KEYWORDS)
        sadness_hit  = len(words & self.SADNESS_KEYWORDS)

        if fear_hit >= 1 and compound <= 0.1:
            label = "fear"
            intensity = self._scale(abs(compound), boost=0.2 * fear_hit)

        elif surprise_hit >= 1 and compound >= 0.0:
            label = "surprise"
            intensity = self._scale(abs(compound), boost=0.15 * surprise_hit)

        elif anger_hit >= 1 and compound <= 0.1:
            label = "anger"
            intensity = self._scale(abs(compound), boost=0.2 * anger_hit)

        elif sadness_hit >= 1 and compound <= 0.1:
            label = "sadness"
            intensity = self._scale(abs(compound), boost=0.2 * sadness_hit)

        # Fall back to compound thresholds
        elif compound >= 0.6:
            label = "joy"
            intensity = self._scale(compound)
        elif compound >= 0.2:
            label = "joy"
            intensity = self._scale(compound)
        elif compound <= -0.6:
            label = "anger"
            intensity = self._scale(abs(compound))
        elif compound <= -0.2:
            label = "sadness"
            intensity = self._scale(abs(compound))
        else:
            label = "neutral"
            intensity = 0.0

        return EmotionResult(
            label=label,
            intensity=round(min(intensity, 1.0), 3),
            scores=scores,
            compound=compound,
        )

    @staticmethod
    def _scale(raw: float, boost: float = 0.0) -> float:
        """Map raw [0,1] to [0.3, 1.0] so even mild emotions get some modulation."""
        base = 0.3 + raw * 0.7
        return min(base + boost, 1.0)


# ── quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = VADEREmotionDetector()
    tests = [
        "This is the best day of my life! I am absolutely thrilled!",
        "I am so frustrated. This keeps failing over and over again.",
        "Please hold while I transfer your call.",
        "I am terrified about what might happen next.",
        "Wait, you are serious? That is completely unbelievable!",
        "I feel so alone and nobody cares.",
    ]
    for t in tests:
        r = detector.detect(t)
        print(f"[{r.label:10s}] intensity={r.intensity:.2f}  compound={r.compound:+.2f}  | {t[:60]}")
