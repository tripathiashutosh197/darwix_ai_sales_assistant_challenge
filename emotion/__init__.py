"""
emotion/__init__.py
--------------------
Factory — returns either VADER or Transformer detector.
Both return the same EmotionVector with all 15 emotion dimensions
so they are fully interchangeable in the pipeline.

Set EMOTION_MODEL in .env:
  EMOTION_MODEL=transformer   (default, most accurate)
  EMOTION_MODEL=vader         (faster, offline, good on short texts)
"""

import os


def get_detector(model_type: str = None):
    """
    Return the configured emotion detector.

    model_type: "vader" | "transformer"
                Defaults to EMOTION_MODEL env var, falls back to "transformer".
    """
    model_type = model_type or os.getenv("EMOTION_MODEL", "transformer")

    if model_type == "vader":
        from emotion.vader_detector import VADEREmotionDetector
        return VADEREmotionDetector()

    elif model_type == "transformer":
        from emotion.transformer_detector import TransformerEmotionDetector
        return TransformerEmotionDetector()

    else:
        raise ValueError(
            f"Unknown emotion model: {model_type!r}. Use 'vader' or 'transformer'."
        )
