"""VADER sentiment baseline -- a quick, rule-based floor to compare FinBERT against."""
from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def vader_score(text: str) -> float:
    """Compound sentiment score in [-1, 1]. Empty/whitespace text scores 0.0."""
    if not text or not text.strip():
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]
