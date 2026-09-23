"""FinBERT inference: load a local fine-tuned checkpoint or a HF Hub model id
and score text for sentiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class SentimentResult:
    label: str
    score: float  # signed: p_positive - p_negative, range [-1, 1]
    confidence: float  # probability of the predicted label


class FinbertModel:
    """Loads any model whose config declares id2label for negative/neutral/positive
    (case-insensitive) -- our own fine-tuned checkpoint or an off-the-shelf hub
    model like ProsusAI/finbert. Label order doesn't matter since it's read from
    the model's own config rather than assumed.
    """

    def __init__(self, model_path: str | Path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def score_batch(self, texts: list[str], batch_size: int = 16, max_length: int = 64) -> list[SentimentResult]:
        results: list[SentimentResult] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            for row in probs:
                per_label = {self.id2label[j]: row[j].item() for j in range(row.shape[0])}
                label = max(per_label, key=per_label.get)
                score = per_label.get("positive", 0.0) - per_label.get("negative", 0.0)
                results.append(SentimentResult(label=label, score=score, confidence=per_label[label]))
        return results
