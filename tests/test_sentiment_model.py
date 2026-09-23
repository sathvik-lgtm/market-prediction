"""Tests for FinbertModel's own aggregation logic (label/score formula),
decoupled from real model downloads via a fake tokenizer/model."""
import pytest
import torch

from market_pred.sentiment import model as model_mod


class FakeTokenizer:
    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="pt"):
        return {"input_ids": torch.zeros((len(texts), 1), dtype=torch.long)}


class FakeConfig:
    id2label = {0: "negative", 1: "neutral", 2: "positive"}


class FakeModel:
    config = FakeConfig()

    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        # Strongly "positive" logits for every row -> deterministic expected output.
        logits = torch.tensor([[0.0, 0.0, 5.0]] * batch_size)
        return type("Output", (), {"logits": logits})()


def test_score_batch_picks_argmax_label_and_signed_score(monkeypatch):
    monkeypatch.setattr(model_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    monkeypatch.setattr(
        model_mod.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: FakeModel()
    )

    m = model_mod.FinbertModel("fake-path")
    results = m.score_batch(["some headline", "another headline"], batch_size=16)

    # softmax([0, 0, 5]) -> [~0.00665, ~0.00665, ~0.98670]
    assert len(results) == 2
    for r in results:
        assert r.label == "positive"
        assert r.score == pytest.approx(0.98005, abs=1e-4)  # p_positive - p_negative
        assert r.confidence == pytest.approx(0.98670, abs=1e-4)


def test_score_batch_respects_batch_size_chunking(monkeypatch):
    calls = []

    class CountingTokenizer(FakeTokenizer):
        def __call__(self, texts, **kwargs):
            calls.append(len(texts))
            return super().__call__(texts, **kwargs)

    monkeypatch.setattr(model_mod.AutoTokenizer, "from_pretrained", lambda *a, **k: CountingTokenizer())
    monkeypatch.setattr(
        model_mod.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: FakeModel()
    )

    m = model_mod.FinbertModel("fake-path")
    m.score_batch(["a", "b", "c", "d", "e"], batch_size=2)

    assert calls == [2, 2, 1]
