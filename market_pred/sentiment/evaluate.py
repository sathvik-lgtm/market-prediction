"""Evaluate VADER, off-the-shelf FinBERT, and our fine-tuned FinBERT on the
Financial PhraseBank held-out test split -- a real three-way comparison for
the write-up, not just a single number in isolation.

    python -m market_pred.sentiment.evaluate
"""
from __future__ import annotations

import json
import logging

from sklearn.metrics import accuracy_score, classification_report, f1_score

from market_pred.config import get_settings
from market_pred.sentiment.baseline import vader_score
from market_pred.sentiment.dataset import LABEL_NAMES, load_phrasebank
from market_pred.sentiment.model import FinbertModel

logger = logging.getLogger(__name__)


def _vader_predict(texts: list[str]) -> list[int]:
    """Map VADER's continuous compound score to the 3-class label space using
    the +/-0.05 thresholds VADER's own documentation recommends."""
    preds = []
    for text in texts:
        s = vader_score(text)
        if s >= 0.05:
            preds.append(LABEL_NAMES.index("positive"))
        elif s <= -0.05:
            preds.append(LABEL_NAMES.index("negative"))
        else:
            preds.append(LABEL_NAMES.index("neutral"))
    return preds


def _finbert_predict(model: FinbertModel, texts: list[str]) -> list[int]:
    return [LABEL_NAMES.index(r.label) for r in model.score_batch(texts)]


def _report(name: str, y_true: list[int], y_pred: list[int]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    logger.info("%s: accuracy=%.4f  f1_macro=%.4f", name, acc, f1_macro)
    logger.info("\n%s", classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))
    return {"name": name, "accuracy": acc, "f1_macro": f1_macro}


def run() -> None:
    settings = get_settings()
    ds = load_phrasebank(
        settings.sentiment_phrasebank_repo,
        settings.sentiment_phrasebank_config,
        settings.sentiment_val_fraction,
        settings.sentiment_seed,
    )
    test = ds["test"]
    texts = list(test["sentence"])
    y_true = list(test["label"])

    results = [_report("VADER (rule-based baseline)", y_true, _vader_predict(texts))]

    pretrained = FinbertModel(settings.sentiment_finbert_pretrained_model)
    results.append(_report("ProsusAI/finbert (off-the-shelf)", y_true, _finbert_predict(pretrained, texts)))

    final_dir = settings.sentiment_model_dir / "final"
    if final_dir.exists():
        finetuned = FinbertModel(final_dir)
        results.append(_report("Our fine-tuned FinBERT", y_true, _finbert_predict(finetuned, texts)))
    else:
        logger.warning("No fine-tuned model at %s -- run train.py first to include it", final_dir)

    report_path = settings.sentiment_model_dir / "eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote comparison report to %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
