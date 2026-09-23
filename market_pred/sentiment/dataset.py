"""Load and split the Financial PhraseBank dataset for fine-tuning."""
from __future__ import annotations

from datasets import DatasetDict, load_dataset

LABEL_NAMES = ["negative", "neutral", "positive"]
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}


def load_phrasebank(repo: str, config: str, val_fraction: float, seed: int) -> DatasetDict:
    """Returns a DatasetDict with 'train', 'validation', 'test' splits.

    The repo's own 'test' split stays fully held out for final evaluation;
    'validation' is carved out of 'train' (stratified) for monitoring during
    fine-tuning.
    """
    raw = load_dataset(repo, config)
    # stratify_by_column requires a ClassLabel feature; this repo's `label` column
    # is a plain int64 Value, so it needs casting first.
    train = raw["train"].class_encode_column("label")
    split = train.train_test_split(test_size=val_fraction, seed=seed, stratify_by_column="label")
    return DatasetDict(train=split["train"], validation=split["test"], test=raw["test"])
