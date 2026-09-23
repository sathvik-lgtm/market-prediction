"""Fine-tune a FinBERT base model on the Financial PhraseBank dataset.

    python -m market_pred.sentiment.train

Saves the fine-tuned model to `sentiment.model_dir/final` and prints held-out
test-set metrics (accuracy, macro F1).
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from market_pred.config import get_settings
from market_pred.sentiment.dataset import ID2LABEL, LABEL2ID, load_phrasebank

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def _load_base_model_and_tokenizer(base_model: str):
    # yiyanghkust/finbert-pretrain's config.json predates the `model_type` field
    # that transformers' Auto* classes now require to resolve the architecture,
    # so it's loaded directly as BERT rather than via AutoModel/AutoTokenizer.
    # Once fine-tuned and saved, our own checkpoint gets a complete, modern
    # config.json -- this workaround is only needed for this one upstream repo.
    config = BertConfig.from_pretrained(
        base_model, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID
    )
    model = BertForSequenceClassification.from_pretrained(base_model, config=config)
    tokenizer = BertTokenizerFast.from_pretrained(base_model)
    return model, tokenizer


def run() -> None:
    settings = get_settings()
    set_seed(settings.sentiment_seed)

    ds = load_phrasebank(
        settings.sentiment_phrasebank_repo,
        settings.sentiment_phrasebank_config,
        settings.sentiment_val_fraction,
        settings.sentiment_seed,
    )
    logger.info(
        "Loaded Financial PhraseBank: train=%d val=%d test=%d",
        len(ds["train"]), len(ds["validation"]), len(ds["test"]),
    )

    model, tokenizer = _load_base_model_and_tokenizer(settings.sentiment_finbert_base_model)

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=settings.sentiment_max_seq_length)

    drop_cols = [c for c in ds["train"].column_names if c not in ("label",)]
    tokenized = ds.map(tokenize, batched=True, remove_columns=drop_cols)
    tokenized = tokenized.rename_column("label", "labels")

    args = TrainingArguments(
        output_dir=str(settings.sentiment_model_dir / "checkpoints"),
        per_device_train_batch_size=settings.sentiment_train_batch_size,
        per_device_eval_batch_size=settings.sentiment_eval_batch_size,
        num_train_epochs=settings.sentiment_num_epochs,
        learning_rate=settings.sentiment_learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        report_to="none",
        seed=settings.sentiment_seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    logger.info("Held-out test metrics: %s", test_metrics)

    final_dir = settings.sentiment_model_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved fine-tuned model to %s", final_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
