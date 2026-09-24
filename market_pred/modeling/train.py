"""Train and walk-forward-evaluate price-direction classifiers.

    python -m market_pred.modeling.train

Runs two things:
1. The primary evaluation: price/technical features only, walk-forward
   validated across the full price history (a statistically meaningful
   sample size, unlike anything involving sentiment right now).
2. A separate sentiment ablation: price-only vs price+sentiment features,
   evaluated on the same small recent overlap window where sentiment
   actually exists -- a low-confidence, small-sample comparison, reported
   honestly as such rather than folded into the headline result.

Writes a JSON report to `modeling.model_dir/report.json` and saves an
XGBoost model fit on all available price history for later (Phase 4) use.
"""
from __future__ import annotations

import json
import logging

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from market_pred.config import Settings, get_settings
from market_pred.db.connection import get_connection
from market_pred.modeling.baselines import mean_daily_return, naive_always_up_accuracy, strategy_mean_return
from market_pred.modeling.features import (
    SENTIMENT_FEATURE_COLUMNS,
    TECHNICAL_FEATURE_COLUMNS,
    build_price_only_dataset,
    build_price_sentiment_dataset,
)
from market_pred.modeling.predict import MODEL_FILENAME
from market_pred.modeling.validation import simple_holdout_split, split_fold, yearly_expanding_folds

logger = logging.getLogger(__name__)

SUMMARY_KEYS = [
    "accuracy", "precision", "recall", "f1",
    "naive_always_up_accuracy", "buy_and_hold_mean_return", "strategy_mean_return",
]


def _make_models(seed: int) -> dict:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=seed, eval_metric="logloss",
        ),
    }


def _evaluate_fold(model, train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str]) -> dict:
    X_train, y_train = train[feature_columns], train["target_up"]
    X_test, y_test = test[feature_columns], test["target_up"]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "naive_always_up_accuracy": naive_always_up_accuracy(y_test),
        "buy_and_hold_mean_return": mean_daily_return(test["next_return"]),
        "strategy_mean_return": strategy_mean_return(y_pred, test["next_return"]),
    }


def _summarize(fold_results: list[dict]) -> dict:
    if not fold_results:
        return {}
    return {k: float(pd.Series([r[k] for r in fold_results]).mean()) for k in SUMMARY_KEYS}


def run_price_only_walk_forward(conn, settings: Settings) -> dict:
    df = build_price_only_dataset(conn, [t.symbol for t in settings.tickers])
    if df.empty:
        raise RuntimeError("No price data available -- run `prices` ingestion first")

    folds = yearly_expanding_folds(df["date"], min_train_years=settings.modeling_min_train_years)
    logger.info("Price-only walk-forward: %d fold(s) over %d row(s)", len(folds), len(df))

    models = _make_models(settings.modeling_seed)
    fold_results = {name: [] for name in models}

    for fold in folds:
        train, test = split_fold(df, fold)
        if train.empty or test.empty:
            continue
        for name, model in models.items():
            metrics = _evaluate_fold(model, train, test, TECHNICAL_FEATURE_COLUMNS)
            metrics["fold"] = fold.test_start[:4]
            fold_results[name].append(metrics)
            logger.info(
                "%s | %s: acc=%.4f (naive=%.4f) strategy_ret=%.5f (buy_hold=%.5f)",
                name, metrics["fold"], metrics["accuracy"], metrics["naive_always_up_accuracy"],
                metrics["strategy_mean_return"], metrics["buy_and_hold_mean_return"],
            )

    return {name: {"folds": folds, "summary": _summarize(folds)} for name, folds in fold_results.items()}


def run_sentiment_ablation(conn, settings: Settings) -> dict:
    df = build_price_sentiment_dataset(conn, [t.symbol for t in settings.tickers])
    if df.empty or df["date"].nunique() < 4:
        logger.warning("Not enough overlapping price+sentiment data for an ablation -- skipping")
        return {}

    train, test = simple_holdout_split(df, test_fraction=settings.modeling_sentiment_test_fraction)
    if train.empty or test.empty:
        logger.warning("Sentiment ablation split produced an empty train/test set -- skipping")
        return {}

    logger.info(
        "Sentiment ablation: %d train / %d test row(s) over %d unique date(s) -- "
        "small-sample, low-confidence comparison, see README",
        len(train), len(test), df["date"].nunique(),
    )

    results = {}
    for feature_set_name, columns in [
        ("price_only", TECHNICAL_FEATURE_COLUMNS),
        ("price_plus_sentiment", TECHNICAL_FEATURE_COLUMNS + SENTIMENT_FEATURE_COLUMNS),
    ]:
        model = _make_models(settings.modeling_seed)["logistic_regression"]
        metrics = _evaluate_fold(model, train, test, columns)
        results[feature_set_name] = metrics
        logger.info(
            "Ablation | %s: acc=%.4f strategy_ret=%.5f",
            feature_set_name, metrics["accuracy"], metrics["strategy_mean_return"],
        )

    return results


def _fit_final_model(conn, settings: Settings) -> XGBClassifier:
    df = build_price_only_dataset(conn, [t.symbol for t in settings.tickers])
    model = _make_models(settings.modeling_seed)["xgboost"]
    model.fit(df[TECHNICAL_FEATURE_COLUMNS], df["target_up"])
    return model


def run() -> None:
    settings = get_settings()
    with get_connection(settings.db_path) as conn:
        price_only_results = run_price_only_walk_forward(conn, settings)
        ablation_results = run_sentiment_ablation(conn, settings)
        final_model = _fit_final_model(conn, settings)

    settings.modeling_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, settings.modeling_model_dir / MODEL_FILENAME)

    report = {"price_only_walk_forward": price_only_results, "sentiment_ablation": ablation_results}
    report_path = settings.modeling_model_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float))
    logger.info("Saved final model and wrote report to %s", settings.modeling_model_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
