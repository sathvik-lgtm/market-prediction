"""Load the persisted price-only direction model and predict on the latest
available technical features for one ticker. Kept separate from
modeling/features.py (which builds full historical training datasets) --
this module serves a single live row for the dashboard, a different
responsibility even though both call compute_technical_features().
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from xgboost import XGBClassifier

from market_pred.config import Settings
from market_pred.db.access import get_price_history
from market_pred.modeling.features import TECHNICAL_FEATURE_COLUMNS, compute_technical_features

MODEL_FILENAME = "final_xgboost.joblib"


class ModelNotFoundError(RuntimeError):
    """Raised when the persisted price-direction model doesn't exist yet --
    caller should show a message pointing at `train-model`."""


def load_final_model(settings: Settings) -> XGBClassifier:
    path = settings.modeling_model_dir / MODEL_FILENAME
    if not path.exists():
        raise ModelNotFoundError(
            f"No trained model found at {path}. Run "
            "`python -m market_pred.pipeline train-model` first."
        )
    return joblib.load(path)


def get_latest_features(conn: sqlite3.Connection, ticker: str) -> pd.Series | None:
    """The most recent trading day's technical-feature row for `ticker` -- the
    row a live 'predict the next session' call should use. None if there's no
    price history yet, or if the last row's rolling-window features aren't
    fully populated yet (insufficient trailing history).
    """
    prices = get_price_history(conn, ticker)
    if prices.empty:
        return None
    feats = compute_technical_features(prices)
    last = feats.iloc[-1]
    if last[TECHNICAL_FEATURE_COLUMNS].isna().any():
        return None
    return last


def predict_direction(model: XGBClassifier, features_row: pd.Series) -> tuple[str, float]:
    """Predict next-session direction from one row of technical features (as
    returned by get_latest_features). Returns (label, confidence) where label
    is "Up"/"Down" and confidence is the probability of the predicted label.

    Columns are explicitly reindexed to TECHNICAL_FEATURE_COLUMNS immediately
    before predict_proba(): XGBoost's sklearn API validates the incoming
    DataFrame's column order against feature_names_in_ and raises ValueError
    on a mismatch -- it does not realign by name. features_row may carry extra
    columns (ticker, date, ...) and/or a different column order than the
    model was fit with, so never pass it through as-is.
    """
    X = pd.DataFrame([features_row])[TECHNICAL_FEATURE_COLUMNS]
    proba = model.predict_proba(X)[0]
    idx = int(np.argmax(proba))
    label = "Up" if int(model.classes_[idx]) == 1 else "Down"
    return label, float(proba[idx])


@lru_cache(maxsize=1)
def _nse_calendar():
    return mcal.get_calendar("NSE")


def next_trading_session(after: date) -> date | None:
    """The next actual NSE trading day strictly after `after`, via NSE's real
    holiday calendar -- not weekend-only inference, so a holiday like Diwali
    or Republic Day isn't mistaken for the next open session. Display-only:
    feature engineering (`assign_effective_trading_day`) and walk-forward
    fold boundaries never call this -- they already derive trading days from
    the actual dates present in price history, not a calendar guess. None in
    the practically-impossible case the calendar has no sessions in the next
    two weeks.
    """
    schedule = _nse_calendar().valid_days(start_date=after + timedelta(days=1), end_date=after + timedelta(days=14))
    return schedule[0].date() if len(schedule) else None


def load_report_summary(settings: Settings) -> dict | None:
    """The xgboost walk-forward summary (accuracy, naive_always_up_accuracy,
    etc.) from the persisted training report, for a caption next to a live
    prediction. None if report.json doesn't exist -- don't crash the
    prediction panel over a missing caption.
    """
    path = settings.modeling_model_dir / "report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    return report.get("price_only_walk_forward", {}).get("xgboost", {}).get("summary")


def load_report_generated_at(settings: Settings) -> str | None:
    """UTC timestamp (ISO 8601) of the last successful `train-model` run, for
    a "Last updated" indicator so a viewer can tell fresh data from stale.
    None if report.json doesn't exist yet, or predates this field.
    """
    path = settings.modeling_model_dir / "report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("generated_at")
