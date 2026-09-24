import json

import joblib
import numpy as np
import pandas as pd
import pytest

from market_pred.db.access import upsert_prices
from market_pred.modeling.features import TECHNICAL_FEATURE_COLUMNS
from market_pred.modeling.predict import (
    MODEL_FILENAME,
    ModelNotFoundError,
    get_latest_features,
    load_final_model,
    load_report_summary,
    predict_direction,
)
from tests.conftest import make_settings


def make_synthetic_prices(n_days: int, start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    price = 100.0
    for i, d in enumerate(dates):
        price += 0.5
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "open": price, "high": price + 1, "low": price - 1, "close": price,
            "volume": 1000 + i, "fetched_at": "2024-01-01T00:00:00Z",
        })
    return pd.DataFrame(rows)


# --- load_final_model ---

def test_load_final_model_raises_when_missing(tmp_path):
    settings = make_settings(tmp_path / "test.db", modeling_model_dir=tmp_path / "nope")
    with pytest.raises(ModelNotFoundError):
        load_final_model(settings)


def test_load_final_model_loads_joblib_file(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    joblib.dump({"fake": "model"}, model_dir / MODEL_FILENAME)
    settings = make_settings(tmp_path / "test.db", modeling_model_dir=model_dir)
    assert load_final_model(settings) == {"fake": "model"}


# --- get_latest_features ---

def test_get_latest_features_none_when_no_price_history(db_conn):
    assert get_latest_features(db_conn, "TEST.NS") is None


def test_get_latest_features_none_when_insufficient_history(db_conn):
    upsert_prices(db_conn, make_synthetic_prices(5), "TEST.NS")  # < 21 rows needed for return_20d
    assert get_latest_features(db_conn, "TEST.NS") is None


def test_get_latest_features_returns_last_row_when_enough_history(db_conn):
    prices = make_synthetic_prices(30)
    upsert_prices(db_conn, prices, "TEST.NS")

    row = get_latest_features(db_conn, "TEST.NS")

    assert row is not None
    assert row["date"] == prices["date"].iloc[-1]
    assert not row[TECHNICAL_FEATURE_COLUMNS].isna().any()


# --- predict_direction ---

class FakeModel:
    classes_ = np.array([0, 1])

    def __init__(self, proba):
        self._proba = proba
        self.received_columns = None

    def predict_proba(self, X):
        self.received_columns = list(X.columns)
        return np.array([self._proba])


def _make_shuffled_row() -> pd.Series:
    # Deliberately reversed column order plus extra non-feature columns --
    # exactly what a real row from get_latest_features carries (a full row of
    # compute_technical_features's output, not just the 10 feature columns).
    # Regression test for the column-order ValueError found against the real
    # persisted XGBoost model during planning.
    row = {"ticker": "TEST.NS", "date": "2024-01-05"}
    row.update({c: 0.01 for c in reversed(TECHNICAL_FEATURE_COLUMNS)})
    return pd.Series(row)


def test_predict_direction_reindexes_columns_to_fitted_order():
    model = FakeModel([0.3, 0.7])
    predict_direction(model, _make_shuffled_row())
    assert model.received_columns == TECHNICAL_FEATURE_COLUMNS


def test_predict_direction_up_case():
    label, confidence = predict_direction(FakeModel([0.3, 0.7]), _make_shuffled_row())
    assert label == "Up"
    assert confidence == pytest.approx(0.7)


def test_predict_direction_down_case():
    label, confidence = predict_direction(FakeModel([0.8, 0.2]), _make_shuffled_row())
    assert label == "Down"
    assert confidence == pytest.approx(0.8)


# --- load_report_summary ---

def test_load_report_summary_none_when_missing(tmp_path):
    settings = make_settings(tmp_path / "test.db", modeling_model_dir=tmp_path / "nope")
    assert load_report_summary(settings) is None


def test_load_report_summary_reads_xgboost_summary(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    report = {"price_only_walk_forward": {"xgboost": {"summary": {"accuracy": 0.512}}}}
    (model_dir / "report.json").write_text(json.dumps(report))
    settings = make_settings(tmp_path / "test.db", modeling_model_dir=model_dir)

    assert load_report_summary(settings) == {"accuracy": 0.512}
