from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_pred.modeling.features import (
    assign_effective_trading_day,
    compute_sentiment_features,
    compute_technical_features,
)

TRADING_DATES = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 8)]  # Fri->Mon gap


def test_assign_effective_trading_day_before_close_stays_same_day():
    # 10:00 IST = 04:30 UTC, well before the 15:30 IST close.
    assert assign_effective_trading_day("2024-01-02T04:30:00Z", TRADING_DATES) == date(2024, 1, 2)


def test_assign_effective_trading_day_after_close_rolls_forward():
    # 22:00 IST = 16:30 UTC, after that day's 15:30 IST close.
    assert assign_effective_trading_day("2024-01-02T16:30:00Z", TRADING_DATES) == date(2024, 1, 3)


def test_assign_effective_trading_day_exactly_at_close_stays_same_day():
    # 15:30 IST = 10:00 UTC exactly -- known "by" the close, so counts for that day.
    assert assign_effective_trading_day("2024-01-02T10:00:00Z", TRADING_DATES) == date(2024, 1, 2)


def test_assign_effective_trading_day_weekend_rolls_to_next_trading_day():
    # Jan 4-7 2024 aren't in TRADING_DATES (weekend + a gap) -- should roll to Jan 8.
    assert assign_effective_trading_day("2024-01-06T04:30:00Z", TRADING_DATES) == date(2024, 1, 8)


def test_assign_effective_trading_day_newer_than_all_known_days_returns_none():
    assert assign_effective_trading_day("2024-06-01T04:30:00Z", TRADING_DATES) is None


def test_compute_sentiment_features_uses_effective_day_not_raw_publish_date():
    news = pd.DataFrame([
        {"published_at_utc": "2024-01-02T16:30:00Z", "finbert_score": 1.0, "vader_score": 0.5},  # -> Jan 3
        {"published_at_utc": "2024-01-03T04:00:00Z", "finbert_score": -0.5, "vader_score": -0.2},  # -> Jan 3
    ])
    result = compute_sentiment_features(news, TRADING_DATES)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["date"] == "2024-01-03"
    assert row["sentiment_article_count"] == 2
    assert row["mean_finbert_score"] == pytest.approx(0.25)


def test_compute_technical_features_last_row_has_no_target():
    prices = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "open": [100.0, 101.0, 99.0],
        "high": [101.0, 102.0, 100.0],
        "low": [99.0, 100.0, 98.0],
        "close": [100.0, 102.0, 99.0],
        "volume": [1000, 1100, 900],
    })
    feats = compute_technical_features(prices)

    assert np.isnan(feats.loc[0, "return_1d"])  # no prior day to compute a return from
    assert feats.loc[1, "return_1d"] == pytest.approx(0.02)  # (102-100)/100
    # Row 0 -> next close 102 > 100 -> up. Row 1 -> next close 99 < 102 -> down.
    assert feats.loc[0, "target_up"] == 1.0
    assert feats.loc[1, "target_up"] == 0.0
    assert np.isnan(feats.loc[2, "target_up"])  # last row: no "tomorrow" yet
