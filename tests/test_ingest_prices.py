from datetime import date

import pandas as pd

from market_pred.config import Settings, TickerInfo
from market_pred.db.access import get_price_history
from market_pred.db.connection import get_connection
from market_pred.ingest import prices as prices_ingest


def make_raw_history_df():
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame({
        "Open": [100.0, 101.0, None],
        "High": [105.0, 106.0, 107.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [104.0, 105.0, 106.0],
        "Volume": [1000, 1100, 1200],
    }, index=index)


def test_clean_price_df_normalizes_columns_and_drops_nan_rows():
    cleaned = prices_ingest.clean_price_df(make_raw_history_df())

    assert list(cleaned.columns) == ["date", "open", "high", "low", "close", "volume", "fetched_at"]
    assert len(cleaned) == 2  # the row with a NaN open was dropped
    assert cleaned.iloc[0]["date"] == "2024-01-01"
    assert cleaned.iloc[0]["close"] == 104.0


def test_clean_price_df_handles_empty_input():
    assert prices_ingest.clean_price_df(pd.DataFrame()).empty


def _fake_settings(db_path):
    return Settings(
        tickers=[TickerInfo(symbol="TEST.NS", company="Test Co")],
        db_path=db_path,
        price_history_start=date(2024, 1, 1),
        price_backfill_buffer_days=5,
        news_lookback_days=29,
        news_page_size=100,
        news_max_pages_per_ticker=3,
        news_max_requests_per_run=90,
    )


def test_run_upserts_fetched_prices(tmp_path, monkeypatch):
    settings = _fake_settings(tmp_path / "test.db")
    monkeypatch.setattr(prices_ingest, "get_settings", lambda: settings)
    monkeypatch.setattr(
        prices_ingest, "fetch_price_history", lambda ticker, start, end: make_raw_history_df()
    )

    prices_ingest.run()

    with get_connection(settings.db_path) as conn:
        history = get_price_history(conn, "TEST.NS")
    assert len(history) == 2


def test_run_skips_ticker_on_empty_response(tmp_path, monkeypatch):
    settings = _fake_settings(tmp_path / "test.db")
    monkeypatch.setattr(prices_ingest, "get_settings", lambda: settings)
    monkeypatch.setattr(prices_ingest, "fetch_price_history", lambda ticker, start, end: pd.DataFrame())

    prices_ingest.run()  # should not raise even though yfinance returned nothing

    with get_connection(settings.db_path) as conn:
        history = get_price_history(conn, "TEST.NS")
    assert history.empty
