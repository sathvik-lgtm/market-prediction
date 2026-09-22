"""Price/volume ingestion via yfinance."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from market_pred.config import get_settings
from market_pred.db.access import get_last_price_date, upsert_prices
from market_pred.db.connection import get_connection
from market_pred.db.schema import init_db

logger = logging.getLogger(__name__)


def fetch_price_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily OHLCV bars for one ticker.

    `end` is exclusive per yfinance convention, so callers wanting to include
    `end` itself should pass `end + 1 day`.
    """
    t = yf.Ticker(ticker)
    return t.history(start=start.isoformat(), end=end.isoformat(), interval="1d", auto_adjust=True)


def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw yfinance history DataFrame into the `prices` table's row shape."""
    columns = ["date", "open", "high", "low", "close", "volume", "fetched_at"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    before = len(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d row(s) with missing OHLCV data", dropped)

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = pd.DataFrame({
        # yfinance's index is already tz-aware in the exchange's local timezone (IST for
        # .NS tickers) for daily bars -- take the date directly, don't convert through UTC.
        "date": df.index.strftime("%Y-%m-%d"),
        "open": df["Open"].astype(float),
        "high": df["High"].astype(float),
        "low": df["Low"].astype(float),
        "close": df["Close"].astype(float),
        "volume": df["Volume"].astype(int),
    })
    out["fetched_at"] = fetched_at
    return out.reset_index(drop=True)


def run(tickers: list[str] | None = None, full_refresh: bool = False) -> None:
    settings = get_settings()
    ticker_symbols = tickers or [t.symbol for t in settings.tickers]

    with get_connection(settings.db_path) as conn:
        init_db(conn)
        today = date.today()

        for ticker in ticker_symbols:
            try:
                if full_refresh:
                    start = settings.price_history_start
                else:
                    last_date = get_last_price_date(conn, ticker)
                    start = (
                        settings.price_history_start
                        if last_date is None
                        else last_date - timedelta(days=settings.price_backfill_buffer_days)
                    )

                end = today + timedelta(days=1)  # yfinance's `end` is exclusive
                raw = fetch_price_history(ticker, start, end)
                if raw.empty:
                    logger.warning("No price data returned for %s (start=%s) -- skipping", ticker, start)
                    continue

                cleaned = clean_price_df(raw)
                n = upsert_prices(conn, cleaned, ticker)
                logger.info("Upserted %d price row(s) for %s", n, ticker)
            except Exception:
                logger.exception("Failed to fetch/upsert prices for %s -- skipping", ticker)
                continue


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
