"""Feature engineering: technical indicators from price history, and a
lookahead-bias-safe join of sentiment onto trading days.
"""
from __future__ import annotations

import bisect
import sqlite3
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_pred.db.access import get_price_history, get_scored_news_for_ticker

IST = ZoneInfo("Asia/Kolkata")
NSE_CLOSE_TIME_IST = time(15, 30)

TECHNICAL_FEATURE_COLUMNS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "price_to_sma_5", "price_to_sma_10", "price_to_sma_20",
    "volume_ratio_20", "volatility_10", "volatility_20",
]

SENTIMENT_FEATURE_COLUMNS = ["mean_finbert_score", "mean_vader_score", "sentiment_article_count"]


def compute_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Technical indicators + next-day direction target for one ticker's sorted
    price history. Every feature only looks backward from its own row's date
    (rolling windows, pct_change), so there's no lookahead risk in this function.
    """
    df = prices.sort_values("date").reset_index(drop=True).copy()
    close = df["close"]

    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_20d"] = close.pct_change(20)

    df["price_to_sma_5"] = close / close.rolling(5).mean() - 1
    df["price_to_sma_10"] = close / close.rolling(10).mean() - 1
    df["price_to_sma_20"] = close / close.rolling(20).mean() - 1

    df["volume_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean()

    daily_return = close.pct_change(1)
    df["volatility_10"] = daily_return.rolling(10).std()
    df["volatility_20"] = daily_return.rolling(20).std()

    # Target: does tomorrow's close beat today's? The last row has no "tomorrow"
    # yet, so it gets NaN and is dropped downstream rather than given a fake label.
    df["next_return"] = close.pct_change(1).shift(-1)
    df["target_up"] = np.where(df["next_return"].isna(), np.nan, (df["next_return"] > 0).astype(float))

    return df


def assign_effective_trading_day(published_at_utc: str, trading_dates: list[date]) -> date | None:
    """The earliest trading day whose 15:30 IST close is at/after the headline's
    publish time -- i.e. the first day's close by which this headline was known.

    A headline published during session D, before D's close, is attributed to D
    itself (it's part of what's "known" going into predicting D+1). One published
    after D's close -- including on a weekend/holiday -- rolls forward to the next
    actual trading day in `trading_dates`, matching the README's documented rule
    (`published_date_ist` alone is not safe for this; this is the real
    implementation of the cutoff/rolling-forward logic that was deferred to
    Phase 3's feature engineering). Returns None if the headline is newer than
    every known trading day.
    """
    publish_dt = datetime.strptime(published_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    idx = bisect.bisect_left(
        trading_dates, publish_dt, key=lambda d: datetime.combine(d, NSE_CLOSE_TIME_IST, tzinfo=IST)
    )
    return trading_dates[idx] if idx < len(trading_dates) else None


def compute_sentiment_features(news: pd.DataFrame, trading_dates: list[date]) -> pd.DataFrame:
    """Aggregate scored headlines onto trading days using the lookahead-safe
    assignment above. This is deliberately NOT the same as the `daily_sentiment`
    table, which groups by the naive calendar day (`published_date_ist`) for
    display/EDA and is documented as unsafe to join directly to price targets.
    """
    if news.empty:
        return pd.DataFrame(columns=["date"] + SENTIMENT_FEATURE_COLUMNS)

    df = news.copy()
    df["date"] = df["published_at_utc"].apply(lambda ts: assign_effective_trading_day(ts, trading_dates))
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].apply(lambda d: d.isoformat())  # match prices' date string format

    return df.groupby("date").agg(
        mean_finbert_score=("finbert_score", "mean"),
        mean_vader_score=("vader_score", "mean"),
        sentiment_article_count=("finbert_score", "count"),
    ).reset_index()


def build_price_only_dataset(conn: sqlite3.Connection, tickers: list[str]) -> pd.DataFrame:
    """Pooled, full-history technical-feature dataset across all tickers.
    One row per (ticker, date) with features + target; warm-up rows (before the
    longest rolling window has enough history) and each ticker's last (targetless)
    row are dropped.
    """
    frames = []
    for ticker in tickers:
        prices = get_price_history(conn, ticker)
        if prices.empty:
            continue
        feats = compute_technical_features(prices)
        feats["ticker"] = ticker
        frames.append(feats)

    if not frames:
        return pd.DataFrame()

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.dropna(subset=TECHNICAL_FEATURE_COLUMNS + ["target_up"])
    pooled["target_up"] = pooled["target_up"].astype(int)
    return pooled.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_price_sentiment_dataset(conn: sqlite3.Connection, tickers: list[str]) -> pd.DataFrame:
    """Pooled dataset restricted to rows where lookahead-safe sentiment features
    are actually available (inner join) -- the small recent-window overlap used
    for the sentiment ablation, not the full-history price-only model.
    """
    frames = []
    for ticker in tickers:
        prices = get_price_history(conn, ticker)
        if prices.empty:
            continue
        feats = compute_technical_features(prices)
        trading_dates = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in prices["date"])

        news = get_scored_news_for_ticker(conn, ticker)
        sentiment = compute_sentiment_features(news, trading_dates)
        if sentiment.empty:
            continue

        merged = feats.merge(sentiment, on="date", how="inner")
        merged["ticker"] = ticker
        frames.append(merged)

    if not frames:
        return pd.DataFrame()

    pooled = pd.concat(frames, ignore_index=True)
    required = TECHNICAL_FEATURE_COLUMNS + SENTIMENT_FEATURE_COLUMNS + ["target_up"]
    pooled = pooled.dropna(subset=required)
    pooled["target_up"] = pooled["target_up"].astype(int)
    return pooled.sort_values(["date", "ticker"]).reset_index(drop=True)
