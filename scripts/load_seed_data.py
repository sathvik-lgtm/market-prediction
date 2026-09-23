"""Load the committed CSV seed snapshots into a fresh local SQLite DB, so a
fresh clone is demoable without needing a NewsAPI key or a ~20-minute FinBERT
fine-tuning run.

    python scripts/load_seed_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from market_pred.config import REPO_ROOT, get_settings
from market_pred.db.access import (
    INSERT_NEWS_SQL,
    UPDATE_NEWS_SENTIMENT_SQL,
    UPSERT_DAILY_SENTIMENT_SQL,
    UPSERT_PRICE_SQL,
)
from market_pred.db.connection import get_connection
from market_pred.db.schema import init_db

SEED_DIR = REPO_ROOT / "data" / "seed"

SENTIMENT_COLUMNS = ["vader_score", "finbert_label", "finbert_score", "finbert_confidence", "sentiment_scored_at"]


def main() -> None:
    settings = get_settings()

    prices = pd.read_csv(SEED_DIR / "sample_prices.csv", dtype={"date": str})
    news = pd.read_csv(SEED_DIR / "sample_news.csv")
    news = news.where(pd.notnull(news), None)
    daily_sentiment = pd.read_csv(SEED_DIR / "sample_daily_sentiment.csv", dtype={"date": str})

    n_sentiment_rows = 0
    with get_connection(settings.db_path) as conn:
        init_db(conn)
        if not prices.empty:
            conn.executemany(UPSERT_PRICE_SQL, prices.to_dict(orient="records"))
        if not news.empty:
            conn.executemany(INSERT_NEWS_SQL, news.to_dict(orient="records"))

            # The CSV's `id` values won't match this fresh DB's own autoincrement
            # ids, so resolve each row's real id via its natural (ticker, url)
            # key before writing its sentiment scores back.
            scored = news[news["sentiment_scored_at"].notna()].drop(columns=["id"])
            if not scored.empty:
                id_lookup = pd.read_sql_query("SELECT id, ticker, url FROM news", conn)
                merged = scored.merge(id_lookup, on=["ticker", "url"])
                update_rows = merged[["id"] + SENTIMENT_COLUMNS].to_dict(orient="records")
                conn.executemany(UPDATE_NEWS_SENTIMENT_SQL, update_rows)
                n_sentiment_rows = len(update_rows)
        if not daily_sentiment.empty:
            conn.executemany(UPSERT_DAILY_SENTIMENT_SQL, daily_sentiment.to_dict(orient="records"))

    print(
        f"Loaded {len(prices)} price rows, {len(news)} news rows ({n_sentiment_rows} with "
        f"sentiment scores), and {len(daily_sentiment)} daily_sentiment rows into {settings.db_path}"
    )


if __name__ == "__main__":
    main()
