"""Export the live SQLite DB's prices/news/daily_sentiment tables to small
committed CSV snapshots so a fresh clone can be demoed without an API key or
a ~20-minute FinBERT fine-tuning run. Re-run occasionally and commit the
refreshed CSVs.

    python scripts/export_seed_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from market_pred.config import REPO_ROOT, get_settings
from market_pred.db.connection import get_connection

SEED_DIR = REPO_ROOT / "data" / "seed"


def main() -> None:
    settings = get_settings()
    SEED_DIR.mkdir(parents=True, exist_ok=True)

    with get_connection(settings.db_path) as conn:
        prices = pd.read_sql_query("SELECT * FROM prices ORDER BY ticker, date", conn)
        news = pd.read_sql_query("SELECT * FROM news ORDER BY ticker, published_at_utc", conn)
        daily_sentiment = pd.read_sql_query(
            "SELECT * FROM daily_sentiment ORDER BY ticker, date", conn
        )

    prices.to_csv(SEED_DIR / "sample_prices.csv", index=False)
    news.to_csv(SEED_DIR / "sample_news.csv", index=False)
    daily_sentiment.to_csv(SEED_DIR / "sample_daily_sentiment.csv", index=False)
    print(
        f"Wrote {len(prices)} price rows, {len(news)} news rows, and "
        f"{len(daily_sentiment)} daily_sentiment rows to {SEED_DIR}"
    )


if __name__ == "__main__":
    main()
