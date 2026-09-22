"""Load the committed CSV seed snapshots into a fresh local SQLite DB, so a
fresh clone is demoable without needing a NewsAPI key.

    python scripts/load_seed_data.py
"""
import pandas as pd

from market_pred.config import REPO_ROOT, get_settings
from market_pred.db.access import INSERT_NEWS_SQL, UPSERT_PRICE_SQL
from market_pred.db.connection import get_connection
from market_pred.db.schema import init_db

SEED_DIR = REPO_ROOT / "data" / "seed"


def main() -> None:
    settings = get_settings()

    prices = pd.read_csv(SEED_DIR / "sample_prices.csv", dtype={"date": str})
    news = pd.read_csv(SEED_DIR / "sample_news.csv")
    news = news.where(pd.notnull(news), None)

    with get_connection(settings.db_path) as conn:
        init_db(conn)
        if not prices.empty:
            conn.executemany(UPSERT_PRICE_SQL, prices.to_dict(orient="records"))
        if not news.empty:
            conn.executemany(INSERT_NEWS_SQL, news.to_dict(orient="records"))

    print(f"Loaded {len(prices)} price rows and {len(news)} news rows into {settings.db_path}")


if __name__ == "__main__":
    main()
