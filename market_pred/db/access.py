"""Data access layer. This is the only module that writes/reads SQL directly;
later phases (sentiment, modeling) should go through these functions instead
of touching sqlite3 or the schema directly.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from market_pred.ingest.news import NormalizedArticle

UPSERT_PRICE_SQL = """
INSERT INTO prices (ticker, date, open, high, low, close, volume, fetched_at)
VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :fetched_at)
ON CONFLICT (ticker, date) DO UPDATE SET
    open = excluded.open,
    high = excluded.high,
    low = excluded.low,
    close = excluded.close,
    volume = excluded.volume,
    fetched_at = excluded.fetched_at
"""

INSERT_NEWS_SQL = """
INSERT OR IGNORE INTO news (
    ticker, query_used, source_name, author, title, description,
    url, url_to_image, content, published_at_utc, published_date_ist, fetched_at
) VALUES (
    :ticker, :query_used, :source_name, :author, :title, :description,
    :url, :url_to_image, :content, :published_at_utc, :published_date_ist, :fetched_at
)
"""


def upsert_prices(conn: sqlite3.Connection, df: pd.DataFrame, ticker: str) -> int:
    """Upsert a DataFrame of OHLCV rows (indexed or columned by date) for one ticker.

    Expects columns: date (str ISO), open, high, low, close, volume, fetched_at.
    Returns the number of rows written.
    """
    if df.empty:
        return 0
    rows = df.to_dict(orient="records")
    for row in rows:
        row["ticker"] = ticker
    conn.executemany(UPSERT_PRICE_SQL, rows)
    return len(rows)


def get_last_price_date(conn: sqlite3.Connection, ticker: str) -> date | None:
    cur = conn.execute("SELECT MAX(date) AS max_date FROM prices WHERE ticker = ?", (ticker,))
    row = cur.fetchone()
    if row is None or row["max_date"] is None:
        return None
    return datetime.strptime(row["max_date"], "%Y-%m-%d").date()


def get_price_history(
    conn: sqlite3.Connection,
    ticker: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    query = "SELECT ticker, date, open, high, low, close, volume FROM prices WHERE ticker = ?"
    params: list = [ticker]
    if start is not None:
        query += " AND date >= ?"
        params.append(start.isoformat())
    if end is not None:
        query += " AND date <= ?"
        params.append(end.isoformat())
    query += " ORDER BY date"
    return pd.read_sql_query(query, conn, params=params)


def insert_news_articles(conn: sqlite3.Connection, articles: list["NormalizedArticle"]) -> int:
    """Insert normalized articles, silently skipping duplicates on (ticker, url).

    Returns the number of newly inserted rows.
    """
    if not articles:
        return 0
    rows = [vars(a) for a in articles]
    cur = conn.executemany(INSERT_NEWS_SQL, rows)
    return cur.rowcount if cur.rowcount is not None and cur.rowcount > 0 else 0


def get_last_news_date(conn: sqlite3.Connection, ticker: str) -> datetime | None:
    cur = conn.execute(
        "SELECT MAX(published_at_utc) AS max_pub FROM news WHERE ticker = ?", (ticker,)
    )
    row = cur.fetchone()
    if row is None or row["max_pub"] is None:
        return None
    return datetime.strptime(row["max_pub"], "%Y-%m-%dT%H:%M:%SZ")


def get_news_for_ticker(
    conn: sqlite3.Connection,
    ticker: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    query = (
        "SELECT ticker, source_name, author, title, description, url, "
        "published_at_utc, published_date_ist FROM news WHERE ticker = ?"
    )
    params: list = [ticker]
    if start is not None:
        query += " AND published_date_ist >= ?"
        params.append(start.isoformat())
    if end is not None:
        query += " AND published_date_ist <= ?"
        params.append(end.isoformat())
    query += " ORDER BY published_at_utc"
    return pd.read_sql_query(query, conn, params=params)
