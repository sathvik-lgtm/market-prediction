"""DDL for the market_pred SQLite database."""
import sqlite3

CREATE_PRICES_TABLE = """
CREATE TABLE IF NOT EXISTS prices (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      INTEGER NOT NULL,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
)
"""

CREATE_NEWS_TABLE = """
CREATE TABLE IF NOT EXISTS news (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT NOT NULL,
    query_used          TEXT NOT NULL,
    source_name         TEXT,
    author              TEXT,
    title               TEXT NOT NULL,
    description         TEXT,
    url                 TEXT NOT NULL,
    url_to_image        TEXT,
    content             TEXT,
    published_at_utc    TEXT NOT NULL,
    published_date_ist  TEXT NOT NULL,
    fetched_at          TEXT NOT NULL,
    UNIQUE (ticker, url)
)
"""

CREATE_NEWS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_news_ticker_pubdate
ON news (ticker, published_date_ist)
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Idempotently create all tables/indexes. Safe to call on every run."""
    conn.execute(CREATE_PRICES_TABLE)
    conn.execute(CREATE_NEWS_TABLE)
    conn.execute(CREATE_NEWS_INDEX)
    conn.commit()
