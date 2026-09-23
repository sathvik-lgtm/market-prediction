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

# Per-headline sentiment scores, added to `news` in Phase 2. sentiment_scored_at
# is NULL until scored -- that's how the scoring pipeline finds unscored rows.
NEWS_SENTIMENT_COLUMNS = {
    "vader_score": "REAL",
    "finbert_label": "TEXT",
    "finbert_score": "REAL",
    "finbert_confidence": "REAL",
    "sentiment_scored_at": "TEXT",
}

CREATE_DAILY_SENTIMENT_TABLE = """
CREATE TABLE IF NOT EXISTS daily_sentiment (
    ticker               TEXT NOT NULL,
    date                 TEXT NOT NULL,
    mean_finbert_score   REAL NOT NULL,
    mean_vader_score     REAL NOT NULL,
    article_count        INTEGER NOT NULL,
    computed_at          TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
)
"""


def _add_missing_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, sql_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


def init_db(conn: sqlite3.Connection) -> None:
    """Idempotently create all tables/indexes/columns. Safe to call on every run."""
    conn.execute(CREATE_PRICES_TABLE)
    conn.execute(CREATE_NEWS_TABLE)
    conn.execute(CREATE_NEWS_INDEX)
    conn.execute(CREATE_DAILY_SENTIMENT_TABLE)
    _add_missing_columns(conn, "news", NEWS_SENTIMENT_COLUMNS)
    conn.commit()
