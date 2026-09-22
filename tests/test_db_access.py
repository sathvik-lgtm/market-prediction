from datetime import date

import pandas as pd

from market_pred.db.access import (
    get_last_news_date,
    get_last_price_date,
    get_news_for_ticker,
    get_price_history,
    insert_news_articles,
    upsert_prices,
)
from market_pred.ingest.news import NormalizedArticle


def make_price_df(rows):
    return pd.DataFrame(rows)


def test_upsert_prices_inserts_and_updates(db_conn):
    df = make_price_df([
        {"date": "2024-01-02", "open": 100.0, "high": 105.0, "low": 99.0, "close": 104.0,
         "volume": 1000, "fetched_at": "2024-01-03T00:00:00Z"},
    ])
    n = upsert_prices(db_conn, df, "TEST.NS")
    assert n == 1

    history = get_price_history(db_conn, "TEST.NS")
    assert len(history) == 1
    assert history.iloc[0]["close"] == 104.0

    # Re-upserting the same (ticker, date) with revised values should update, not duplicate.
    df2 = make_price_df([
        {"date": "2024-01-02", "open": 100.0, "high": 106.0, "low": 99.0, "close": 105.5,
         "volume": 1200, "fetched_at": "2024-01-04T00:00:00Z"},
    ])
    upsert_prices(db_conn, df2, "TEST.NS")

    history = get_price_history(db_conn, "TEST.NS")
    assert len(history) == 1
    assert history.iloc[0]["close"] == 105.5


def test_get_last_price_date(db_conn):
    assert get_last_price_date(db_conn, "TEST.NS") is None

    df = make_price_df([
        {"date": "2024-01-02", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "fetched_at": "x"},
        {"date": "2024-01-05", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "fetched_at": "x"},
    ])
    upsert_prices(db_conn, df, "TEST.NS")
    assert get_last_price_date(db_conn, "TEST.NS") == date(2024, 1, 5)


def make_article(url="https://example.com/a", ticker="TEST.NS", published="2024-01-02T10:00:00Z"):
    return NormalizedArticle(
        ticker=ticker,
        query_used="TEST",
        source_name="Test Source",
        author=None,
        title="Some headline",
        description=None,
        url=url,
        url_to_image=None,
        content=None,
        published_at_utc=published,
        published_date_ist="2024-01-02",
        fetched_at="2024-01-02T11:00:00Z",
    )


def test_insert_news_articles_dedups_on_ticker_and_url(db_conn):
    article = make_article()
    n = insert_news_articles(db_conn, [article])
    assert n == 1

    # Re-inserting the exact same (ticker, url) should be ignored, not duplicated.
    n2 = insert_news_articles(db_conn, [article])
    assert n2 == 0

    rows = get_news_for_ticker(db_conn, "TEST.NS")
    assert len(rows) == 1


def test_get_last_news_date(db_conn):
    assert get_last_news_date(db_conn, "TEST.NS") is None

    insert_news_articles(db_conn, [
        make_article(url="https://example.com/a", published="2024-01-02T10:00:00Z"),
        make_article(url="https://example.com/b", published="2024-01-05T10:00:00Z"),
    ])
    last = get_last_news_date(db_conn, "TEST.NS")
    assert last.strftime("%Y-%m-%d") == "2024-01-05"
