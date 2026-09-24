from datetime import date

import pandas as pd
import pytest

from market_pred.db.access import (
    get_daily_sentiment,
    get_last_news_date,
    get_last_price_date,
    get_news_for_ticker,
    get_price_history,
    get_unscored_news,
    insert_news_articles,
    recompute_daily_sentiment,
    update_news_sentiment,
    upsert_daily_sentiment,
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
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    dt_utc = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    published_date_ist = dt_utc.astimezone(ZoneInfo("Asia/Kolkata")).date().isoformat()

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
        published_date_ist=published_date_ist,
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


def test_get_news_for_ticker_sentiment_columns_null_when_unscored(db_conn):
    insert_news_articles(db_conn, [make_article()])

    row = get_news_for_ticker(db_conn, "TEST.NS").iloc[0]

    assert pd.isna(row["finbert_label"])
    assert pd.isna(row["finbert_score"])
    assert pd.isna(row["vader_score"])


def test_get_news_for_ticker_includes_sentiment_columns_when_scored(db_conn):
    insert_news_articles(db_conn, [make_article()])
    unscored = get_unscored_news(db_conn)
    update_news_sentiment(db_conn, [{
        "id": int(unscored.iloc[0]["id"]),
        "vader_score": 0.4,
        "finbert_label": "positive",
        "finbert_score": 0.6,
        "finbert_confidence": 0.8,
        "sentiment_scored_at": "2024-01-03T00:00:00Z",
    }])

    row = get_news_for_ticker(db_conn, "TEST.NS").iloc[0]

    assert row["finbert_label"] == "positive"
    assert row["finbert_score"] == pytest.approx(0.6)
    assert row["vader_score"] == pytest.approx(0.4)
    assert row["finbert_confidence"] == pytest.approx(0.8)


def test_get_last_news_date(db_conn):
    assert get_last_news_date(db_conn, "TEST.NS") is None

    insert_news_articles(db_conn, [
        make_article(url="https://example.com/a", published="2024-01-02T10:00:00Z"),
        make_article(url="https://example.com/b", published="2024-01-05T10:00:00Z"),
    ])
    last = get_last_news_date(db_conn, "TEST.NS")
    assert last.strftime("%Y-%m-%d") == "2024-01-05"


def test_get_unscored_news_and_update_news_sentiment(db_conn):
    insert_news_articles(db_conn, [
        make_article(url="https://example.com/a", published="2024-01-02T10:00:00Z"),
        make_article(url="https://example.com/b", published="2024-01-02T11:00:00Z"),
    ])

    unscored = get_unscored_news(db_conn)
    assert len(unscored) == 2

    row = unscored.iloc[0]
    update_news_sentiment(db_conn, [{
        "id": int(row["id"]),
        "vader_score": 0.5,
        "finbert_label": "positive",
        "finbert_score": 0.8,
        "finbert_confidence": 0.9,
        "sentiment_scored_at": "2024-01-03T00:00:00Z",
    }])

    # The scored row should no longer show up as unscored; the other one still does.
    unscored_after = get_unscored_news(db_conn)
    assert len(unscored_after) == 1
    assert unscored_after.iloc[0]["id"] != row["id"]


def test_upsert_daily_sentiment_inserts_and_updates(db_conn):
    df = pd.DataFrame([{
        "ticker": "TEST.NS", "date": "2024-01-02", "mean_finbert_score": 0.3,
        "mean_vader_score": 0.1, "article_count": 2, "computed_at": "2024-01-03T00:00:00Z",
    }])
    n = upsert_daily_sentiment(db_conn, df)
    assert n == 1

    result = get_daily_sentiment(db_conn, "TEST.NS")
    assert len(result) == 1
    assert result.iloc[0]["article_count"] == 2

    # Re-upserting the same (ticker, date) should update, not duplicate.
    df2 = pd.DataFrame([{
        "ticker": "TEST.NS", "date": "2024-01-02", "mean_finbert_score": 0.6,
        "mean_vader_score": 0.2, "article_count": 3, "computed_at": "2024-01-04T00:00:00Z",
    }])
    upsert_daily_sentiment(db_conn, df2)

    result = get_daily_sentiment(db_conn, "TEST.NS")
    assert len(result) == 1
    assert result.iloc[0]["article_count"] == 3


def test_recompute_daily_sentiment_aggregates_only_scored_rows(db_conn):
    insert_news_articles(db_conn, [
        make_article(url="https://example.com/a", published="2024-01-02T05:00:00Z"),  # -> 2024-01-02 IST
        make_article(url="https://example.com/b", published="2024-01-02T06:00:00Z"),  # -> 2024-01-02 IST
        make_article(url="https://example.com/c", published="2024-01-03T05:00:00Z"),  # unscored, excluded
    ])
    unscored = get_unscored_news(db_conn).sort_values("id")
    ids = unscored["id"].tolist()

    update_news_sentiment(db_conn, [
        {"id": ids[0], "vader_score": 0.4, "finbert_label": "positive", "finbert_score": 0.6,
         "finbert_confidence": 0.7, "sentiment_scored_at": "2024-01-03T00:00:00Z"},
        {"id": ids[1], "vader_score": -0.2, "finbert_label": "negative", "finbert_score": -0.4,
         "finbert_confidence": 0.65, "sentiment_scored_at": "2024-01-03T00:00:00Z"},
    ])

    n = recompute_daily_sentiment(db_conn)
    assert n == 1  # only one (ticker, date) group among the scored rows

    result = get_daily_sentiment(db_conn, "TEST.NS")
    assert len(result) == 1
    assert result.iloc[0]["date"] == "2024-01-02"
    assert result.iloc[0]["article_count"] == 2
    assert result.iloc[0]["mean_finbert_score"] == pytest.approx(0.1)  # mean(0.6, -0.4)
    assert result.iloc[0]["mean_vader_score"] == pytest.approx(0.1)  # mean(0.4, -0.2)
