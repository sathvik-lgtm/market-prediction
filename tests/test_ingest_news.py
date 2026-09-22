from datetime import date

import pytest

from market_pred.config import Settings, TickerInfo
from market_pred.db.access import get_news_for_ticker
from market_pred.db.connection import get_connection
from market_pred.ingest import news as news_ingest


class FakeResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


def make_raw_article(url="https://example.com/1", published="2024-01-02T10:00:00Z", title="Headline"):
    return {
        "source": {"id": None, "name": "Test Source"},
        "author": "Jane Doe",
        "title": title,
        "description": "desc",
        "url": url,
        "urlToImage": None,
        "publishedAt": published,
        "content": "content",
    }


def make_newsapi_payload(articles):
    return {"status": "ok", "totalResults": len(articles), "articles": articles}


def test_build_query():
    assert news_ingest.build_query("RELIANCE.NS", "Reliance Industries") == '"Reliance Industries" OR RELIANCE'


def test_fetch_news_for_ticker_normalizes_and_stops_pagination(monkeypatch):
    payload = make_newsapi_payload([make_raw_article()])
    monkeypatch.setattr(news_ingest.requests, "get", lambda *a, **k: FakeResponse(200, payload))

    articles, requests_used = news_ingest.fetch_news_for_ticker(
        ticker="RELIANCE.NS", company="Reliance Industries",
        from_date=date(2024, 1, 1), to_date=date(2024, 1, 10),
        api_key="fake-key", page_size=100, max_pages=3,
    )

    assert requests_used == 1  # totalResults <= page_size, no further pages fetched
    assert len(articles) == 1
    assert articles[0].ticker == "RELIANCE.NS"
    assert articles[0].published_date_ist == "2024-01-02"  # 10:00 UTC -> 15:30 IST, same day


def test_fetch_news_for_ticker_skips_articles_missing_required_fields(monkeypatch):
    payload = make_newsapi_payload([make_raw_article(), {"title": None, "url": None, "publishedAt": None}])
    monkeypatch.setattr(news_ingest.requests, "get", lambda *a, **k: FakeResponse(200, payload))

    articles, _ = news_ingest.fetch_news_for_ticker(
        ticker="RELIANCE.NS", company="Reliance Industries",
        from_date=date(2024, 1, 1), to_date=date(2024, 1, 10),
        api_key="fake-key",
    )
    assert len(articles) == 1


def test_query_newsapi_raises_auth_error_on_401(monkeypatch):
    monkeypatch.setattr(news_ingest.requests, "get", lambda *a, **k: FakeResponse(401, {}))
    with pytest.raises(news_ingest.NewsApiAuthError):
        news_ingest._query_newsapi("q", date(2024, 1, 1), date(2024, 1, 2), "key", 1, 100)


def test_query_newsapi_raises_recoverable_error_on_429(monkeypatch):
    monkeypatch.setattr(news_ingest.requests, "get", lambda *a, **k: FakeResponse(429, {}))
    with pytest.raises(news_ingest.NewsApiError):
        news_ingest._query_newsapi("q", date(2024, 1, 1), date(2024, 1, 2), "key", 1, 100)


def _fake_settings(db_path):
    return Settings(
        tickers=[TickerInfo(symbol="TEST.NS", company="Test Co")],
        db_path=db_path,
        price_history_start=date(2024, 1, 1),
        price_backfill_buffer_days=5,
        news_lookback_days=29,
        news_page_size=100,
        news_max_pages_per_ticker=3,
        news_max_requests_per_run=90,
    )


def test_run_inserts_articles_and_dedups_across_runs(tmp_path, monkeypatch):
    settings = _fake_settings(tmp_path / "test.db")
    monkeypatch.setattr(news_ingest, "get_settings", lambda: settings)
    monkeypatch.setenv("NEWSAPI_KEY", "fake-key")

    payload = make_newsapi_payload([make_raw_article(url="https://example.com/1")])
    monkeypatch.setattr(news_ingest.requests, "get", lambda *a, **k: FakeResponse(200, payload))

    news_ingest.run()
    news_ingest.run()  # re-run must not duplicate rows

    with get_connection(settings.db_path) as conn:
        rows = get_news_for_ticker(conn, "TEST.NS")
    assert len(rows) == 1


def test_run_raises_immediately_on_auth_error(tmp_path, monkeypatch):
    settings = _fake_settings(tmp_path / "test.db")
    monkeypatch.setattr(news_ingest, "get_settings", lambda: settings)
    monkeypatch.setenv("NEWSAPI_KEY", "fake-key")
    monkeypatch.setattr(news_ingest.requests, "get", lambda *a, **k: FakeResponse(401, {}))

    with pytest.raises(news_ingest.NewsApiAuthError):
        news_ingest.run()
