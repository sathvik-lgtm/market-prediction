from datetime import date, timedelta

import pytest

from market_pred.db.access import get_news_for_ticker
from market_pred.db.connection import get_connection
from market_pred.ingest import news as news_ingest
from tests.conftest import make_settings


class FakeResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data
        self.text = str(json_data)

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


def test_fetch_news_for_ticker_searches_titles_only_and_passes_domains(monkeypatch):
    payload = make_newsapi_payload([make_raw_article()])
    captured = {}

    def fake_get(url, params, timeout):
        captured.update(params)
        return FakeResponse(200, payload)

    monkeypatch.setattr(news_ingest.requests, "get", fake_get)

    news_ingest.fetch_news_for_ticker(
        ticker="RELIANCE.NS", company="Reliance Industries",
        from_date=date(2024, 1, 1), to_date=date(2024, 1, 10),
        api_key="fake-key", page_size=100, max_pages=3,
        domains="moneycontrol.com,livemint.com",
    )

    assert captured["qInTitle"] == '"Reliance Industries" OR RELIANCE'
    assert "q" not in captured
    assert captured["domains"] == "moneycontrol.com,livemint.com"


def test_fetch_news_for_ticker_never_pages_past_the_free_tier_result_cap(monkeypatch):
    payload = make_newsapi_payload([make_raw_article()])
    payload["totalResults"] = 250  # far more than the free tier can actually page through
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["page"])
        return FakeResponse(200, payload)

    monkeypatch.setattr(news_ingest.requests, "get", fake_get)

    articles, requests_used = news_ingest.fetch_news_for_ticker(
        ticker="RELIANCE.NS", company="Reliance Industries",
        from_date=date(2024, 1, 1), to_date=date(2024, 1, 10),
        api_key="fake-key", page_size=100, max_pages=3,
    )

    assert calls == [1]  # never requested page 2 even though totalResults implied more
    assert requests_used == 1


def test_fetch_news_for_ticker_preserves_earlier_pages_when_a_later_page_errors(monkeypatch):
    page1_payload = make_newsapi_payload([make_raw_article(url="https://example.com/1")])
    page1_payload["totalResults"] = 150  # more than one page's worth at page_size=50

    def fake_get(url, params, timeout):
        if params["page"] == 1:
            return FakeResponse(200, page1_payload)
        return FakeResponse(400, {"message": "You have requested too many results."})

    monkeypatch.setattr(news_ingest.requests, "get", fake_get)

    articles, requests_used = news_ingest.fetch_news_for_ticker(
        ticker="RELIANCE.NS", company="Reliance Industries",
        from_date=date(2024, 1, 1), to_date=date(2024, 1, 10),
        api_key="fake-key", page_size=50, max_pages=3,
    )

    assert requests_used == 1  # page 1 succeeded; page 2's failed request wasn't counted
    assert len(articles) == 1  # page 1's article is preserved, not lost


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


def test_earliest_allowed_date_parses_iso_date_from_message():
    err = news_ingest.NewsApiDateRangeError(
        "You are trying to request results too far in the past. Your plan permits "
        "you to request articles as far back as 2024-08-24, but you have requested "
        "2024-08-01."
    )
    assert err.earliest_allowed_date() == date(2024, 8, 24)


def test_earliest_allowed_date_returns_none_without_a_date():
    assert news_ingest.NewsApiDateRangeError("Upgrade to a paid plan.").earliest_allowed_date() is None


def test_query_newsapi_raises_date_range_error_on_426(monkeypatch):
    body = {"status": "error", "message": "as far back as 2024-08-24"}
    monkeypatch.setattr(news_ingest.requests, "get", lambda *a, **k: FakeResponse(426, body))
    with pytest.raises(news_ingest.NewsApiDateRangeError) as exc_info:
        news_ingest._query_newsapi("q", date(2024, 1, 1), date(2024, 1, 2), "key", 1, 100)
    assert exc_info.value.earliest_allowed_date() == date(2024, 8, 24)


def _fake_settings(db_path):
    return make_settings(db_path, news_domains=["example-financial-news.com"])


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


def test_run_retries_with_adjusted_date_on_426(tmp_path, monkeypatch):
    settings = _fake_settings(tmp_path / "test.db")
    monkeypatch.setattr(news_ingest, "get_settings", lambda: settings)
    monkeypatch.setenv("NEWSAPI_KEY", "fake-key")

    allowed_date = date.today() - timedelta(days=10)  # stricter than the configured 29-day lookback
    error_body = {"message": f"as far back as {allowed_date.isoformat()}"}
    success_payload = make_newsapi_payload([make_raw_article(url="https://example.com/1")])
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["from"])
        if params["from"] != allowed_date.isoformat():
            return FakeResponse(426, error_body)
        return FakeResponse(200, success_payload)

    monkeypatch.setattr(news_ingest.requests, "get", fake_get)

    news_ingest.run()

    assert calls[0] != allowed_date.isoformat()  # first attempt used the configured lookback
    assert calls[-1] == allowed_date.isoformat()  # retry used the API's actual cutoff

    with get_connection(settings.db_path) as conn:
        rows = get_news_for_ticker(conn, "TEST.NS")
    assert len(rows) == 1
