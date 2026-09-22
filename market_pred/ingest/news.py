"""News ingestion via NewsAPI's /v2/everything endpoint.

Returns a source-agnostic NormalizedArticle from fetch_news_for_ticker so a
second news source could later be added as a new fetch+mapping function that
also produces NormalizedArticle objects, without touching storage/dedup/CLI.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from market_pred.config import get_settings
from market_pred.db.access import get_last_news_date, insert_news_articles
from market_pred.db.connection import get_connection
from market_pred.db.schema import init_db

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"
IST = ZoneInfo("Asia/Kolkata")


@dataclass
class NormalizedArticle:
    ticker: str
    query_used: str
    source_name: str | None
    author: str | None
    title: str
    description: str | None
    url: str
    url_to_image: str | None
    content: str | None
    published_at_utc: str
    published_date_ist: str
    fetched_at: str


class NewsApiError(RuntimeError):
    """A recoverable NewsAPI failure (e.g. rate limit) -- caller should skip and continue."""


class NewsApiAuthError(NewsApiError):
    """An unrecoverable auth failure -- caller should stop the whole run."""


def build_query(ticker: str, company: str) -> str:
    bare_symbol = ticker.split(".")[0]
    return f'"{company}" OR {bare_symbol}'


def _query_newsapi(
    query: str, from_date: date, to_date: date, api_key: str, page: int, page_size: int
) -> dict:
    params = {
        "q": query,
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "page": page,
        "apiKey": api_key,
    }
    resp = requests.get(NEWSAPI_URL, params=params, timeout=30)
    if resp.status_code == 401:
        raise NewsApiAuthError("NewsAPI rejected the request (401) -- check NEWSAPI_KEY in .env")
    if resp.status_code == 429:
        raise NewsApiError("NewsAPI rate limit hit (429)")
    if resp.status_code == 426:
        raise NewsApiError("NewsAPI rejected the date range (426) -- free tier lookback exceeded")
    resp.raise_for_status()
    return resp.json()


def _normalize_article(raw: dict, ticker: str, query: str, fetched_at: str) -> NormalizedArticle | None:
    published_at_utc = raw.get("publishedAt")
    url = raw.get("url")
    title = raw.get("title")
    if not published_at_utc or not url or not title:
        return None

    try:
        dt_utc = datetime.strptime(published_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        logger.warning("Skipping article with unparseable publishedAt=%r", published_at_utc)
        return None
    published_date_ist = dt_utc.astimezone(IST).date().isoformat()

    source = raw.get("source") or {}
    return NormalizedArticle(
        ticker=ticker,
        query_used=query,
        source_name=source.get("name"),
        author=raw.get("author"),
        title=title,
        description=raw.get("description"),
        url=url,
        url_to_image=raw.get("urlToImage"),
        content=raw.get("content"),
        published_at_utc=published_at_utc,
        published_date_ist=published_date_ist,
        fetched_at=fetched_at,
    )


def fetch_news_for_ticker(
    ticker: str,
    company: str,
    from_date: date,
    to_date: date,
    api_key: str,
    page_size: int = 100,
    max_pages: int = 3,
) -> tuple[list[NormalizedArticle], int]:
    """Fetch and normalize articles for one ticker. Returns (articles, requests_used)."""
    query = build_query(ticker, company)
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    articles: list[NormalizedArticle] = []
    requests_used = 0

    for page in range(1, max_pages + 1):
        payload = _query_newsapi(query, from_date, to_date, api_key, page, page_size)
        requests_used += 1

        raw_articles = payload.get("articles", [])
        for raw in raw_articles:
            normalized = _normalize_article(raw, ticker, query, fetched_at)
            if normalized is not None:
                articles.append(normalized)

        total_results = payload.get("totalResults", 0)
        if page * page_size >= total_results or not raw_articles:
            break

    return articles, requests_used


def run(tickers: list[str] | None = None) -> None:
    settings = get_settings()
    api_key = settings.get_newsapi_key()
    ticker_infos = settings.tickers
    if tickers:
        wanted = set(tickers)
        ticker_infos = [t for t in ticker_infos if t.symbol in wanted]

    today = date.today()
    # NewsAPI's free tier only serves roughly the last month -- clamp proactively
    # rather than reacting to the 426 the API returns for an out-of-range request.
    min_from_date = today - timedelta(days=settings.news_lookback_days)
    requests_used_total = 0

    with get_connection(settings.db_path) as conn:
        init_db(conn)

        for info in ticker_infos:
            if requests_used_total >= settings.news_max_requests_per_run:
                logger.warning(
                    "Reached max_requests_per_run (%d) -- stopping early",
                    settings.news_max_requests_per_run,
                )
                break

            try:
                last_pub = get_last_news_date(conn, info.symbol)
                from_date = min_from_date if last_pub is None else max(min_from_date, last_pub.date())

                articles, requests_used = fetch_news_for_ticker(
                    ticker=info.symbol,
                    company=info.company,
                    from_date=from_date,
                    to_date=today,
                    api_key=api_key,
                    page_size=settings.news_page_size,
                    max_pages=settings.news_max_pages_per_ticker,
                )
                requests_used_total += requests_used

                n_inserted = insert_news_articles(conn, articles)
                logger.info(
                    "%s: fetched %d article(s), inserted %d new (used %d request(s))",
                    info.symbol, len(articles), n_inserted, requests_used,
                )
            except NewsApiAuthError:
                raise
            except NewsApiError as e:
                logger.warning("News fetch failed for %s: %s -- skipping", info.symbol, e)
                continue
            except Exception:
                logger.exception("Failed to fetch/insert news for %s -- skipping", info.symbol)
                continue


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
