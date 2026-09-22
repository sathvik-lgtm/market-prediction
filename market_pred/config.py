"""Load .env secrets and config.yaml settings into a single Settings object."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TickerInfo:
    symbol: str
    company: str


@dataclass(frozen=True)
class Settings:
    tickers: list[TickerInfo]
    db_path: Path
    price_history_start: date
    price_backfill_buffer_days: int
    news_lookback_days: int
    news_page_size: int
    news_max_pages_per_ticker: int
    news_max_requests_per_run: int

    def get_newsapi_key(self) -> str:
        key = os.environ.get("NEWSAPI_KEY")
        if not key:
            raise RuntimeError(
                "NEWSAPI_KEY is not set. Copy .env.example to .env and add your "
                "free API key from https://newsapi.org/register"
            )
        return key


@lru_cache(maxsize=1)
def get_settings(config_path: str | Path = REPO_ROOT / "config.yaml") -> Settings:
    load_dotenv(REPO_ROOT / ".env")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    tickers = [TickerInfo(symbol=t["symbol"], company=t["company"]) for t in raw["tickers"]]
    db_path = REPO_ROOT / raw["paths"]["db_path"]

    price_cfg = raw["price_ingestion"]
    news_cfg = raw["news_ingestion"]

    return Settings(
        tickers=tickers,
        db_path=db_path,
        price_history_start=datetime.strptime(price_cfg["history_start"], "%Y-%m-%d").date(),
        price_backfill_buffer_days=int(price_cfg["backfill_buffer_days"]),
        news_lookback_days=int(news_cfg["lookback_days"]),
        news_page_size=int(news_cfg["page_size"]),
        news_max_pages_per_ticker=int(news_cfg["max_pages_per_ticker"]),
        news_max_requests_per_run=int(news_cfg["max_requests_per_run"]),
    )
