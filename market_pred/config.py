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
    news_domains: list[str]
    sentiment_finbert_base_model: str
    sentiment_finbert_pretrained_model: str
    sentiment_model_dir: Path
    sentiment_phrasebank_repo: str
    sentiment_phrasebank_config: str
    sentiment_val_fraction: float
    sentiment_max_seq_length: int
    sentiment_train_batch_size: int
    sentiment_eval_batch_size: int
    sentiment_num_epochs: int
    sentiment_learning_rate: float
    sentiment_seed: int

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
    sentiment_cfg = raw["sentiment"]

    return Settings(
        tickers=tickers,
        db_path=db_path,
        price_history_start=datetime.strptime(price_cfg["history_start"], "%Y-%m-%d").date(),
        price_backfill_buffer_days=int(price_cfg["backfill_buffer_days"]),
        news_lookback_days=int(news_cfg["lookback_days"]),
        news_page_size=int(news_cfg["page_size"]),
        news_max_pages_per_ticker=int(news_cfg["max_pages_per_ticker"]),
        news_max_requests_per_run=int(news_cfg["max_requests_per_run"]),
        news_domains=list(news_cfg.get("domains", [])),
        sentiment_finbert_base_model=sentiment_cfg["finbert_base_model"],
        sentiment_finbert_pretrained_model=sentiment_cfg["finbert_pretrained_model"],
        sentiment_model_dir=REPO_ROOT / sentiment_cfg["model_dir"],
        sentiment_phrasebank_repo=sentiment_cfg["phrasebank_repo"],
        sentiment_phrasebank_config=sentiment_cfg["phrasebank_config"],
        sentiment_val_fraction=float(sentiment_cfg["val_fraction"]),
        sentiment_max_seq_length=int(sentiment_cfg["max_seq_length"]),
        sentiment_train_batch_size=int(sentiment_cfg["train_batch_size"]),
        sentiment_eval_batch_size=int(sentiment_cfg["eval_batch_size"]),
        sentiment_num_epochs=int(sentiment_cfg["num_epochs"]),
        sentiment_learning_rate=float(sentiment_cfg["learning_rate"]),
        sentiment_seed=int(sentiment_cfg["seed"]),
    )
