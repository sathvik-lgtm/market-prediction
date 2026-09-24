import sqlite3
from datetime import date
from pathlib import Path

import pytest

from market_pred.config import Settings, TickerInfo
from market_pred.db.schema import init_db


@pytest.fixture
def db_conn(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "test.db")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


def make_settings(db_path: Path, **overrides) -> Settings:
    """A fully-populated Settings for tests, with sensible defaults for every
    field. Pass overrides for the ones a specific test cares about, so tests
    don't break every time Settings grows a field.
    """
    defaults = dict(
        tickers=[TickerInfo(symbol="TEST.NS", company="Test Co")],
        db_path=db_path,
        price_history_start=date(2024, 1, 1),
        price_backfill_buffer_days=5,
        news_lookback_days=29,
        news_page_size=100,
        news_max_pages_per_ticker=3,
        news_max_requests_per_run=90,
        news_domains=[],
        sentiment_finbert_base_model="yiyanghkust/finbert-pretrain",
        sentiment_finbert_pretrained_model="ProsusAI/finbert",
        sentiment_model_dir=db_path.parent / "models" / "finbert_finetuned",
        sentiment_phrasebank_repo="gtfintechlab/financial_phrasebank_sentences_allagree",
        sentiment_phrasebank_config="5768",
        sentiment_val_fraction=0.1,
        sentiment_max_seq_length=64,
        sentiment_train_batch_size=8,
        sentiment_eval_batch_size=16,
        sentiment_num_epochs=4,
        sentiment_learning_rate=2e-5,
        sentiment_seed=42,
        modeling_model_dir=db_path.parent / "models" / "direction_model",
        modeling_min_train_years=2,
        modeling_sentiment_test_fraction=0.3,
        modeling_seed=42,
    )
    defaults.update(overrides)
    return Settings(**defaults)
