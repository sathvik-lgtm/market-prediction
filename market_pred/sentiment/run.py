"""Score any unscored `news` headlines with VADER + fine-tuned FinBERT, then
recompute the daily_sentiment aggregate table. Idempotent: only scores rows
that don't have a score yet.

    python -m market_pred.sentiment.run
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from market_pred.config import get_settings
from market_pred.db.access import get_unscored_news, recompute_daily_sentiment, update_news_sentiment
from market_pred.db.connection import get_connection
from market_pred.db.schema import init_db
from market_pred.sentiment.baseline import vader_score
from market_pred.sentiment.model import FinbertModel

logger = logging.getLogger(__name__)


def run() -> None:
    settings = get_settings()
    final_dir = settings.sentiment_model_dir / "final"
    if not final_dir.exists():
        raise RuntimeError(
            f"No fine-tuned model found at {final_dir}. Run "
            "`python -m market_pred.sentiment.train` first."
        )

    with get_connection(settings.db_path) as conn:
        init_db(conn)

        unscored = get_unscored_news(conn)
        if unscored.empty:
            logger.info("No unscored headlines -- nothing to do")
        else:
            logger.info("Scoring %d headline(s)", len(unscored))
            model = FinbertModel(final_dir)
            finbert_results = model.score_batch(unscored["title"].tolist())
            scored_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            rows = []
            for row, result in zip(unscored.itertuples(), finbert_results):
                rows.append({
                    "id": row.id,
                    "vader_score": vader_score(row.title),
                    "finbert_label": result.label,
                    "finbert_score": result.score,
                    "finbert_confidence": result.confidence,
                    "sentiment_scored_at": scored_at,
                })
            update_news_sentiment(conn, rows)
            logger.info("Scored %d headline(s)", len(rows))

        n = recompute_daily_sentiment(conn)
        logger.info("Recomputed daily_sentiment: %d ticker-day row(s)", n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
