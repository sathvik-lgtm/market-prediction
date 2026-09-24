"""CLI entrypoint for the data pipeline.

    python -m market_pred.pipeline init-db
    python -m market_pred.pipeline prices [--tickers RELIANCE.NS TCS.NS] [--full-refresh]
    python -m market_pred.pipeline news [--tickers RELIANCE.NS TCS.NS]
    python -m market_pred.pipeline refresh
    python -m market_pred.pipeline train-sentiment
    python -m market_pred.pipeline sentiment
    python -m market_pred.pipeline evaluate-sentiment
    python -m market_pred.pipeline train-model
"""
from __future__ import annotations

import argparse
import logging

from market_pred.config import get_settings
from market_pred.db.connection import get_connection
from market_pred.db.schema import init_db
from market_pred.ingest import news as news_ingest
from market_pred.ingest import prices as prices_ingest
from market_pred.modeling import train as model_train
from market_pred.sentiment import evaluate as sentiment_evaluate
from market_pred.sentiment import run as sentiment_run
from market_pred.sentiment import train as sentiment_train

logger = logging.getLogger(__name__)


def cmd_init_db(_args: argparse.Namespace) -> None:
    settings = get_settings()
    with get_connection(settings.db_path) as conn:
        init_db(conn)
    logger.info("Database ready at %s", settings.db_path)


def cmd_prices(args: argparse.Namespace) -> None:
    prices_ingest.run(tickers=args.tickers, full_refresh=args.full_refresh)


def cmd_news(args: argparse.Namespace) -> None:
    news_ingest.run(tickers=args.tickers)


def cmd_refresh(args: argparse.Namespace) -> None:
    prices_ingest.run(tickers=args.tickers)
    news_ingest.run(tickers=args.tickers)
    settings = get_settings()
    if (settings.sentiment_model_dir / "final").exists():
        sentiment_run.run()
    else:
        logger.info(
            "No fine-tuned sentiment model yet -- skipping sentiment scoring "
            "(run `train-sentiment` first)"
        )


def cmd_train_sentiment(_args: argparse.Namespace) -> None:
    sentiment_train.run()


def cmd_sentiment(_args: argparse.Namespace) -> None:
    sentiment_run.run()


def cmd_evaluate_sentiment(_args: argparse.Namespace) -> None:
    sentiment_evaluate.run()


def cmd_train_model(_args: argparse.Namespace) -> None:
    model_train.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="market_pred.pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init-db", help="Create the SQLite database and tables")
    p_init.set_defaults(func=cmd_init_db)

    p_prices = subparsers.add_parser("prices", help="Fetch/upsert price history")
    p_prices.add_argument("--tickers", nargs="+", default=None, help="Subset of tickers to fetch")
    p_prices.add_argument("--full-refresh", action="store_true", help="Refetch full history, not just new days")
    p_prices.set_defaults(func=cmd_prices)

    p_news = subparsers.add_parser("news", help="Fetch/insert news headlines")
    p_news.add_argument("--tickers", nargs="+", default=None, help="Subset of tickers to fetch")
    p_news.set_defaults(func=cmd_news)

    p_refresh = subparsers.add_parser("refresh", help="Run prices + news ingestion")
    p_refresh.add_argument("--tickers", nargs="+", default=None, help="Subset of tickers to fetch")
    p_refresh.set_defaults(func=cmd_refresh)

    p_train_sentiment = subparsers.add_parser(
        "train-sentiment", help="Fine-tune FinBERT on Financial PhraseBank"
    )
    p_train_sentiment.set_defaults(func=cmd_train_sentiment)

    p_sentiment = subparsers.add_parser(
        "sentiment", help="Score unscored news headlines and recompute daily sentiment"
    )
    p_sentiment.set_defaults(func=cmd_sentiment)

    p_evaluate_sentiment = subparsers.add_parser(
        "evaluate-sentiment", help="Compare VADER/off-the-shelf/fine-tuned FinBERT on the held-out test split"
    )
    p_evaluate_sentiment.set_defaults(func=cmd_evaluate_sentiment)

    p_train_model = subparsers.add_parser(
        "train-model", help="Walk-forward train/evaluate direction classifiers + sentiment ablation"
    )
    p_train_model.set_defaults(func=cmd_train_model)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
