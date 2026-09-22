"""CLI entrypoint for the data pipeline.

    python -m market_pred.pipeline init-db
    python -m market_pred.pipeline prices [--tickers RELIANCE.NS TCS.NS] [--full-refresh]
    python -m market_pred.pipeline news [--tickers RELIANCE.NS TCS.NS]
    python -m market_pred.pipeline refresh
"""
from __future__ import annotations

import argparse
import logging

from market_pred.config import get_settings
from market_pred.db.connection import get_connection
from market_pred.db.schema import init_db
from market_pred.ingest import news as news_ingest
from market_pred.ingest import prices as prices_ingest

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

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
