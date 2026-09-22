# Market Sentiment + Price Movement Predictor

Predicts short-term price direction (up/down) for NSE-listed Indian stocks by combining
financial news sentiment with historical price/technical data.

**Status: Phase 1 of 5 (Data Pipeline).** This phase covers data acquisition only —
price ingestion, news ingestion, and storage. Sentiment scoring (FinBERT), predictive
modeling, and the Streamlit dashboard are later phases and are not implemented yet.

## Tech stack (Phase 1)

- Python
- [`yfinance`](https://pypi.org/project/yfinance/) — historical daily OHLCV price data
- [NewsAPI](https://newsapi.org/) — news headlines per ticker
- SQLite — storage, via a thin data access layer in `market_pred/db/access.py`

## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `pip install` fails because a dependency has no prebuilt wheel for your Python version
yet (this project was scaffolded against a very new Python 3.14 — if you hit build errors,
recreate the venv with `py -3.12 -m venv venv` or `py -3.11 -m venv venv` instead).

Get a free NewsAPI key at [newsapi.org/register](https://newsapi.org/register), then:

```powershell
copy .env.example .env
# edit .env and paste your key into NEWSAPI_KEY=
```

Run the pipeline:

```powershell
python -m market_pred.pipeline refresh
```

This creates `data/market_pred.db` if it doesn't exist, then fetches/upserts price
history and news headlines for the 5 tickers configured in `config.yaml`.

### Try it without a NewsAPI key

A small demo dataset is committed at `data/seed/`. To load it into a fresh local DB:

```powershell
python scripts/load_seed_data.py
```

## Usage

```powershell
python -m market_pred.pipeline init-db                              # create tables only
python -m market_pred.pipeline prices                               # fetch/upsert prices, all tickers
python -m market_pred.pipeline prices --tickers RELIANCE.NS TCS.NS  # subset of tickers
python -m market_pred.pipeline prices --full-refresh                # refetch full history
python -m market_pred.pipeline news                                 # fetch/insert news, all tickers
python -m market_pred.pipeline refresh                               # prices + news
```

All commands are safe to re-run: price upserts are idempotent (keyed on `ticker, date`),
and news inserts are deduplicated (keyed on `ticker, url`).

To regenerate the committed demo snapshot from your local DB:

```powershell
python scripts/export_seed_data.py
```

## Schema

**`prices`** — one row per `(ticker, date)`, the NSE trading day in IST.
`date, open, high, low, close, volume, fetched_at`. `close` is split/dividend-adjusted
(`auto_adjust=True`).

**`news`** — one row per `(ticker, url)`. `title, description, source_name, author, url,
content, published_at_utc, published_date_ist, fetched_at`, plus `query_used` (the exact
NewsAPI query string, for reproducibility).

**Lookahead-bias note:** `published_date_ist` is a convenience column for grouping/EDA —
it must **not** be used directly to join a headline to the same day's price return. NSE
trades 09:15–15:30 IST; a headline published at 22:00 IST could not have influenced that
day's close, and weekend/holiday articles have no matching trading bar at all. The
full-precision `published_at_utc` timestamp is preserved so that a correct cutoff/rolling-
forward rule (attributing a headline to the next tradeable session it could actually have
influenced) can be applied during feature engineering in a later phase — that rule is
deliberately not baked into storage now.

## News query design

Articles are matched by `qInTitle` (headline only, not full article body) restricted to
a curated allowlist of Indian financial news domains (`config.yaml`'s
`news_ingestion.domains`), not NewsAPI's full ~150,000-source index. Both choices exist
because searching full article content across all domains initially made bare ticker
symbols collide with unrelated uses — e.g. "TCS" matched car reviews ("Traction Control
System") and even a PyPI package release, and "RELIANCE" matched any article using the
common English word "reliance". Restricting to financial-press domains and headline-only
matching eliminates nearly all of that noise while keeping recall of genuinely
company-specific headlines.

## Known limitations

- **NewsAPI free tier**: the "Developer" plan only returns articles from roughly the last
  month and caps requests at 100/day. Historical news from before this pipeline's first
  run cannot be recovered. Run `refresh` regularly (e.g. a daily scheduled task) to
  accumulate news history over time — price history will span years while news history
  starts short and grows from here.
- **NewsAPI's actual date-range cutoff and per-query result cap are enforced by the API,
  not just documented values.** The pipeline handles both automatically: a 426 (lookback
  exceeded) is retried once with the cutoff date NewsAPI's own error message reports, and
  pagination never requests past the ~100-result-per-query ceiling the free tier enforces.
- `yfinance` is an unofficial wrapper around Yahoo Finance endpoints with no SLA; it can
  break without notice if Yahoo changes its API.
- A second news source could be added later without a rewrite: `ingest/news.py`'s
  `fetch_news_for_ticker` always returns a list of source-agnostic `NormalizedArticle`
  objects, so storage, dedup, and the CLI wouldn't need to change.

## Running tests

```powershell
pytest
```

All tests mock `yfinance`/`requests` and use a temp SQLite DB — no network calls or real
API keys are needed to run the test suite.
