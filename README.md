# Market Sentiment + Price Movement Predictor

Predicts short-term price direction (up/down) for NSE-listed Indian stocks by combining
financial news sentiment with historical price/technical data.

**Status: Phase 2 of 5 (Sentiment Modeling).** Data acquisition (price/news ingestion)
and sentiment scoring (a self-fine-tuned FinBERT) are implemented. Predictive modeling
and the Streamlit dashboard are later phases and are not implemented yet.

## Tech stack (Phases 1-2)

- Python
- [`yfinance`](https://pypi.org/project/yfinance/) — historical daily OHLCV price data
- [NewsAPI](https://newsapi.org/) — news headlines per ticker
- SQLite — storage, via a thin data access layer in `market_pred/db/access.py`
- [VADER](https://github.com/cjhutto/vaderSentiment) — rule-based sentiment baseline
- [`transformers`](https://huggingface.co/docs/transformers)/`torch` — FinBERT fine-tuning and inference
- [Financial PhraseBank](https://huggingface.co/datasets/gtfintechlab/financial_phrasebank_sentences_allagree) — labeled dataset used to fine-tune FinBERT

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
history and news headlines for the 5 tickers configured in `config.yaml`. If a
fine-tuned sentiment model already exists (see below), `refresh` also scores any new
headlines automatically.

### Try it without a NewsAPI key or training a model

A small demo dataset — including already-scored news and computed daily sentiment — is
committed at `data/seed/`. To load it into a fresh local DB:

```powershell
python scripts/load_seed_data.py
```

### Fine-tune the sentiment model

One-time step, not part of the regular `refresh` cycle (CPU fine-tuning takes ~20
minutes on this dataset size):

```powershell
python -m market_pred.pipeline train-sentiment
```

Fine-tunes `yiyanghkust/finbert-pretrain` (BERT further pretrained on financial text,
with no classification head yet) on the [Financial PhraseBank](https://huggingface.co/datasets/gtfintechlab/financial_phrasebank_sentences_allagree)
dataset, and saves the result to `models/finbert_finetuned/final` (gitignored — model
weights don't belong in git; re-run this to reproduce). After training, score the
actual news headlines and recompute daily aggregates:

```powershell
python -m market_pred.pipeline sentiment
```

## Usage

```powershell
python -m market_pred.pipeline init-db                              # create tables only
python -m market_pred.pipeline prices                               # fetch/upsert prices, all tickers
python -m market_pred.pipeline prices --tickers RELIANCE.NS TCS.NS  # subset of tickers
python -m market_pred.pipeline prices --full-refresh                # refetch full history
python -m market_pred.pipeline news                                 # fetch/insert news, all tickers
python -m market_pred.pipeline refresh                               # prices + news + sentiment (if trained)
python -m market_pred.pipeline train-sentiment                      # fine-tune FinBERT (one-time, ~20 min on CPU)
python -m market_pred.pipeline sentiment                            # score unscored headlines + recompute aggregates
python -m market_pred.pipeline evaluate-sentiment                   # VADER vs off-the-shelf vs fine-tuned comparison
```

All commands are safe to re-run: price upserts are idempotent (keyed on `ticker, date`),
news inserts are deduplicated (keyed on `ticker, url`), and sentiment scoring only
processes headlines that don't have a score yet.

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

**Sentiment columns on `news`** (added in Phase 2, nullable until scored):
`vader_score` (VADER compound score, -1 to 1), `finbert_label`
(negative/neutral/positive), `finbert_score` (signed: p_positive - p_negative, -1 to 1),
`finbert_confidence` (probability of the predicted label), `sentiment_scored_at`
(NULL = not yet scored; this is how the scoring pipeline finds unscored rows).

**`daily_sentiment`** — one row per `(ticker, date)`, aggregated from all *scored*
`news` rows sharing that `published_date_ist`: `mean_finbert_score`, `mean_vader_score`,
`article_count`, `computed_at`. Same lookahead-bias caveat as `published_date_ist`
applies here — this is a plain calendar-day aggregate, not yet joined to any price
target.

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

## Sentiment model

Three sentiment scorers exist, in increasing order of sophistication:

1. **VADER** — rule-based, no training, used only as a floor/benchmark.
2. **`ProsusAI/finbert`** (off-the-shelf) — already fine-tuned on Financial PhraseBank
   by its authors, used only as an evaluation comparison point, not in the regular
   scoring pipeline.
3. **Our own fine-tuned FinBERT** (used in the actual pipeline) — starts from
   [`yiyanghkust/finbert-pretrain`](https://huggingface.co/yiyanghkust/finbert-pretrain),
   a BERT further pretrained on a large financial corpus (10-K/8-K filings, earnings
   calls, analyst reports) but with **no classification head yet**. Fine-tuning
   `ProsusAI/finbert` again on the same Financial PhraseBank data it was already
   trained on would be redundant — this base model exists specifically so people
   fine-tune it themselves, which is what `market_pred/sentiment/train.py` does: a
   3-class head trained via HF `Trainer` on an 80/10/10-equivalent train/validation/test
   split (validation carved out of the training set with stratified sampling; the
   dataset's own test split stays fully held out).

**Evaluation** (`python -m market_pred.pipeline evaluate-sentiment`), all three scorers
on the same held-out Financial PhraseBank test split (680 examples, never touched
during training):

| Model | Accuracy | Macro F1 |
|---|---|---|
| VADER (rule-based baseline) | 56.2% | 48.4% |
| ProsusAI/finbert (off-the-shelf) | 97.1% | 96.0% |
| **Our fine-tuned FinBERT** | **96.6%** | **95.7%** |

Our fine-tuned model performs essentially on par with the professionally fine-tuned
off-the-shelf FinBERT (well within noise given the test set's class imbalance: 93
negative / 417 neutral / 170 positive), and both dramatically outperform the rule-based
baseline — validating that the fine-tuning pipeline itself is sound, not just that
"FinBERT is good."

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
- **Sentiment coverage is sparse relative to price history.** Price history spans
  2018-present (~2,159 trading days/ticker); news — and therefore `daily_sentiment` —
  only covers the last ~29 days per NewsAPI's free-tier lookback, growing forward from
  whenever `refresh` starts running regularly. A later phase's feature engineering will
  need a strategy for the vast majority of price history having no sentiment signal
  (e.g. training/evaluating only on the overlapping window, or treating missing days as
  neutral) — not solved here, just flagged so it isn't a surprise.
- The Financial PhraseBank test-set evaluation measures sentiment classification
  quality on analyst-style sentences, not on our actual NSE headlines (which have no
  ground-truth sentiment labels to evaluate against) — a proxy, not a direct measure of
  how well it scores our specific data.

## Running tests

```powershell
pytest
```

All tests mock `yfinance`/`requests` and use a temp SQLite DB — no network calls or real
API keys are needed to run the test suite.
