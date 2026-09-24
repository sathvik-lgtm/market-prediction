# Market Sentiment + Price Movement Predictor

Predicts short-term price direction (up/down) for NSE-listed Indian stocks by combining
financial news sentiment with historical price/technical data.

**Status: Phase 4 of 5 (Dashboard).** Data acquisition, sentiment scoring, walk-forward-validated
direction classifiers, and a read-only Streamlit dashboard are all implemented. Only Phase 5
(write-up) remains.

## Tech stack (Phases 1-4)

- Python
- [`yfinance`](https://pypi.org/project/yfinance/) — historical daily OHLCV price data
- [NewsAPI](https://newsapi.org/) — news headlines per ticker
- SQLite — storage, via a thin data access layer in `market_pred/db/access.py`
- [VADER](https://github.com/cjhutto/vaderSentiment) — rule-based sentiment baseline
- [`transformers`](https://huggingface.co/docs/transformers)/`torch` — FinBERT fine-tuning and inference
- [Financial PhraseBank](https://huggingface.co/datasets/gtfintechlab/financial_phrasebank_sentences_allagree) — labeled dataset used to fine-tune FinBERT
- `scikit-learn` (logistic regression, metrics) / `xgboost` — direction classifiers
- `streamlit` / `plotly` — the dashboard

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
python -m market_pred.pipeline train-model                          # walk-forward train/evaluate direction models
streamlit run streamlit_app.py                                      # launch the dashboard (separate from pipeline.py)
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

## Predictive modeling

**Target**: next-day direction — `close[t+1] > close[t]`, binary. **Features**
(`market_pred/modeling/features.py`): lagged returns (1/5/10/20-day), price relative to
5/10/20-day moving averages, 20-day volume ratio, 10/20-day return volatility — all
computed with only backward-looking rolling windows, so no lookahead risk by
construction. **Models**: logistic regression and XGBoost (per the plan; LSTM is an
explicit stretch goal, not attempted yet — the baselines' results below don't currently
justify the added complexity). **Validation**: walk-forward only, never a random split —
an expanding window seeded with 2 years of history, one fold per subsequent calendar
year, split by date across all 5 pooled tickers at once so no fold's boundary can leak
one ticker's future into another's past.

**The sentiment lookahead-bias rule, finally implemented** (flagged back in Phase 1's
schema notes and deferred until this exact point): `assign_effective_trading_day()`
attributes each headline to the earliest trading day whose 15:30 IST close is at or
after its publish time. A headline published during session D counts toward predicting
D+1; one published after D's close — including across a weekend/holiday — rolls forward
to whatever the next actual trading day is. This is a *different, lookahead-safe*
aggregation from the `daily_sentiment` table (which groups by raw calendar day for
display purposes only, as documented above) — modeling recomputes its own sentiment
features from scored headlines directly rather than reusing that table.

**Results** (`python -m market_pred.pipeline train-model`), 7 walk-forward folds
(2020–2026) over 10,695 pooled rows, averaged:

| Model | Accuracy | Naive ("always up") | Strategy mean return | Buy-and-hold mean return |
|---|---|---|---|---|
| Logistic regression | 49.4% | 50.5% | 0.00013 | 0.00037 |
| XGBoost | 51.2% | 50.5% | 0.00043 | 0.00037 |

XGBoost edges out the naive baseline and buy-and-hold on average, but only marginally —
consistent with next-day direction from technical indicators alone being a genuinely
hard, close-to-efficient-market problem, not a sign of a bug. Logistic regression
doesn't beat naive at all. Neither result accounts for transaction costs or slippage,
so read the strategy-return edge as illustrative, not a claim that this is tradeable.

**Sentiment ablation** (same price-only vs. price+sentiment features, evaluated on the
identical small overlap window — 38 train / 14 test rows, 20 unique dates — since
that's all that currently exists): price-only scored 71.4% accuracy vs. 64.3% with
sentiment added. Sentiment did not help here, and this specific comparison is not
strong evidence that it can't — 14 test rows means a single flipped prediction moves
accuracy by ~7 points. A meaningful answer needs more overlapping history, which only
accumulates as `refresh` keeps running past NewsAPI's lookback window.

## Dashboard

A Streamlit app (`streamlit_app.py`, repo root) that's **read-only** against whatever's
currently in `data/market_pred.db` and `models/` — it does not trigger ingestion,
sentiment scoring, or training itself. Those stay CLI-only; use the sidebar's "Clear
cache" button after re-running one of them so the dashboard picks up the change without
restarting the process.

**Running it:**

```powershell
# Fast/offline path -- no NEWSAPI_KEY needed, seconds not minutes:
python scripts/load_seed_data.py
python -m market_pred.pipeline train-model
streamlit run streamlit_app.py

# Full/live path -- real current data:
python -m market_pred.pipeline refresh
python -m market_pred.pipeline train-sentiment   # one-time, ~20 min on CPU
python -m market_pred.pipeline sentiment
python -m market_pred.pipeline train-model
streamlit run streamlit_app.py
```

**What each panel shows**, selecting a ticker from the sidebar:
- **Price** — candlestick chart (split/dividend-adjusted close), defaulting to the last
  ~6 months with the full history reachable via the range slider.
- **News sentiment trend** — `daily_sentiment`'s calendar-day aggregate (naturally
  covers only the last ~29 days per NewsAPI's free tier, as above) plus a recent-headlines
  table with each one's FinBERT label.
- **Model prediction** — the persisted price-only XGBoost model's next-session call and
  confidence, with an expander showing its own walk-forward accuracy (~51.2% vs. ~50.5%
  naive) so a single prediction isn't over-trusted. Deliberately **not** sentiment-fused,
  matching the sentiment ablation finding above — only the price-only model was persisted.

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
- **Sentiment coverage is sparse relative to price history**, resolved in Phase 3 by
  running two separate evaluations rather than forcing one dataset to serve both: a
  full-history (2018+) price-only model for statistically meaningful walk-forward
  results, plus a small-sample ablation on just the overlapping recent window to check
  whether sentiment helps when it's actually available. See "Predictive modeling" above
  — this will keep improving as `refresh` accumulates more overlapping history over time.
- The Financial PhraseBank test-set evaluation measures sentiment classification
  quality on analyst-style sentences, not on our actual NSE headlines (which have no
  ground-truth sentiment labels to evaluate against) — a proxy, not a direct measure of
  how well it scores our specific data.
- **The price-direction models are unvalidated as a trading strategy**: the reported
  "strategy return" is a simple long/flat simulation with no transaction costs,
  slippage, or position sizing — useful for comparing models against buy-and-hold on
  equal footing, not a claim about real-world profitability.
- The dashboard is read-only and shows a point-in-time snapshot — it will not reflect
  newer data or a freshly retrained model until the relevant CLI command is re-run and
  the sidebar's cache is cleared.
- The prediction panel can't name an exact "next trading day" — this codebase has no
  market holiday calendar, only weekends implied by gaps in price history.
- A ticker with too little trailing price history (fewer than ~20 trading days) can't
  produce a prediction; the dashboard shows a clean message rather than crashing. Not
  expected for the 5 configured tickers (8 years of history each), but would apply to
  a newly added ticker.

## Running tests

```powershell
pytest
```

All tests mock `yfinance`/`requests` and use a temp SQLite DB — no network calls or real
API keys are needed to run the test suite.
