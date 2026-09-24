# Market Sentiment + Price Movement Predictor

Predicts short-term price direction (up/down) for NSE-listed Indian stocks by combining
financial news sentiment with historical price/technical data.

**Status: complete (Phases 1-5).** Data acquisition, sentiment scoring, walk-forward-validated
direction classifiers, and a Streamlit dashboard (with an in-app refresh/retrain button) are
all implemented and documented end to end.

> This is a living project — all 5 planned phases are done, but I'll keep coming back to
> improve it (more tickers, better sentiment coverage, a stronger model) as I learn more.
> See [Known limitations](#known-limitations) for what's on my radar next.

## Tech stack

- Python
- [`yfinance`](https://pypi.org/project/yfinance/) — historical daily OHLCV price data
- [NewsAPI](https://newsapi.org/) — news headlines per ticker
- SQLite — storage, via a thin data access layer in `market_pred/db/access.py`
- [VADER](https://github.com/cjhutto/vaderSentiment) — rule-based sentiment baseline
- [`transformers`](https://huggingface.co/docs/transformers)/`torch` — FinBERT fine-tuning and inference
- [Financial PhraseBank](https://huggingface.co/datasets/gtfintechlab/financial_phrasebank_sentences_allagree) — labeled dataset used to fine-tune FinBERT
- `scikit-learn` (logistic regression, metrics) / `xgboost` — direction classifiers
- `streamlit` / `plotly` — the dashboard
- [`pandas_market_calendars`](https://github.com/rsheftel/pandas_market_calendars) — NSE's actual
  trading-day calendar, so the dashboard's "next session" date correctly skips holidays, not just weekends

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
history and news headlines for the 51 tickers configured in `config.yaml` (the Nifty 50
plus BSE.NS — see "Ticker universe" below). If a fine-tuned sentiment model already
exists (see below), `refresh` also scores any new headlines automatically.

While developing, use `--tickers SYMBOL.NS ...` or `--limit N` (first N configured
tickers) to work against a small subset instead of all 51 — see "Ticker universe."

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
python -m market_pred.pipeline prices --limit 5                     # first N configured tickers (dev testing)
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

## Ticker universe

`config.yaml`'s `tickers:` list (51 entries: the Nifty 50 as of 2026-09-24, plus
`BSE.NS` added ahead of its officially-confirmed 2026-09-30 index entry — both it and
`WIPRO.NS`, which is leaving the index that same day, are kept rather than dropping one
a week early) is the **single source of truth** for every consumer — price/news
ingestion, sentiment scoring, modeling, and the dashboard's ticker dropdown all read it
via `get_settings().tickers`. There's no second place to update.

**Index membership drifts over time** (rebalanced roughly semi-annually) — this list is
a snapshot, not something that stays current automatically. Expect to revisit it
periodically rather than treating it as permanent.

**Rate-limit budget at 51 tickers**: verified live — a steady-state `refresh`/`news` run
costs almost exactly 1 NewsAPI request per ticker (the free tier's 100-result cap
already forces one page per ticker regardless of config), so ~51 requests/run,
comfortably under both the `max_requests_per_run: 90` safety cap and NewsAPI's 100/day
account cap **for one run**. That cap resets per invocation, not per day — running a
full 51-ticker `refresh`/`news` more than once or twice in the same day will exceed the
account's actual daily quota. Use `--tickers`/`--limit` with a small subset while
iterating during development; reserve full-universe runs to once or twice a day.

**News relevance at scale — verified, not assumed**: the same manual headline-review
process that caught the original RELIANCE noise bug (see "News query design" above) was
re-run against the riskiest-looking new tickers before trusting the expansion. Most held
up well — `ETERNAL.NS` (Zomato/Blinkit's new legal name, a plain English word),
`TRENT.NS`, and `TMPV.NS` (the October 2025 Tata Motors passenger-vehicle demerger, a
name that could plausibly collide with its separately-listed sibling `TMCV.NS`) all came
back essentially 100% on-topic. Two did not, in ways not predicted in advance:
- **`BSE.NS`** is the worst case found: roughly 85% of its headlines are generic press
  use of "BSE" as shorthand for the Bombay Stock Exchange itself (index names like "BSE
  Information Technology index", unrelated companies' listings on the BSE SME platform,
  market-holiday notices) rather than anything about BSE Ltd (the company) specifically.
  Same root cause as the original RELIANCE bug — the bare symbol is OR'd into every
  query — just a more severe collision, since nearly every Indian markets article
  mentions "BSE" as the venue.
- **`ITC.NS`** is a smaller version of the same problem (~30% off-topic): "ITC" collides
  with "Input Tax Credit" (a routine GST/tax term) and, in press-agency wire content,
  occasionally the unrelated US "International Trade Commission."

Not fixed here — per-ticker query overrides would need the same care the original
RELIANCE fix got (iteration + re-verification), and this project's existing domain
allowlist + `qInTitle` restriction already prevented what would otherwise be far worse.
Treat `BSE.NS`'s (and to a lesser extent `ITC.NS`'s) sentiment/headline data as
noisier than the rest of the universe until specifically revisited.

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
year, split by date across all pooled tickers at once so no fold's boundary can leak
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
(2020–2026) over 106,222 pooled rows (51 tickers), averaged:

| Model | Accuracy | Naive ("always up") | Strategy mean return | Buy-and-hold mean return |
|---|---|---|---|---|
| Logistic regression | 50.6% | 50.9% | 0.00051 | 0.00085 |
| XGBoost | 50.5% | 50.9% | 0.00051 | 0.00085 |

At 10x the pooled data (up from 5 tickers/10,695 rows), both models now land almost
exactly on the naive baseline — neither shows a real edge. (An earlier run on just 5
tickers had shown XGBoost slightly ahead of naive; that gap didn't survive scaling up,
which is itself informative — it was more likely 5-ticker noise than a real effect.)
This is the expected, credible outcome for next-day direction from technical indicators
alone on liquid large-caps, not a sign of a bug — see the literature comparison earlier
in this project's history for context on what a rigorous walk-forward setup on this
problem typically finds. Neither result accounts for transaction costs or slippage, so
read the strategy-return figures as illustrative, not a claim that this is tradeable.

**Sentiment ablation** (same price-only vs. price+sentiment features, evaluated on the
overlap window where sentiment actually exists): expanding to 51 tickers grew this
window from 52 rows (38 train / 14 test, 20 dates) to **239 rows (164 train / 75 test,
22 dates)** — directly realizing one of the intended benefits of the ticker expansion.
With more data, the result also changed direction: price-only scored 53.3% accuracy vs.
**56.0% with sentiment added** — a small positive gap, where the earlier 14-test-row
version had (noisily) shown the opposite. 75 test rows is still a modest sample and this
is still not strong evidence either way — but it's a more credible small sample than
before, and the direction is now at least consistent with the project's original
premise. A firmer answer keeps needing more overlapping history, which only accumulates
as `refresh` keeps running past NewsAPI's lookback window.

## Dashboard

A Streamlit app (`streamlit_app.py`, repo root) that's **read-only by default** against
whatever's currently in `data/market_pred.db` and `models/`. The sidebar's **"Refresh
data & retrain"** button is the one exception: it runs the same ingestion + sentiment
scoring + `train-model` pipeline in-process, clears the dashboard's caches, and reruns
the page when it finishes — no separate terminal needed. It needs `NEWSAPI_KEY` and (if
a fine-tuned sentiment model exists) `torch`/`transformers` to fully succeed, can take up
to a minute, and is subject to the same NewsAPI rate limits as the CLI (a 429 mid-run is
logged and skipped per ticker, not fatal — see "Known limitations"). The sidebar also
shows "Model last trained" so it's always clear how fresh the current prediction is. The
CLI commands below still work exactly as before, for a scripted/scheduled refresh instead
of a manual button click — either way, a plain "Clear cache" button remains for picking up
a CLI-driven change without restarting the process.

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

**Selecting a ticker**: either the sidebar dropdown (any configured ticker) or the
**Top Movers** list below it — the 15 biggest gainers or losers by the most recent
day's % close-to-close change (toggle via the Gainers/Losers filter), each showing its
% change next to its name. Computed live from `prices` (one query across every ticker,
via `get_latest_daily_changes`), so it's never a stale/precomputed snapshot — it reflects
whatever's currently in the DB and updates automatically whenever that changes (rerun
`prices`/`refresh`, then use "Clear cache" so the dashboard picks it up without a
restart). Clicking an entry sets it as the active ticker, same as picking it from the
dropdown — both stay in sync via `st.session_state`.

**What each panel shows**, once a ticker is selected:
- **Price** — candlestick chart (split/dividend-adjusted close), defaulting to the last
  ~6 months with the full history reachable via the range slider.
- **News sentiment trend** — `daily_sentiment`'s calendar-day aggregate (naturally
  covers only the last ~29 days per NewsAPI's free tier, as above) plus a recent-headlines
  table with each one's FinBERT label.
- **Model prediction** — the persisted price-only XGBoost model's call and confidence for
  the next actual NSE trading session (named by date, via `pandas_market_calendars`'s NSE
  calendar — holidays like Diwali/Republic Day are excluded, not just weekends), with an
  expander showing its own walk-forward accuracy (~50.5% vs. ~50.9% naive) so a single
  prediction isn't over-trusted. Deliberately **not** sentiment-fused, matching the
  sentiment ablation finding above — only the price-only model was persisted.

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
- The dashboard's "Refresh data & retrain" button runs the pipeline synchronously
  in-process (a spinner covers the wait) — there's no background job queue, so the
  browser tab needs to stay open until it finishes, and only one refresh can run at a
  time. Fine at this project's single-user scale; would need a real task queue to serve
  multiple concurrent users.
- Feature engineering (`assign_effective_trading_day`) and walk-forward fold boundaries
  were never affected by the old weekend-only gap inference — both already derive
  trading days from the actual dates present in price history, not a calendar guess.
  The NSE calendar (`pandas_market_calendars`) is used only for the dashboard's
  human-readable "next session" date, a display concern, not a modeling one.
- A ticker with too little trailing price history (fewer than ~20 trading days) can't
  produce a prediction; the dashboard shows a clean message rather than crashing. Not
  expected for any of the 51 configured tickers (8 years of history each), but would
  apply to a newly added ticker.

## What I learned

- Lookahead bias is easy to miss even when you're looking for it. A headline published at
  4pm didn't exist yet when the market closed — `assign_effective_trading_day()` rolls it
  forward to the next trading day instead of naively joining by calendar date.
- Testing on one example isn't testing. I "fixed" a news-search bug for RELIANCE.NS,
  skimmed a few headlines, called it done — and missed that the same bug was polluting
  every other ticker too, just less visibly.
- ~50% accuracy isn't the model failing. For next-day direction from technical indicators
  alone, that's the believable outcome — a much higher number would've worried me more.
- Small samples can flip conclusions entirely. The sentiment-vs-no-sentiment result
  reversed between 52 rows (5 tickers) and 239 rows (51 tickers), same method.
- I'm bad at predicting where bugs will come from. I expected renamed tickers like
  ETERNAL.NS to cause news-matching noise; they were fine. The real offenders — BSE.NS,
  ITC.NS — weren't on my radar at all.
- Free API tiers shape the design from day one, not after the fact. NewsAPI's 100
  requests/day and ~29-day lookback meant the ingestion pipeline had to be incremental
  and quota-aware from the start.
- Libraries fail in ways docs don't warn you about. XGBoost's sklearn wrapper raises if a
  DataFrame's columns aren't in the exact fitted order — it won't quietly realign by name.

## Running tests

```powershell
pytest
```

All tests mock `yfinance`/`requests` and use a temp SQLite DB — no network calls or real
API keys are needed to run the test suite.
