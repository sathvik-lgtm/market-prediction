# Project: Market Sentiment + Price Movement Predictor

## Overview
Build an end-to-end data science project that predicts short-term stock price movement direction (up/down) for NSE-listed Indian stocks by combining financial news sentiment (via FinBERT) with historical price/technical data. Target audience: resume/portfolio piece for a 3rd-year data science student. Full semester timeline (~14 weeks).

## Goals
- Demonstrate end-to-end ML skills: data acquisition, NLP, tabular modeling, rigorous evaluation, deployment.
- Avoid common finance-ML mistakes (lookahead bias, random train/test splits on time series).
- Produce a working, demoable app (not just a notebook).
- Produce a clear README/write-up suitable for a portfolio and interview discussion.

## Target stocks (initial scope)
Start with 3-5 liquid NSE tickers, e.g.: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS

## Tech stack
- Language: Python
- Data: `yfinance` (price/volume), NewsAPI or Google News RSS feeds (news headlines), optionally `praw` (Reddit) for social sentiment
- NLP: HuggingFace `transformers` (FinBERT for sentiment)
- Modeling: `scikit-learn`, `xgboost`, optionally `pytorch` for an LSTM/sequence model
- Storage: SQLite or Parquet files
- Deployment: Streamlit dashboard
- Versioning: Git, with commits at each milestone

## Phase 1 — Data Pipeline (Weeks 1-3)
- Set up repo structure, README with project scope, Git init
- Build price/volume ingestion using `yfinance` for target tickers (historical daily data)
- Build news ingestion pipeline (NewsAPI free tier or Google News RSS) pulling headlines per ticker/company name
- (Optional) Reddit ingestion via `praw` for r/IndianStreetBets, r/IndiaInvestments
- Design and implement a clean storage schema: date, ticker, headline/text, source, price, volume
- Store in SQLite (or Parquet files) with a simple data access layer

## Phase 2 — Sentiment Modeling (Weeks 4-7)
- Baseline sentiment: VADER or TextBlob on headlines (quick floor/benchmark)
- Real model: FinBERT (pretrained, or fine-tuned if feasible) for financial sentiment classification
- Aggregate sentiment per ticker per day (mean or weighted daily sentiment score)
- Store aggregated sentiment scores alongside price data

## Phase 3 — Predictive Modeling (Weeks 8-11)
- Feature engineering: lagged returns, moving averages, volume changes, volatility, daily sentiment score(s)
- Define target: next-day (or next-N-day) price direction (up/down) classification
- Baseline models: logistic regression, XGBoost
- Advanced (stretch): LSTM or small transformer on the time series
- **Critical**: use walk-forward validation (not random split) to avoid lookahead bias
- Compare against naive baselines: "always predict up", buy-and-hold
- Evaluate with appropriate metrics (accuracy, precision/recall, and ideally a trading-relevant metric like simulated returns vs baseline)

## Phase 4 — Deployment (Weeks 12-14)
- Build a Streamlit dashboard: select a ticker, view recent news sentiment trend, price chart, and model prediction with confidence
- Polish UI/UX for demo purposes
- Ensure the app runs cleanly from a fresh clone (requirements.txt, setup instructions)

## Phase 5 — Write-up (Final 1-2 weeks)
- Write a clear README covering: problem statement, data sources, methodology, results, limitations, what was learned
- Optionally write a short blog-style post summarizing the project for a portfolio/LinkedIn
- Prepare a concise resume bullet point summarizing the project

## Success criteria / "Definition of done"
- Working data pipeline that can be re-run to refresh data
- Sentiment scores generated via FinBERT and stored
- At least 2 predictive models trained and rigorously backtested with walk-forward validation
- Model performance compared against naive baselines, with honest discussion of results
- Deployed, working Streamlit demo
- Clean repo with README, requirements.txt, and clear commit history

## Notes for Claude Code
- Work phase by phase — do not attempt to scaffold all phases at once.
- Confirm/review data schema before building modeling code on top of it.
- Flag any lookahead bias risks explicitly when designing train/test splits.
- Prioritize a working end-to-end pipeline (even if simple) over a highly polished but incomplete single phase.
