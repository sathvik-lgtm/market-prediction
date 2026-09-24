"""NSE Market Sentiment + Direction Predictor dashboard.

    streamlit run streamlit_app.py

Read-only against whatever's currently in data/market_pred.db and models/ --
does not trigger ingestion, sentiment scoring, or training itself. Those stay
CLI-only (python -m market_pred.pipeline refresh / train-sentiment / sentiment
/ train-model), run externally; use the "Clear cache" button afterward so the
dashboard picks up the change without restarting the process.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_pred.config import get_settings
from market_pred.db.access import (
    get_daily_sentiment,
    get_last_news_date,
    get_last_price_date,
    get_news_for_ticker,
    get_price_history,
)
from market_pred.db.connection import get_connection
from market_pred.modeling.predict import (
    ModelNotFoundError,
    get_latest_features,
    load_final_model,
    load_report_summary,
    predict_direction,
)

st.set_page_config(page_title="NSE Market Sentiment + Direction Predictor", page_icon="📈", layout="wide")

GOOD_COLOR = "#0ca30c"
CRITICAL_COLOR = "#d03b3b"
NEUTRAL_COLOR = "#6b7280"
FINBERT_COLOR = "#2a78d6"
VADER_COLOR = "#eb6834"
LABEL_EMOJI = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}


@st.cache_resource
def _load_model():
    return load_final_model(get_settings())


@st.cache_data
def _load_price_history(ticker: str) -> pd.DataFrame:
    settings = get_settings()
    with get_connection(settings.db_path) as conn:
        return get_price_history(conn, ticker)


@st.cache_data
def _load_daily_sentiment(ticker: str) -> pd.DataFrame:
    settings = get_settings()
    with get_connection(settings.db_path) as conn:
        return get_daily_sentiment(conn, ticker)


@st.cache_data
def _load_recent_news(ticker: str) -> pd.DataFrame:
    settings = get_settings()
    with get_connection(settings.db_path) as conn:
        return get_news_for_ticker(conn, ticker)


@st.cache_data
def _load_latest_features(ticker: str) -> pd.Series | None:
    settings = get_settings()
    with get_connection(settings.db_path) as conn:
        return get_latest_features(conn, ticker)


@st.cache_data
def _load_freshness(ticker: str) -> tuple[str | None, str | None]:
    settings = get_settings()
    with get_connection(settings.db_path) as conn:
        last_price = get_last_price_date(conn, ticker)
        last_news = get_last_news_date(conn, ticker)
    return (
        last_price.isoformat() if last_price else None,
        last_news.strftime("%Y-%m-%d") if last_news else None,
    )


def render_price_panel(ticker: str) -> None:
    st.header("Price")
    prices = _load_price_history(ticker)
    if prices.empty:
        st.info("No price history for this ticker yet.")
        return

    fig = go.Figure(data=[go.Candlestick(
        x=prices["date"], open=prices["open"], high=prices["high"],
        low=prices["low"], close=prices["close"],
        increasing_line_color=GOOD_COLOR, increasing_fillcolor=GOOD_COLOR,
        decreasing_line_color=CRITICAL_COLOR, decreasing_fillcolor=CRITICAL_COLOR,
        name=ticker,
    )])
    max_date = prices["date"].max()
    cutoff = (pd.to_datetime(max_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    fig.update_layout(
        xaxis_rangeslider_visible=True,
        xaxis_range=[cutoff, max_date],
        height=450,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig)
    st.caption("Close is split/dividend-adjusted. Defaults to the last ~6 months -- drag the slider below the chart to see the full history.")

    with st.expander("Raw price data"):
        st.dataframe(prices, hide_index=True)


def render_sentiment_panel(ticker: str) -> None:
    st.header("News Sentiment Trend")
    sentiment = _load_daily_sentiment(ticker)
    if sentiment.empty:
        st.info(
            "No scored news yet for this ticker. Run `python -m market_pred.pipeline news` "
            "then `sentiment` (after `train-sentiment` once)."
        )
    else:
        fig = go.Figure()
        fig.add_hline(y=0, line_color=NEUTRAL_COLOR, line_width=1)
        fig.add_trace(go.Scatter(
            x=sentiment["date"], y=sentiment["mean_finbert_score"],
            mode="lines+markers", name="FinBERT (fine-tuned)",
            line=dict(color=FINBERT_COLOR),
            customdata=sentiment["article_count"],
            hovertemplate="%{x}<br>FinBERT: %{y:.3f}<br>Articles: %{customdata}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=sentiment["date"], y=sentiment["mean_vader_score"],
            mode="lines+markers", name="VADER (baseline)",
            line=dict(color=VADER_COLOR), opacity=0.6,
        ))
        fig.update_layout(height=350, margin=dict(t=20, b=20), yaxis_title="Mean sentiment score")
        st.plotly_chart(fig)
        st.caption(
            "Grouped by naive calendar day, for display only -- not the lookahead-safe "
            "aggregation the prediction model uses. Typically covers the last ~29 days "
            "(NewsAPI free-tier lookback)."
        )

    st.subheader("Recent headlines")
    news = _load_recent_news(ticker)
    if news.empty:
        st.info("No news headlines for this ticker yet.")
        return

    recent = news.sort_values("published_at_utc", ascending=False).head(20).copy()
    recent["Sentiment"] = recent["finbert_label"].apply(
        lambda label: f"{LABEL_EMOJI.get(label, '')} {label}" if pd.notna(label) else "not yet scored"
    )
    display = recent[["published_at_utc", "source_name", "title", "Sentiment"]]
    display.columns = ["Published (UTC)", "Source", "Headline", "Sentiment"]
    st.dataframe(display, hide_index=True, width="stretch")


def render_prediction_panel(ticker: str, settings) -> None:
    st.header("Model Prediction")
    try:
        model = _load_model()
    except ModelNotFoundError as e:
        st.warning(str(e))
        return

    features_row = _load_latest_features(ticker)
    if features_row is None:
        st.info("Not enough price history yet for this ticker to generate a prediction.")
        return

    label, confidence = predict_direction(model, features_row)
    arrow = "▲" if label == "Up" else "▼"
    swatch = "green" if label == "Up" else "red"

    with st.container(border=True):
        st.markdown(f"### :{swatch}[{arrow} {label}] — {confidence:.0%} confidence")
        st.caption(
            f"Based on data through {features_row['date']} "
            f"(last close ₹{features_row['close']:.2f}) — predicts the next trading "
            "session's direction. Price-only model, not sentiment-fused (see README's "
            "sentiment ablation results for why)."
        )

    summary = load_report_summary(settings)
    if summary:
        with st.expander("Model accuracy (walk-forward validated)"):
            st.write(
                f"XGBoost walk-forward accuracy: **{summary['accuracy']:.1%}** vs. "
                f"naive \"always predict up\" baseline: **{summary['naive_always_up_accuracy']:.1%}** "
                "(7 expanding-window folds, 2020-2026, pooled across all 5 tickers). "
                "A small, honest edge -- not a reliable trading signal. \"Down\" includes "
                "flat/zero-return days."
            )


def main() -> None:
    settings = get_settings()

    if not settings.db_path.exists():
        st.error(
            f"No database found at `{settings.db_path}`. Run "
            "`python scripts/load_seed_data.py` (no API key needed) or "
            "`python -m market_pred.pipeline refresh` first."
        )
        st.stop()

    st.title("📈 NSE Market Sentiment + Direction Predictor")

    with st.sidebar:
        st.header("Settings")
        ticker_info = st.selectbox(
            "Ticker", settings.tickers, format_func=lambda t: f"{t.company} ({t.symbol})"
        )
        ticker = ticker_info.symbol

        last_price_date, last_news_date = _load_freshness(ticker)
        st.caption(f"Prices through {last_price_date or '—'} · News through {last_news_date or '—'}")

        if st.button("🔄 Clear cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    render_price_panel(ticker)
    render_sentiment_panel(ticker)
    render_prediction_panel(ticker, settings)


main()
