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

from market_pred.config import Settings, get_settings
from market_pred.db.access import (
    get_daily_sentiment,
    get_last_news_date,
    get_last_price_date,
    get_latest_daily_changes,
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

# Status colors (mode-invariant by design in the dataviz palette -- same hex
# in light and dark). The sentiment chart's two lines deliberately do NOT get
# hardcoded colors here -- they're left unset so Plotly assigns them from the
# theme's chartCategoricalColors (.streamlit/config.toml), which gives correct
# per-mode colors (light vs dark step) for free.
GOOD_COLOR = "#0ca30c"
CRITICAL_COLOR = "#d03b3b"
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


@st.cache_data
def _load_daily_changes() -> pd.DataFrame:
    settings = get_settings()
    with get_connection(settings.db_path) as conn:
        return get_latest_daily_changes(conn)


def render_top_movers(settings: Settings) -> None:
    st.subheader("Top Movers")
    changes = _load_daily_changes()
    if changes.empty:
        st.caption("Not enough price history yet.")
        return

    company_by_symbol = {t.symbol: t.company for t in settings.tickers}
    changes = changes[changes["ticker"].isin(company_by_symbol)].copy()
    changes["company"] = changes["ticker"].map(company_by_symbol)

    direction = st.segmented_control(
        "Filter", ["Gainers", "Losers"], default="Gainers", required=True, key="movers_filter"
    )
    top = changes.sort_values("pct_change", ascending=(direction == "Losers")).head(15)

    for row in top.itertuples():
        arrow = "▲" if row.pct_change >= 0 else "▼"
        swatch = "green" if row.pct_change >= 0 else "red"
        name_col, pct_col = st.columns([5, 2])
        with name_col:
            if st.button(row.company, key=f"mover_{row.ticker}", width="stretch"):
                st.session_state["ticker_symbol"] = row.ticker
                st.rerun()
        with pct_col:
            st.markdown(f":{swatch}[{arrow}{row.pct_change:+.1f}%]")


def render_price_panel(ticker: str) -> None:
    st.header("💰 Price")
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
        showlegend=False,
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
    st.header("📰 News Sentiment Trend")
    sentiment = _load_daily_sentiment(ticker)
    if sentiment.empty:
        st.info(
            "No scored news yet for this ticker. Run `python -m market_pred.pipeline news` "
            "then `sentiment` (after `train-sentiment` once)."
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sentiment["date"], y=sentiment["mean_finbert_score"],
            mode="lines+markers", name="FinBERT (fine-tuned)",
            customdata=sentiment["article_count"],
            hovertemplate="FinBERT: %{y:.3f}<br>Articles: %{customdata}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=sentiment["date"], y=sentiment["mean_vader_score"],
            mode="lines+markers", name="VADER (baseline)", opacity=0.6,
            hovertemplate="VADER: %{y:.3f}<extra></extra>",
        ))
        fig.update_yaxes(zeroline=True, zerolinewidth=1)
        fig.update_layout(
            height=350,
            margin=dict(t=20, b=20),
            yaxis_title="Mean sentiment score",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig)
        st.caption(
            "Grouped by naive calendar day, for display only -- not the lookahead-safe "
            "aggregation the prediction model uses. Typically covers the last ~29 days "
            "(NewsAPI free-tier lookback)."
        )
        with st.expander("Raw sentiment data"):
            st.dataframe(sentiment, hide_index=True)

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
    st.header("🎯 Model Prediction")
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
            col1, col2 = st.columns(2)
            col1.metric("XGBoost accuracy (walk-forward)", f"{summary['accuracy']:.1%}")
            col2.metric("Naive \"always up\" baseline", f"{summary['naive_always_up_accuracy']:.1%}")
            st.caption(
                "7 expanding-window folds, 2020-2026, pooled across all configured tickers. A "
                "small, honest edge -- not a reliable trading signal. \"Down\" includes flat/zero-return days."
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

    symbols = [t.symbol for t in settings.tickers]
    if st.session_state.get("ticker_symbol") not in symbols:
        st.session_state["ticker_symbol"] = symbols[0]

    with st.sidebar:
        st.header("Settings")
        current_index = symbols.index(st.session_state["ticker_symbol"])
        ticker_info = st.selectbox(
            "Ticker", settings.tickers, index=current_index,
            format_func=lambda t: f"{t.company} ({t.symbol})",
        )
        st.session_state["ticker_symbol"] = ticker_info.symbol

        last_price_date, last_news_date = _load_freshness(st.session_state["ticker_symbol"])
        st.caption(f"Prices through {last_price_date or '—'} · News through {last_news_date or '—'}")

        if st.button("🔄 Clear cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.divider()
        render_top_movers(settings)

    ticker = st.session_state["ticker_symbol"]

    render_price_panel(ticker)
    st.divider()
    render_sentiment_panel(ticker)
    st.divider()
    render_prediction_panel(ticker, settings)


main()
