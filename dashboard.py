from __future__ import annotations


import warnings
warnings.filterwarnings("ignore", message=".*keyword arguments have been deprecated.*")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

import sqlite3
from pathlib import Path

from config import (
    load_config, get_all_tickers, get_tickers_by_sector,
    get_tickers_by_subsector, get_ticker_metadata, get_indicator_config,
)
from data_fetcher import fetch_all, fetch_sector, fetch_ticker, data_summary
from indicators import (
    compute_all_indicators, score_ticker, score_all,
    INDICATOR_WEIGHTS, INDICATOR_LABELS, MAX_SCORE,
)
from subsector_breakout import (
    compute_subsector_metrics,
    run_breakout_detection,
)
from subsector_store import get_breakout_states, init_db

DB_PATH = Path(__file__).parent / "breakout_tracker.db"


def get_pct_since_breakout(tickers: list, data: dict, threshold: float = 9.0) -> dict:
    """For each ticker, find the most recent date it crossed FROM below threshold
    TO >= threshold, get the close price on that date, and return % change."""
    result = {}
    if not tickers or not data:
        return result
    try:
        conn = sqlite3.connect(DB_PATH)
        for ticker in tickers:
            # Get full score history ordered by date
            rows = conn.execute(
                """SELECT date, score FROM ticker_scores
                   WHERE ticker = ?
                   ORDER BY date ASC""",
                (ticker,),
            ).fetchall()
            if not rows:
                continue
            # Find the most recent transition: previous score < threshold, current >= threshold
            crossing_date = None
            for i in range(1, len(rows)):
                prev_score = rows[i - 1][1]
                curr_score = rows[i][1]
                if prev_score < threshold and curr_score >= threshold:
                    crossing_date = rows[i][0]
            # Also check if the very first record is >= threshold (initial entry)
            if crossing_date is None and rows[0][1] >= threshold:
                crossing_date = rows[0][0]
            if crossing_date is None:
                continue
            ticker_df = data.get(ticker)
            if ticker_df is None or ticker_df.empty:
                continue
            breakout_dt = pd.Timestamp(crossing_date)
            # Match timezone of the price data index
            if ticker_df.index.tz is not None:
                breakout_dt = breakout_dt.tz_localize(ticker_df.index.tz)
            idx = ticker_df.index.get_indexer([breakout_dt], method="nearest")[0]
            if idx < 0 or idx >= len(ticker_df):
                continue
            breakout_price = ticker_df["Close"].iloc[idx]
            current_price = ticker_df["Close"].iloc[-1]
            if breakout_price > 0:
                result[ticker] = {
                    "pct_change": (current_price - breakout_price) / breakout_price * 100,
                    "breakout_date": crossing_date,
                    "breakout_price": breakout_price,
                }
        conn.close()
    except Exception:
        pass
    return result


# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="Alpha Scanner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================
# SCORE THRESHOLDS (for the 0-10.0 scale)
# =============================================================
# Fire:     9+   (top breakout setup)
# Strong:   7-9  (strong signal)
# Moderate: 5-7  (moderate signal)
# Weak:     <5   (minimal signals)

def score_color(score: float) -> str:
    """Return a hex color for the score."""
    if score >= 9:
        return "#ef4444"  # red (fire)
    elif score >= 7:
        return "#10b981"  # green
    elif score >= 5:
        return "#eab308"  # yellow
    else:
        return "#6b7280"  # gray


def score_tier(score: float) -> str:
    """Return a tier label for the score."""
    if score >= 9:
        return "fire"
    elif score >= 7:
        return "strong"
    elif score >= 5:
        return "moderate"
    else:
        return "weak"


# =============================================================
# CUSTOM STYLING
# =============================================================
st.markdown("""
<style>
    /* Tighten up spacing */
    .block-container { padding-top: 1rem; }

    /* Score badge styling — tier-based */
    .score-fire { background-color: rgba(239, 68, 68, 0.75); color: white; padding: 4px 12px; border-radius: 12px; font-weight: 700; }
    .score-strong { background-color: rgba(16, 185, 129, 0.75); color: white; padding: 4px 12px; border-radius: 12px; font-weight: 700; }
    .score-moderate { background-color: rgba(234, 179, 8, 0.75); color: black; padding: 4px 12px; border-radius: 12px; font-weight: 700; }
    .score-weak { background-color: rgba(107, 114, 128, 0.75); color: white; padding: 4px 12px; border-radius: 12px; font-weight: 700; }

    /* Signal pill styling */
    .signal-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 0.8em;
        margin: 1px;
    }
    .signal-pill-primary {
        background-color: #dbeafe;
        color: #1e40af;
    }
    .signal-pill-confirm {
        background-color: #f3e8ff;
        color: #6b21a8;
    }

    /* Status tooltip */
    .status-tooltip {
        cursor: default;
    }
    .status-tooltip .status-tooltiptext {
        visibility: hidden;
        opacity: 0;
        background-color: #1e293b;
        color: #cbd5e1;
        font-size: 0.8rem;
        font-weight: 400;
        text-align: center;
        padding: 6px 10px;
        border-radius: 6px;
        position: absolute;
        top: 110%;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
        z-index: 10;
        transition: opacity 0.2s;
    }
    .status-tooltip:hover .status-tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)


# Indicators classified as primary vs confirmation
PRIMARY_INDICATORS = {"relative_strength", "ichimoku_cloud", "higher_lows", "cmf", "roc"}
CONFIRMATION_INDICATORS = {"dual_tf_rs", "atr_expansion"}

SIGNAL_PILL_LABELS = {
    "relative_strength": "💪 Rel Strength",
    "higher_lows":       "📶 Higher Lows",
    "ichimoku_cloud":    "☁️ Ichimoku",
    "cmf":               "💰 CMF",
    "roc":               "🚀 ROC",
    "dual_tf_rs":        "🔀 Dual-TF RS",
    "atr_expansion":     "📊 ATR Expand",
}


# =============================================================
# PRICE CHART
# =============================================================
def render_price_chart(ticker: str, df: pd.DataFrame):
    """Render an interactive candlestick chart with SMA overlays and volume."""
    if df is None or df.empty:
        st.warning(f"No price data available for {ticker}")
        return

    # Filter to last 4 months for display
    df = df.copy()
    cutoff = df.index.max() - pd.Timedelta(days=120)
    df_display = df[df.index >= cutoff].copy()
    # Compute moving averages on full data, then filter
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df_display["SMA_50"] = df["SMA_50"]
    df_display["SMA_200"] = df["SMA_200"]
    df = df_display

    # Create subplots: candlestick + volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
            increasing_line_color="#10b981", decreasing_line_color="#ef4444",
        ),
        row=1, col=1,
    )

    # 50-day SMA
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["SMA_50"], name="50 SMA",
            line=dict(color="#3b82f6", width=1.5),
        ),
        row=1, col=1,
    )

    # 200-day SMA
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["SMA_200"], name="200 SMA",
            line=dict(color="#ef4444", width=1.5, dash="dot"),
        ),
        row=1, col=1,
    )

    # Volume bars
    colors = [
        "#10b981" if c >= o else "#ef4444"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume",
               marker_color=colors, opacity=0.5),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{ticker} — 4 Months",
        height=450,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# =============================================================
# DATA LOADING (cached)
# =============================================================
@st.cache_data(ttl=3600, show_spinner="Fetching market data...")
def load_and_score(period: str = "1y"):
    """Fetch data and score all tickers. Cached for 1 hour."""
    cfg = load_config()
    t_fetch_start = datetime.now()
    data = fetch_all(cfg, period=period, verbose=False)
    t_fetch_end = datetime.now()
    results = score_all(data, cfg)
    t_score_end = datetime.now()
    timestamps = {
        "price_data": t_fetch_end.strftime("%m/%d/%y %H:%M"),
        "scoring": t_score_end.strftime("%m/%d/%y %H:%M"),
    }
    return cfg, data, results, timestamps


@st.cache_data(ttl=3600, show_spinner="Fetching sector data...")
def load_sector_data(sector_key: str, period: str = "1y"):
    """Fetch data for a specific sector."""
    cfg = load_config()
    data = fetch_sector(cfg, sector_key, period=period, verbose=False)
    return data


# =============================================================
# HELPER FUNCTIONS
# =============================================================
def score_badge(score: float) -> str:
    """Return an HTML badge for the score."""
    tier = score_tier(score)
    return f'<span class="score-{tier}">{score}/{MAX_SCORE}</span>'


def signal_pills(signals: list, signal_weights: dict = None) -> str:
    """Return HTML pills for each triggered signal, colored by tier."""
    if not signals:
        return '<span style="color: #9ca3af;">—</span>'
    pills = []
    for s in signals:
        label = SIGNAL_PILL_LABELS.get(s, s)
        css_class = "signal-pill-primary" if s in PRIMARY_INDICATORS else "signal-pill-confirm"
        # Show actual contributed weight (gradient/conditional may vary from max)
        weight = (signal_weights or {}).get(s, INDICATOR_WEIGHTS.get(s, 0))
        pills.append(f'<span class="signal-pill {css_class}">{label} ({weight})</span>')
    return " ".join(pills)


def get_current_close(indicators: dict) -> float:
    """Extract the current close price from indicator results."""
    if indicators.get("moving_averages", {}).get("current_close"):
        return indicators["moving_averages"]["current_close"]
    if indicators.get("near_52w_high", {}).get("current_close"):
        return indicators["near_52w_high"]["current_close"]
    return 0


def results_to_dataframe(results: list) -> pd.DataFrame:
    """Convert results list to a clean DataFrame."""
    rows = []
    for r in results:
        ind = r["indicators"]
        rows.append({
            "Ticker": r["ticker"],
            "Name": r["name"],
            "Sector": r["sector"],
            "Subsector": r["subsector"],
            "Score": r["score"],
            "Signals": r["signals"],
            "Close": get_current_close(ind),
            # Primary indicators
            "RS Percentile": ind["relative_strength"]["rs_percentile"],
            "Ichimoku": ind["ichimoku_cloud"]["triggered"],
            "Higher Lows": ind["higher_lows"]["triggered"],
            "Above 50 SMA": ind["moving_averages"]["price_above_50"],
            "Above 200 SMA": ind["moving_averages"]["price_above_200"],
            "Golden Cross": ind["moving_averages"]["golden_cross_recent"],
            "ROC %": ind["roc"]["roc"],
            "CMF": ind["cmf"]["cmf"],
            "Dual-TF RS": ind["dual_tf_rs"]["triggered"],
            "ATR Pctl": ind["atr_expansion"]["atr_percentile"],
            # Display-only (not scored)
            "Above 50 SMA": ind["moving_averages"]["price_above_50"],
            "Above 200 SMA": ind["moving_averages"]["price_above_200"],
            "Golden Cross": ind["moving_averages"]["golden_cross_recent"],
            "% from 52w High": round(ind["near_52w_high"]["pct_from_high"] * 100, 2),
        })
    return pd.DataFrame(rows)


# =============================================================
# SIDEBAR
# =============================================================
def render_sidebar(cfg: dict, timestamps: dict = None):
    """Render the sidebar with navigation."""
    st.sidebar.title("🚀 Alpha Scanner")
    st.sidebar.divider()

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["🔥 Subsectors", "📈 Tickers", "📊 Historical Charts"],
        label_visibility="collapsed",
    )

    # Data freshness timestamps
    if timestamps:
        st.sidebar.divider()
        st.sidebar.markdown(
            f"<div style='font-size:0.9rem;color:#64748b;line-height:1.8'>"
            f"<strong style='color:#94a3b8;font-size:0.95rem'>Data Freshness</strong><br>"
            f"📊 Prices: {timestamps.get('price_data', '—')}<br>"
            f"🧮 Scores: {timestamps.get('scoring', '—')}<br>"
            f"🔥 Breakouts: {timestamps.get('breakout', '—')}<br>"
            f"🗄️ DB: {timestamps.get('db_last', '—')}"
            f"</div>",
            unsafe_allow_html=True,
        )

    return page


# =============================================================
# DASHBOARD PAGE
# =============================================================
def render_dashboard(results: list, cfg: dict, data: dict = None):
    """Tickers page — top signals table + ticker deep dive."""

    filtered = results

    # --- Header metrics ---
    st.title("📈 Tickers")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} · {len(results)} tickers scored · Max score: {MAX_SCORE}")

    fire = sum(1 for r in results if r["score"] >= 9)
    strong = sum(1 for r in results if 7 <= r["score"] < 9)
    moderate = sum(1 for r in results if 5 <= r["score"] < 7)
    weak = sum(1 for r in results if r["score"] < 5)

    st.markdown(
        f"<div style='background-color:#2d3348;padding:20px 16px;border-radius:10px;display:flex;justify-content:space-around;gap:8px'>"
        f"<div style='text-align:center;flex:1'>"
        f"<div style='font-size:1.1rem;color:#94a3b8'>Total Tickers</div>"
        f"<div style='font-size:2.2rem;font-weight:700;color:white'>{len(results)}</div>"
        f"</div>"
        f"<div style='text-align:center;flex:1'>"
        f"<div style='font-size:1.1rem;color:#94a3b8'>Fire (9+)</div>"
        f"<div style='font-size:2.2rem;font-weight:700;color:#ef4444'>{fire}</div>"
        f"</div>"
        f"<div style='text-align:center;flex:1'>"
        f"<div style='font-size:1.1rem;color:#94a3b8'>Strong (7-9)</div>"
        f"<div style='font-size:2.2rem;font-weight:700;color:#10b981'>{strong}</div>"
        f"</div>"
        f"<div style='text-align:center;flex:1'>"
        f"<div style='font-size:1.1rem;color:#94a3b8'>Moderate (5-7)</div>"
        f"<div style='font-size:2.2rem;font-weight:700;color:#eab308'>{moderate}</div>"
        f"</div>"
        f"<div style='text-align:center;flex:1'>"
        f"<div style='font-size:1.1rem;color:#94a3b8'>Weak (&lt;5)</div>"
        f"<div style='font-size:2.2rem;font-weight:700;color:#6b7280'>{weak}</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Breakouts table (7+) ---
    st.subheader("Breakouts")

    sorted_filtered = sorted(filtered, key=lambda r: r["score"], reverse=True)
    top = [r for r in sorted_filtered if r["score"] >= 7]

    if not top:
        st.info("No tickers scoring 7+ right now.")
    else:
        top_df = results_to_dataframe(top)

        # Add "% Since 9+" column
        top_tickers = top_df["Ticker"].tolist()
        breakout_data = get_pct_since_breakout(top_tickers, data, threshold=9.0)
        top_df["Δ Since Hot"] = top_df["Ticker"].map(
            lambda t: breakout_data[t]["pct_change"] if t in breakout_data else None
        )

        display_cols = [
            "Ticker", "Name", "Score", "Close", "Subsector", "Δ Since Hot",
            "RS Percentile", "Ichimoku", "Higher Lows",
            "ROC %", "CMF", "Dual-TF RS", "ATR Pctl",
            "Above 50 SMA", "Above 200 SMA", "% from 52w High",
        ]

        st.dataframe(
            top_df[display_cols],
            width="stretch",
            hide_index=True,
            height=min(460, len(top_df) * 35 + 40),
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=MAX_SCORE, format="%.1f",
                ),
                "Close": st.column_config.NumberColumn("Close", format="$%.2f"),
                "Δ Since Hot": st.column_config.NumberColumn("Δ Since Hot", format="%+.1f%%"),
                "RS Percentile": st.column_config.NumberColumn("RS %ile", format="%.0f"),
                "ROC %": st.column_config.NumberColumn("ROC %", format="%.1f%%"),
                "CMF": st.column_config.NumberColumn("CMF", format="%.4f"),
                "% from 52w High": st.column_config.NumberColumn("% from 52w High", format="%.1f%%"),
                "ATR Pctl": st.column_config.NumberColumn("ATR %ile", format="%.0f"),
            },
        )

    # --- Ticker deep-dive ---
    st.divider()
    st.subheader("Ticker Deep Dive")

    ticker_options = [f"{r['ticker']} — {r['name']} (Score: {r['score']})" for r in sorted_filtered]
    if ticker_options:
        selected = st.selectbox("Select a ticker for detailed breakdown", ticker_options)
        selected_ticker = selected.split(" — ")[0]

        result = next(r for r in filtered if r["ticker"] == selected_ticker)
        ind = result["indicators"]

        # Price chart (full width above the detail columns)
        if data and selected_ticker in data:
            render_price_chart(selected_ticker, data[selected_ticker])

        # Compact header
        st.markdown(
            f"**{result['ticker']} — {result['name']}** &nbsp;|&nbsp; "
            f"Sector: {result['sector']} &nbsp;|&nbsp; "
            f"Subsector: {result['subsector']}",
            unsafe_allow_html=True,
        )

        # Build indicator detail table
        sw = result.get("signal_weights", {})
        table_rows = []

        for ind_name, max_weight in INDICATOR_WEIGHTS.items():
            ind_data = ind.get(ind_name, {})
            actual_weight = sw.get(ind_name, 0)
            is_active = ind_name in result.get("signals", [])
            label = INDICATOR_LABELS.get(ind_name, ind_name)

            # Build detail string based on indicator type
            detail = ""
            if ind_name == "relative_strength":
                detail = (
                    f"Percentile: {ind_data.get('rs_percentile', 0):.0f}th · "
                    f"Stock: {ind_data.get('stock_return', 0)*100:+.1f}% · "
                    f"SPY: {ind_data.get('benchmark_return', 0)*100:+.1f}%"
                )
            elif ind_name == "higher_lows":
                detail = f"Consecutive: {ind_data.get('consecutive_higher_lows', 0)}"
            elif ind_name == "ichimoku_cloud":
                detail = (
                    f"Above cloud: {'Yes' if ind_data.get('above_cloud') else 'No'} · "
                    f"Bullish: {'Yes' if ind_data.get('cloud_bullish') else 'No'} · "
                    f"TK>KJ: {'Yes' if ind_data.get('tenkan_above_kijun') else 'No'}"
                )
            elif ind_name == "roc":
                detail = f"21d ROC: {ind_data.get('roc', 0):+.1f}%"
            elif ind_name == "cmf":
                detail = f"CMF: {ind_data.get('cmf', 0):+.4f}"
            elif ind_name == "dual_tf_rs":
                accel = "Yes" if ind_data.get("accelerating") else "No"
                detail = (
                    f"126d: {ind_data.get('rs_126d_percentile', 0):.0f}th · "
                    f"63d: {ind_data.get('rs_63d_percentile', 0):.0f}th · "
                    f"21d: {ind_data.get('rs_21d_percentile', 0):.0f}th · "
                    f"Accelerating: {accel}"
                )
            elif ind_name == "atr_expansion":
                detail = (
                    f"ATR: {ind_data.get('atr', 0):.4f} · "
                    f"Percentile: {ind_data.get('atr_percentile', 0):.0f}th"
                )
            elif ind_name == "moving_averages":
                gc = " · Golden Cross!" if ind_data.get("golden_cross_recent") else ""
                detail = (
                    f"50 SMA: ${ind_data.get('sma_50', 0):,.2f} · "
                    f"200 SMA: ${ind_data.get('sma_200', 0):,.2f}{gc}"
                )

            table_rows.append({
                "Status": "✅" if is_active else "❌",
                "Indicator": label,
                "Score": actual_weight if is_active else 0,
                "Max": max_weight,
                "Details": detail,
            })

        # Near 52w High — display only row
        n52w = ind.get("near_52w_high", {})
        if n52w.get("current_close"):
            detail = (
                f"Close: ${n52w.get('current_close', 0):,.2f} · "
                f"High: ${n52w.get('high_52w', 0):,.2f} · "
                f"Gap: {n52w.get('pct_from_high', 0)*100:+.1f}%"
            )
            table_rows.append({
                "Status": "📌",
                "Indicator": "Near 52w High",
                "Score": "—",
                "Max": "—",
                "Details": detail,
            })

        # Total row
        table_rows.append({
            "Status": "",
            "Indicator": "Total Score",
            "Score": result["score"],
            "Max": MAX_SCORE,
            "Details": "",
        })

        detail_df = pd.DataFrame(table_rows)

        def style_detail_table(styler):
            def highlight_total(row):
                if row["Indicator"] == "Total Score":
                    return ["font-weight: bold; background-color: #f1f5f9;"] * len(row)
                return [""] * len(row)
            styler.apply(highlight_total, axis=1)
            return styler

        styled = detail_df.style.pipe(style_detail_table).format(
            {"Score": lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x,
             "Max": lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x},
        )
        st.dataframe(styled, width="stretch", hide_index=True, height=(len(table_rows) + 1) * 35 + 10)

    # --- All Tickers table ---
    st.divider()
    st.subheader("All Tickers")

    indicator_keys = [
        "relative_strength", "higher_lows", "ichimoku_cloud",
        "cmf", "roc", "dual_tf_rs", "atr_expansion",
    ]
    indicator_short = {
        "relative_strength": "RS",
        "higher_lows": "HL",
        "ichimoku_cloud": "Ichimoku",
        "cmf": "CMF",
        "roc": "ROC",
        "dual_tf_rs": "Dual-TF",
        "atr_expansion": "ATR Exp",
    }

    # Filters
    all_sectors = sorted(set(r["sector"] for r in results))
    all_subsectors = sorted(set(r["subsector"] for r in results))
    score_options = ["All", "Fire (9+)", "Strong (7+)", "Moderate (5+)", "Weak (<5)"]

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        sel_sector = st.selectbox("Sector", ["All"] + all_sectors, key="all_tickers_sector")
    with fcol2:
        # Filter subsectors based on selected sector
        if sel_sector != "All":
            available_subs = sorted(set(r["subsector"] for r in results if r["sector"] == sel_sector))
        else:
            available_subs = all_subsectors
        sel_subsector = st.selectbox("Subsector", ["All"] + available_subs, key="all_tickers_subsector")
    with fcol3:
        sel_score = st.selectbox("Score", score_options, key="all_tickers_score")

    # Apply filters
    filtered_all = results
    if sel_sector != "All":
        filtered_all = [r for r in filtered_all if r["sector"] == sel_sector]
    if sel_subsector != "All":
        filtered_all = [r for r in filtered_all if r["subsector"] == sel_subsector]
    if sel_score == "Fire (9+)":
        filtered_all = [r for r in filtered_all if r["score"] >= 9]
    elif sel_score == "Strong (7+)":
        filtered_all = [r for r in filtered_all if r["score"] >= 7]
    elif sel_score == "Moderate (5+)":
        filtered_all = [r for r in filtered_all if r["score"] >= 5]
    elif sel_score == "Weak (<5)":
        filtered_all = [r for r in filtered_all if r["score"] < 5]

    filtered_all = sorted(filtered_all, key=lambda r: -r["score"])

    if not filtered_all:
        st.info("No tickers match the selected filters.")
    else:
        all_rows = []
        for r in filtered_all:
            close = get_current_close(r["indicators"])
            # 7-day price change
            pct_7d = None
            if data:
                df_ticker = data.get(r["ticker"])
                if df_ticker is not None and len(df_ticker) >= 6:
                    price_now = df_ticker["Close"].iloc[-1]
                    price_7d = df_ticker["Close"].iloc[-6]
                    pct_7d = (price_now - price_7d) / price_7d * 100

            sw = r.get("signal_weights", {})
            row = {
                "Ticker": r["ticker"],
                "Name": r["name"],
                "Subsector": r["subsector"],
                "Price": close if close else None,
                "7d Δ": pct_7d,
                "Score": r["score"],
            }
            for k in indicator_keys:
                row[indicator_short[k]] = sw.get(k, 0)
            all_rows.append(row)

        all_df = pd.DataFrame(all_rows)

        def color_score_all(val):
            if val >= 9:
                return "background-color: rgba(220, 38, 38, 0.75); color: white; font-weight: bold;"
            elif val >= 7:
                return "background-color: rgba(22, 163, 74, 0.75); color: white; font-weight: bold;"
            elif val >= 5:
                return "background-color: rgba(234, 179, 8, 0.75); color: white; font-weight: bold;"
            else:
                return "background-color: rgba(107, 114, 128, 0.75); color: white;"

        styled_all = all_df.style.map(
            color_score_all, subset=["Score"]
        ).format(
            {"Price": "${:,.2f}", "Score": "{:.1f}", "7d Δ": "{:+.1f}%",
             **{indicator_short[k]: "{:.1f}" for k in indicator_keys}},
            na_rep="—",
        )
        st.dataframe(
            styled_all,
            width="stretch",
            hide_index=True,
            height=min(700, len(all_df) * 35 + 40),
        )



# =============================================================
# SUBSECTOR BREAKOUTS PAGE
# =============================================================

# Display labels and styling for breakout states
BREAKOUT_STATUS_CONFIG = {
    "confirmed": {
        "label": "Confirmed Breakout",
        "emoji": "🔥",
        "color": "#10b981",
        "description": "Actionable — subsector has sustained hot breadth for 3+ readings",
        "priority": 1,
    },
    "emerging": {
        "label": "Emerging",
        "emoji": "👀",
        "color": "#fbbf24",
        "description": "Watching — breadth spiked but not yet confirmed",
        "priority": 2,
    },
    "fading": {
        "label": "Fading",
        "emoji": "📉",
        "color": "#f87171",
        "description": "Breadth declining from confirmed breakout",
        "priority": 3,
    },
    "steady_hot": {
        "label": "Steady Hot",
        "emoji": "🟢",
        "color": "#22c55e",
        "description": "Breadth ≥ 50% — consistently strong but not newly breaking out",
        "priority": 4,
    },
    "warming": {
        "label": "Warming",
        "emoji": "🟡",
        "color": "#eab308",
        "description": "Breadth 25-49% — showing some signs of life",
        "priority": 5,
    },
    "quiet": {
        "label": "Quiet",
        "emoji": "💤",
        "color": "#6b7280",
        "description": "Breadth < 25% — no breakout activity",
        "priority": 6,
    },
}


def compute_breakout_quality(breadth, avg_score, z_score, acceleration, is_reaccel=False):
    """
    Compute breakout quality grade (A/B/C) based on backtest findings.

    A-grade: breadth >= 75% AND avg_score >= 6 AND (acceleration > 0 OR revival)
    B-grade: breadth >= 50% AND avg_score >= 5
    C-grade: met minimum thresholds but marginal

    Returns: (grade, grade_color, grade_description)
    """
    acc_ok = (not np.isnan(acceleration) and acceleration > 0) if not np.isnan(acceleration) else False

    if is_reaccel:
        # Revivals automatically get at least B
        if breadth >= 0.75 and avg_score >= 6:
            return "A", "#10b981", "Strong revival — high breadth + high scores"
        return "B+", "#34d399", "Revival — recovery after pullback"

    if breadth >= 0.75 and avg_score >= 6 and acc_ok:
        return "A", "#10b981", "High conviction — strong breadth, scores, and acceleration"
    elif breadth >= 0.75 and avg_score >= 6:
        return "A-", "#34d399", "Strong breadth and scores"
    elif breadth >= 0.50 and avg_score >= 5 and acc_ok:
        return "B+", "#60a5fa", "Solid breadth with positive momentum"
    elif breadth >= 0.50 and avg_score >= 5:
        return "B", "#60a5fa", "Moderate conviction — meets core thresholds"
    elif breadth >= 0.50:
        return "B-", "#93c5fd", "Breadth above trigger but scores moderate"
    else:
        return "C", "#fbbf24", "Marginal — barely meets thresholds"


def compute_shared_signals(sub_results):
    """
    Analyze which indicators are shared across tickers in a subsector.

    Returns: {
        "shared": [(signal_name, count, pct), ...],  # sorted by count desc
        "consensus_level": "High" | "Moderate" | "Low",
        "top_shared": str,  # most common signal
    }
    """
    if not sub_results:
        return {"shared": [], "consensus_level": "—", "top_shared": "—"}

    signal_counts = {}
    total = len(sub_results)

    for res in sub_results:
        for sig in res.get("signals", []):
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

    shared = [
        (sig, count, count / total)
        for sig, count in signal_counts.items()
    ]
    shared.sort(key=lambda x: -x[1])

    # Consensus: what fraction of tickers share the top signal?
    if shared:
        top_pct = shared[0][2]
        if top_pct >= 0.75:
            consensus = "High"
        elif top_pct >= 0.50:
            consensus = "Moderate"
        else:
            consensus = "Low"
        top_shared = INDICATOR_LABELS.get(shared[0][0], shared[0][0])
    else:
        consensus = "—"
        top_shared = "—"

    return {
        "shared": shared,
        "consensus_level": consensus,
        "top_shared": top_shared,
    }


def render_subsector_breakouts(results: list, cfg: dict, data: dict = None):
    """Subsector breakout status page with drill-down into individual tickers."""

    st.title("Subsectors")
    st.caption("Subsector-level momentum detection — when multiple tickers in a group fire simultaneously")

    # Run breakout detection with current scores
    summary = run_breakout_detection(results, cfg)

    # ── Summary Metrics ──
    all_states = summary.get("all_states", {})
    all_derived = summary.get("all_derived", {})
    reaccel_keys = summary.get("reaccelerations", [])

    # Count display statuses (sub-classify quiet by breadth)
    metrics_list = summary.get("metrics", [])
    metrics_by_key = {m["subsector"]: m for m in metrics_list}
    display_counts = {}
    for sub_key, s in all_states.items():
        status = s.get("status", "quiet")
        if status == "quiet":
            b = metrics_by_key.get(sub_key, {}).get("breadth", 0)
            if b >= 0.50:
                status = "steady_hot"
            elif b >= 0.25:
                status = "warming"
        display_counts[status] = display_counts.get(status, 0) + 1

    # Status card data: (key, emoji, label, alpha, definition, color)
    # (key, emoji, label, definition, color)
    status_cards = [
        ("reaccel", "⚡", "Revival", "Was fading, but breadth recovered ≥ 50% with positive acceleration", "#a78bfa"),
        ("confirmed", "🔥", "Confirmed", "Breadth ≥ 50% for 3+ consecutive readings after emerging", "#10b981"),
        ("steady_hot", "🟢", "Steady Hot", "Breadth ≥ 50% but z-score < 1.0 — strong, not newly breaking out", "#22c55e"),
        ("emerging", "👀", "Emerging", "Breadth ≥ 50%, z-score > 1.0, acceleration ≥ 0 — awaiting confirmation", "#fbbf24"),
        ("warming", "🟡", "Warming", "Breadth 25-49% — some tickers heating up", "#eab308"),
        ("fading", "📉", "Fading", "Breadth dropped < 50% or scores declined > 1pt over 5 days", "#f87171"),
        ("quiet", "💤", "Quiet", "Breadth < 25% — no notable activity", "#6b7280"),
    ]

    # Build all cards as one HTML block with background
    cards_html = "<div style='background-color:#2d3348;padding:20px 16px;border-radius:10px;display:flex;justify-content:space-around;gap:8px'>"
    for key, emoji, label, definition, color in status_cards:
        count = len(reaccel_keys) if key == "reaccel" else display_counts.get(key, 0)
        cards_html += (
            f"<div style='text-align:center;flex:1'>"
            f"<div class='status-tooltip' style='font-size:1.3rem;font-weight:700;color:{color};height:3.2rem;display:flex;align-items:center;justify-content:center;position:relative'>"
            f"{emoji} {label}"
            f"<span class='status-tooltiptext'>{definition}</span>"
            f"</div>"
            f"<div style='font-size:2.4rem;font-weight:700;color:white;margin:4px 0'>{count}</div>"
            f"</div>"
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    st.divider()

    # ── Build subsector info with metrics ──
    metrics = summary.get("metrics", [])
    metrics_lookup = {m["subsector"]: m for m in metrics}

    # Build name lookup
    name_lookup = {}
    sector_lookup = {}
    for sector_key, sector in cfg["sectors"].items():
        for sub_key, sub in sector["subsectors"].items():
            name_lookup[sub_key] = sub["name"]
            sector_lookup[sub_key] = sector["name"]

    # Combine states with metrics, derived signals, and sort by priority then score
    subsector_rows = []
    for sub_key, state in all_states.items():
        m = metrics_lookup.get(sub_key, {})
        d = all_derived.get(sub_key, {})
        status = state.get("status", "quiet")
        is_reaccel = sub_key in reaccel_keys

        breadth = m.get("breadth", 0)

        # Sub-classify "quiet" by breadth level
        display_status = status
        if status == "quiet":
            if breadth >= 0.50:
                display_status = "steady_hot"
            elif breadth >= 0.25:
                display_status = "warming"

        config = BREAKOUT_STATUS_CONFIG.get(display_status, BREAKOUT_STATUS_CONFIG["quiet"])
        avg_score = m.get("avg_score", 0)
        z_score = d.get("z_score", np.nan)
        acceleration = d.get("score_acceleration", np.nan)

        # Compute quality grade (#4 + #8)
        if status in ("confirmed", "emerging"):
            grade, grade_color, grade_desc = compute_breakout_quality(
                breadth, avg_score, z_score, acceleration, is_reaccel=is_reaccel
            )
        else:
            grade, grade_color, grade_desc = "—", "#6b7280", ""

        subsector_rows.append({
            "sub_key": sub_key,
            "name": name_lookup.get(sub_key, sub_key),
            "sector": sector_lookup.get(sub_key, ""),
            "status": display_status,
            "label": "Revival" if is_reaccel else config["label"],
            "emoji": "⚡" if is_reaccel else config["emoji"],
            "color": "#a78bfa" if is_reaccel else config["color"],
            "priority": 0 if is_reaccel else config["priority"],
            "avg_score": avg_score,
            "max_score": m.get("max_score", 0),
            "breadth": breadth,
            "hot_count": m.get("hot_count", 0),
            "ticker_count": m.get("ticker_count", 0),
            "peak_avg_score": state.get("peak_avg_score", 0),
            "peak_breadth": state.get("peak_breadth", 0),
            "status_since": state.get("status_since", ""),
            "consecutive_hot": state.get("consecutive_hot", 0),
            # Derived metrics (#9)
            "z_score": z_score,
            "breadth_z_score": d.get("breadth_z_score", np.nan),
            "acceleration": acceleration,
            "score_change_5d": d.get("score_change_5d", np.nan),
            # Quality grade (#4 + #8)
            "grade": grade,
            "grade_color": grade_color,
            "grade_desc": grade_desc,
        })

    subsector_rows.sort(key=lambda x: (x["priority"], -x["avg_score"]))

    # ── Active Signals (non-quiet, non-warming) ──
    active = [r for r in subsector_rows if r["status"] not in ("quiet", "warming")]

    if active:
        st.subheader("Active Signals")

        for r in active:
            # Build expander title with grade badge for confirmed/emerging
            grade_badge = f" · Grade: {r['grade']}" if r["grade"] != "—" else ""
            with st.expander(
                f"{r['emoji']} **{r['name']}** — {r['label']}{grade_badge} · Breadth: {r['breadth']:.0%} · Avg: {r['avg_score']:.1f}",
                expanded=False,
            ):
                # ── Status header with quality grade ──
                grade_html = ""
                if r["grade"] != "—":
                    grade_html = (
                        f"&nbsp;&nbsp;<span style='background-color: {r['grade_color']}; color: white; "
                        f"padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 0.9rem;'>"
                        f"Grade {r['grade']}</span>"
                    )

                st.markdown(
                    f"<span style='background-color: {r['color']}; color: white; padding: 4px 12px; "
                    f"border-radius: 8px; font-weight: 600; font-size: 0.85rem;'>"
                    f"{r['emoji']} {r['label']}</span>"
                    f"{grade_html}"
                    f"&nbsp;&nbsp;<span style='color: #94a3b8; font-size: 0.85rem;'>since {r['status_since']}</span>",
                    unsafe_allow_html=True,
                )

                if r["grade_desc"]:
                    st.caption(r["grade_desc"])

                # ── Metrics row with z-scores (#9) ──
                mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
                with mc1:
                    st.metric("Avg Score", f"{r['avg_score']:.1f}")
                with mc2:
                    st.metric("Breadth", f"{r['breadth']:.0%}")
                with mc3:
                    st.metric("Hot / Total", f"{r['hot_count']} / {r['ticker_count']}")
                with mc4:
                    z_val = f"{r['z_score']:+.1f}σ" if not np.isnan(r['z_score']) else "—"
                    st.metric("Score Z-Score", z_val)
                with mc5:
                    acc_val = f"{r['acceleration']:+.2f}" if not np.isnan(r['acceleration']) else "—"
                    st.metric("Acceleration", acc_val)
                with mc6:
                    chg = r['score_change_5d']
                    chg_val = f"{chg:+.1f}" if not np.isnan(chg) else "—"
                    st.metric("5d Score Δ", chg_val)

                # ── Shared signals analysis (#7) ──
                sub_results = [
                    res for res in results
                    if res.get("subsector") == r["name"]
                ]
                sub_results.sort(key=lambda x: -x["score"])

                signal_analysis = compute_shared_signals(sub_results)

                if signal_analysis["shared"]:
                    consensus_color = {
                        "High": "#10b981",
                        "Moderate": "#fbbf24",
                        "Low": "#f87171",
                    }.get(signal_analysis["consensus_level"], "#6b7280")

                    # Build shared signals display
                    shared_pills = []
                    for sig_name, count, pct in signal_analysis["shared"]:
                        label = INDICATOR_LABELS.get(sig_name, sig_name)
                        pct_str = f"{pct:.0%}"
                        shared_pills.append(
                            f"<span style='background: #1e293b; padding: 6px 10px; border-radius: 6px; "
                            f"font-size: 0.8rem; color: #e2e8f0; flex: 1; "
                            f"display: inline-flex; flex-direction: column; align-items: center; text-align: center;'>"
                            f"<span>{label}</span>"
                            f"<span style='color: {consensus_color}; font-size: 0.75rem;'>{count}/{len(sub_results)}, {pct_str}</span></span>"
                        )

                    st.markdown(
                        f"**Signal Consensus:** "
                        f"<span style='color: {consensus_color}; font-weight: 600;'>{signal_analysis['consensus_level']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<div style='display: flex; flex-wrap: nowrap; gap: 6px; width: 100%;'>"
                        + "".join(shared_pills)
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                # ── Drill-down: individual tickers (table) ──
                st.markdown("**Individual Tickers:**")

                indicator_keys = [
                    "relative_strength", "higher_lows", "ichimoku_cloud",
                    "cmf", "roc", "dual_tf_rs", "atr_expansion",
                ]
                indicator_short = {
                    "relative_strength": "RS",
                    "higher_lows": "HL",
                    "ichimoku_cloud": "Ichimoku",
                    "cmf": "CMF",
                    "roc": "ROC",
                    "dual_tf_rs": "Dual-TF",
                    "atr_expansion": "ATR Exp",
                }

                ticker_rows = []
                for res in sub_results:
                    close = get_current_close(res["indicators"])
                    # 7-day price change
                    pct_7d = ""
                    df_ticker = data.get(res["ticker"])
                    if df_ticker is not None and len(df_ticker) >= 6:
                        price_now = df_ticker["Close"].iloc[-1]
                        price_7d = df_ticker["Close"].iloc[-6]  # ~5 trading days
                        pct_7d = f"{(price_now - price_7d) / price_7d * 100:+.1f}%"

                    sw = res.get("signal_weights", {})
                    row = {
                        "Ticker": f"{res['ticker']} — {res['name']}",
                        "Price": f"${close:,.2f}" if close else "—",
                        "7d Δ": pct_7d,
                        "Score": res["score"],
                    }
                    for k in indicator_keys:
                        row[indicator_short[k]] = sw.get(k, 0)
                    ticker_rows.append(row)

                ticker_df = pd.DataFrame(ticker_rows)

                def color_score(val):
                    if val >= 9:
                        return "background-color: rgba(220, 38, 38, 0.75); color: white; font-weight: bold;"
                    elif val >= 7:
                        return "background-color: rgba(22, 163, 74, 0.75); color: white; font-weight: bold;"
                    elif val >= 5:
                        return "background-color: rgba(234, 179, 8, 0.75); color: white; font-weight: bold;"
                    else:
                        return "background-color: rgba(107, 114, 128, 0.75); color: white;"

                styled = ticker_df.style.map(
                    color_score, subset=["Score"]
                ).format(
                    {k: "{:.1f}" for k in ["Score"] + [indicator_short[k] for k in indicator_keys]}
                )

                st.dataframe(
                    styled,
                    width="stretch",
                    hide_index=True,
                    height=min(400, len(ticker_df) * 35 + 40),
                )

    else:
        st.info("No active breakout signals. All 31 subsectors are quiet.")

    st.divider()

    # ── Full Status Table ──
    st.subheader("All Subsectors")

    table_data = []
    for r in subsector_rows:
        z_display = f"{r['z_score']:+.1f}" if not np.isnan(r['z_score']) else "—"
        acc_display = f"{r['acceleration']:+.2f}" if not np.isnan(r['acceleration']) else "—"

        table_data.append({
            "Status": f"{r['emoji']} {r['label']}",
            "Grade": r["grade"] if r["grade"] != "—" else "",
            "Subsector": r["name"],
            "Sector": r["sector"],
            "Avg Score": r["avg_score"],
            "Max Score": r["max_score"],
            "Breadth": r["breadth"] * 100,
            "Z-Score": z_display,
            "Accel": acc_display,
            "Hot / Total": f"{r['hot_count']}/{r['ticker_count']}",
            "Since": r["status_since"],
        })

    table_df = pd.DataFrame(table_data)
    st.dataframe(
        table_df,
        width="stretch",
        hide_index=True,
        height=min(700, len(table_df) * 35 + 40),
        column_config={
            "Avg Score": st.column_config.NumberColumn("Avg Score", format="%.1f"),
            "Max Score": st.column_config.NumberColumn("Max Score", format="%.1f"),
            "Breadth": st.column_config.ProgressColumn(
                "Breadth", min_value=0, max_value=100, format="%.0f%%",
            ),
        },
    )


# =============================================================
# HISTORICAL CHARTS PAGE
# =============================================================
def get_db_connection():
    """Get a SQLite connection, or None if DB doesn't exist."""
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        return None
    return sqlite3.connect(str(DB_PATH))


@st.cache_data(ttl=3600)
def load_ticker_scores_db():
    """Load all ticker scores from SQLite."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    df = pd.read_sql_query("SELECT * FROM ticker_scores ORDER BY date, ticker", conn)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def load_subsector_daily_db():
    """Load all subsector daily metrics from SQLite."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    df = pd.read_sql_query("SELECT * FROM subsector_daily ORDER BY date, subsector", conn)
    conn.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def render_historical_charts():
    """Historical charts page — pulls from SQLite database."""

    st.title("📊 Historical Charts")

    ticker_df = load_ticker_scores_db()
    subsector_df = load_subsector_daily_db()

    if ticker_df.empty:
        st.warning("No historical data found. Run `python3 backfill_subsector.py` to populate the database.")
        return

    date_range = f"{ticker_df['date'].min().strftime('%Y-%m-%d')} → {ticker_df['date'].max().strftime('%Y-%m-%d')}"
    n_dates = ticker_df["date"].nunique()
    st.caption(f"Data: {date_range} · {n_dates} snapshots · {ticker_df['ticker'].nunique()} tickers")

    # ─────────────────────────────────────────────
    # CHART 1: Ticker Score History
    # ─────────────────────────────────────────────
    st.subheader("1️⃣ Ticker Score History")
    st.caption("Track how a ticker's breakout score evolved over time")

    all_tickers = sorted(ticker_df["ticker"].unique())
    selected_ticker = st.selectbox("Select ticker", all_tickers, index=0)

    ticker_hist = ticker_df[ticker_df["ticker"] == selected_ticker].sort_values("date")

    if not ticker_hist.empty:
        fig = go.Figure()

        # Score line
        fig.add_trace(go.Scatter(
            x=ticker_hist["date"], y=ticker_hist["score"],
            mode="lines+markers",
            name="Score",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=6),
        ))

        # Threshold lines
        fig.add_hline(y=8, line_dash="dash", line_color="#10b981", opacity=0.5,
                      annotation_text="Strong (8+)", annotation_position="top left")
        fig.add_hline(y=6, line_dash="dot", line_color="#fbbf24", opacity=0.4,
                      annotation_text="Moderate (6+)", annotation_position="top left")

        # Color zones
        fig.add_hrect(y0=8, y1=10, fillcolor="#10b981", opacity=0.06)
        fig.add_hrect(y0=6, y1=8, fillcolor="#34d399", opacity=0.04)

        fig.update_layout(
            title=f"{selected_ticker} — Score History",
            yaxis_title="Breakout Score",
            yaxis=dict(range=[0, 10.5]),
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show signals at each date
        with st.expander("📋 Signal details by date"):
            display = ticker_hist[["date", "score", "subsector", "signals"]].copy()
            display["date"] = display["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display, width="stretch", hide_index=True)
    else:
        st.info(f"No historical data for {selected_ticker}")

    st.divider()

    # ─────────────────────────────────────────────
    # CHART 2: Subsector Breadth Trends
    # ─────────────────────────────────────────────
    st.subheader("2️⃣ Subsector Breadth Trends")
    st.caption("How hot is each subsector over time? (breadth = % of tickers scoring 7+)")

    if not subsector_df.empty:
        all_subsectors = sorted(subsector_df["subsector_name"].unique())

        # Default to top 5 by latest breadth
        latest_date = subsector_df["date"].max()
        latest = subsector_df[subsector_df["date"] == latest_date].sort_values("breadth", ascending=False)
        default_subs = latest["subsector_name"].head(5).tolist()

        selected_subs = st.multiselect(
            "Select subsectors (max 8)",
            all_subsectors,
            default=default_subs,
            max_selections=8,
        )

        if selected_subs:
            fig = go.Figure()

            colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]

            for i, sub_name in enumerate(selected_subs):
                sub_data = subsector_df[subsector_df["subsector_name"] == sub_name].sort_values("date")
                fig.add_trace(go.Scatter(
                    x=sub_data["date"],
                    y=sub_data["breadth"] * 100,
                    mode="lines+markers",
                    name=sub_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=5),
                ))

            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.4,
                          annotation_text="50% breadth trigger")

            fig.update_layout(
                title="Subsector Breadth Over Time",
                yaxis_title="Breadth (%)",
                yaxis=dict(range=[0, 105]),
                height=450,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Also show avg score trends
            fig2 = go.Figure()
            for i, sub_name in enumerate(selected_subs):
                sub_data = subsector_df[subsector_df["subsector_name"] == sub_name].sort_values("date")
                fig2.add_trace(go.Scatter(
                    x=sub_data["date"],
                    y=sub_data["avg_score"],
                    mode="lines+markers",
                    name=sub_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=5),
                ))

            fig2.update_layout(
                title="Subsector Avg Score Over Time",
                yaxis_title="Average Score",
                yaxis=dict(range=[0, 10]),
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────
    # CHART 3: Ticker × Date Heatmap
    # ─────────────────────────────────────────────
    st.subheader("3️⃣ Score Heatmap")
    st.caption("All tickers across all dates — darker = higher score")

    # Let user filter by sector or show top N
    heatmap_mode = st.radio("Show", ["Top 20 by latest score", "By subsector"], horizontal=True)

    if heatmap_mode == "Top 20 by latest score":
        latest_scores = ticker_df[ticker_df["date"] == ticker_df["date"].max()].sort_values("score", ascending=False)
        top_tickers = latest_scores["ticker"].head(20).tolist()
        heat_data = ticker_df[ticker_df["ticker"].isin(top_tickers)]
    else:
        sub_options = sorted(ticker_df["subsector"].dropna().unique())
        selected_sub = st.selectbox("Select subsector for heatmap", sub_options)
        heat_data = ticker_df[ticker_df["subsector"] == selected_sub]

    if not heat_data.empty:
        # Filter to last 3 months
        cutoff_date = heat_data["date"].max() - pd.Timedelta(days=90)
        heat_data = heat_data[heat_data["date"] >= cutoff_date]
        pivot = heat_data.pivot_table(index="ticker", columns="date", values="score", aggfunc="first")
        pivot = pivot.sort_values(pivot.columns[-1], ascending=False)
        pivot.columns = [c.strftime("%m/%d") for c in pivot.columns]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale=[
                [0, "#ffffff"],       # 0 — white
                [0.3, "#dcfce7"],     # 3 — very light green
                [0.5, "#86efac"],     # 5 — light green
                [0.7, "#22c55e"],     # 7 — green
                [0.9, "#16a34a"],     # 9 — darker green
                [1.0, "#047857"],     # 10 — dark green
            ],
            zmin=0, zmax=10,
            text=pivot.values.round(1),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Ticker: %{y}<br>Date: %{x}<br>Score: %{z:.1f}<extra></extra>",
        ))

        fig.update_layout(
            title="Score Heatmap",
            height=max(300, len(pivot) * 28),
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────
    # CHART 4: Universe Score Distribution Over Time
    # ─────────────────────────────────────────────
    st.subheader("4️⃣ Universe Score Distribution Over Time")
    st.caption("How the overall market breadth has shifted across snapshots")

    dates = sorted(ticker_df["date"].unique())

    dist_data = []
    for d in dates:
        day_scores = ticker_df[ticker_df["date"] == d]["score"]
        dist_data.append({
            "date": d,
            "Fire (9+)": (day_scores >= 9).sum(),
            "Strong (7-9)": ((day_scores >= 7) & (day_scores < 9)).sum(),
            "Moderate (5-7)": ((day_scores >= 5) & (day_scores < 7)).sum(),
            "Weak (<5)": (day_scores < 5).sum(),
            "Mean Score": day_scores.mean(),
            "Median Score": day_scores.median(),
        })

    dist_df = pd.DataFrame(dist_data)

    # Stacked area chart of score buckets
    fig = go.Figure()

    bucket_colors = {"Fire (9+)": "#dc2626", "Strong (7-9)": "#16a34a",
                     "Moderate (5-7)": "#eab308", "Weak (<5)": "#d1d5db"}

    for bucket, color in bucket_colors.items():
        fig.add_trace(go.Scatter(
            x=dist_df["date"], y=dist_df[bucket],
            mode="lines",
            name=bucket,
            stackgroup="one",
            fillcolor=color,
            line=dict(color=color, width=0.5),
        ))

    fig.update_layout(
        title="Score Distribution Over Time (Stacked)",
        yaxis_title="Number of Tickers",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mean/Median score trend
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=dist_df["date"], y=dist_df["Mean Score"],
        mode="lines+markers", name="Mean Score",
        line=dict(color="#3b82f6", width=2.5),
    ))
    fig2.add_trace(go.Scatter(
        x=dist_df["date"], y=dist_df["Median Score"],
        mode="lines+markers", name="Median Score",
        line=dict(color="#8b5cf6", width=2, dash="dot"),
    ))

    fig2.update_layout(
        title="Universe Mean & Median Score",
        yaxis_title="Score",
        yaxis=dict(range=[0, 10]),
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================
# MAIN
# =============================================================
def main():
    cfg, data, results, timestamps = load_and_score(period="2y")

    # Add DB last-updated timestamp
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute("SELECT MAX(date) FROM ticker_scores").fetchone()
        conn.close()
        if row and row[0]:
            db_date = pd.Timestamp(row[0])
            timestamps["db_last"] = db_date.strftime("%m/%d/%y")
        else:
            timestamps["db_last"] = "No data"
    except Exception:
        timestamps["db_last"] = "No DB"

    # Breakout detection timestamp gets set after it runs
    timestamps["breakout"] = timestamps["scoring"]

    page = render_sidebar(cfg, timestamps)

    if page == "🔥 Subsectors":
        render_subsector_breakouts(results, cfg, data)
    elif page == "📈 Tickers":
        render_dashboard(results, cfg, data=data)
    elif page == "📊 Historical Charts":
        render_historical_charts()


if __name__ == "__main__":
    main()
