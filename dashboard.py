# =============================================================================
# CryptOracle - Interactive Streamlit Dashboard
# Run with: streamlit run dashboard.py
# =============================================================================

import os
import pickle
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import config

# ---------------------------------------------------------------------------
# Page config — must be the FIRST streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CryptOracle",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .metric-card {
        background: #1A1F2E;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #F7931A;
        margin: 6px 0;
    }
    .metric-title { color: #7F8C8D; font-size: 13px; margin-bottom: 4px; }
    .metric-value { color: white; font-size: 24px; font-weight: bold; }
    .metric-delta { font-size: 13px; margin-top: 4px; }
    .positive { color: #00C896; }
    .negative { color: #FF4B4B; }
    h1, h2, h3 { color: white !important; }
    .stSelectbox label { color: white !important; }
</style>
""", unsafe_allow_html=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(symbol: str = "BTC"):
    """Load raw merged data from disk (collected by data_collection.py)."""
    path = f"{config.DATA_RAW_DIR}/{symbol}_raw.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


@st.cache_data(ttl=3600)
def load_feature_data(symbol: str = "BTC"):
    path = f"{config.DATA_PROCESSED_DIR}/{symbol}_features.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


@st.cache_resource
def load_model_and_scaler(symbol: str = "BTC"):
    """Load trained model and scaler (cached in memory)."""
    from model import AttentionLayer
    import tensorflow as tf

    model_path  = f"{config.MODELS_DIR}/cryptoracle_best.keras"
    scaler_path = f"{config.DATA_PROCESSED_DIR}/{symbol}_scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"AttentionLayer": AttentionLayer}
    )
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
def candlestick_chart(df: pd.DataFrame, title: str = "Price Chart"):
    """Interactive candlestick + volume chart."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.75, 0.25])

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00C896",
        decreasing_line_color="#FF4B4B",
        name="OHLCV"
    ), row=1, col=1)

    # Add SMAs if available
    for w in config.SMA_WINDOWS:
        col = f"sma_{w}"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                mode="lines", name=f"SMA {w}",
                line=dict(width=1)
            ), row=1, col=1)

    # Volume bars
    colours = ["#00C896" if c >= o else "#FF4B4B"
               for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colours, name="Volume", opacity=0.7
    ), row=2, col=1)

    fig.update_layout(
        title=title, height=600,
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font_color="white", xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="#1A1F2E", bordercolor="#444", borderwidth=1)
    )
    fig.update_xaxes(gridcolor="#1A1F2E")
    fig.update_yaxes(gridcolor="#1A1F2E")
    return fig


def fear_greed_gauge(value: float):
    """Gauge chart for Fear & Greed Index."""
    if value < 25:
        colour, label = "#FF4B4B", "Extreme Fear 😨"
    elif value < 45:
        colour, label = "#E67E22", "Fear 😟"
    elif value < 55:
        colour, label = "#F1C40F", "Neutral 😐"
    elif value < 75:
        colour, label = "#2ECC71", "Greed 😀"
    else:
        colour, label = "#00C896", "Extreme Greed 🤑"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": f"Fear & Greed: {label}", "font": {"color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white"},
            "bar": {"color": colour},
            "bgcolor": "#1A1F2E",
            "steps": [
                {"range": [0, 25],  "color": "#2D0A0A"},
                {"range": [25, 45], "color": "#2D1A0A"},
                {"range": [45, 55], "color": "#2D2A0A"},
                {"range": [55, 75], "color": "#0A2D1A"},
                {"range": [75, 100],"color": "#0A2D0A"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "value": value}
        },
        number={"font": {"color": "white"}},
    ))
    fig.update_layout(paper_bgcolor="#0E1117", font_color="white", height=280)
    return fig


def prediction_vs_actual_chart(dates, y_true, y_pred):
    """Actual vs predicted price comparison."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=y_true, mode="lines", name="Actual",
        line=dict(color="#F7931A", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=y_pred, mode="lines", name="Predicted",
        line=dict(color="#627EEA", width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(y_pred * 1.03) + list(y_pred[::-1] * 0.97),
        fill="toself", fillcolor="rgba(98,126,234,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="±3% Band"
    ))
    fig.update_layout(
        title="Actual vs Predicted Close Price",
        height=450, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font_color="white", xaxis_title="Date", yaxis_title="Price (USD)"
    )
    fig.update_xaxes(gridcolor="#1A1F2E")
    fig.update_yaxes(gridcolor="#1A1F2E")
    return fig


def monte_carlo_chart(mc_results: dict, historical_prices, historical_dates):
    """Probability cone chart for Monte Carlo simulation."""
    import pandas as pd
    n_days = len(mc_results["mean"])
    future_dates = pd.date_range(
        start=historical_dates[-1], periods=n_days + 1, freq="D")[1:]

    fig = go.Figure()

    # Historical prices (last 90 days)
    fig.add_trace(go.Scatter(
        x=historical_dates[-90:], y=historical_prices[-90:],
        mode="lines", name="Historical",
        line=dict(color="#F7931A", width=2)
    ))

    # 90% confidence band
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(mc_results["upper_95"]) + list(mc_results["lower_5"][::-1]),
        fill="toself", fillcolor="rgba(98,126,234,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="90% Confidence"
    ))

    # 50% confidence band
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(mc_results["upper_75"]) + list(mc_results["lower_25"][::-1]),
        fill="toself", fillcolor="rgba(98,126,234,0.30)",
        line=dict(color="rgba(0,0,0,0)"), name="50% Confidence"
    ))

    # Mean forecast
    fig.add_trace(go.Scatter(
        x=future_dates, y=mc_results["mean"],
        mode="lines", name="Mean Forecast",
        line=dict(color="#00C896", width=2.5, dash="dash")
    ))

    # Divider
    fig.add_vline(x=str(historical_dates[-1]), line_color="white",
                  line_dash="dash", opacity=0.5)

    fig.update_layout(
        title=f"30-Day Monte Carlo Forecast ({config.MONTE_CARLO_RUNS} paths)",
        height=500, paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        font_color="white", xaxis_title="Date", yaxis_title="Price (USD)"
    )
    fig.update_xaxes(gridcolor="#1A1F2E")
    fig.update_yaxes(gridcolor="#1A1F2E")
    return fig


# ---------------------------------------------------------------------------
# Main Dashboard Layout
# ---------------------------------------------------------------------------
def main():
    # ── Header ──────────────────────────────────────────────────────────────
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.markdown("# 🔮")
    with col_title:
        st.markdown("# CryptOracle")
        st.markdown("*Sentiment-Aware Multi-Signal Crypto Forecasting*")

    st.markdown("---")

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        symbol     = st.selectbox("Cryptocurrency", list(config.CRYPTO_SYMBOLS.keys()))
        date_range = st.selectbox("Date Range", ["6 Months", "1 Year", "2 Years", "All"])
        show_sma   = st.checkbox("Show Moving Averages", value=True)
        show_bb    = st.checkbox("Show Bollinger Bands", value=False)
        run_mc     = st.button("🎲 Run Monte Carlo Simulation")

        st.markdown("---")
        st.markdown("**Model Info**")
        st.markdown(f"Sequence Length: **{config.SEQUENCE_LENGTH} days**")
        st.markdown(f"Architecture: **Bi-LSTM + Attention**")
        st.markdown(f"Features: **60+**")
        st.markdown("---")
        st.markdown("⚠️ *For educational purposes only. Not financial advice.*")

    # ── Load data ────────────────────────────────────────────────────────────
    df = load_data(symbol)
    if df is None:
        st.error("⚠️ No data found. Please run `python run_pipeline.py` first.")
        st.stop()

    # Apply date range filter
    range_map = {"6 Months": 180, "1 Year": 365, "2 Years": 730, "All": 9999}
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=range_map[date_range])
    df_filtered = df[df.index >= cutoff].copy()

    # ── KPI Row ──────────────────────────────────────────────────────────────
    latest    = df_filtered["Close"].iloc[-1]
    prev_day  = df_filtered["Close"].iloc[-2]
    change_1d = (latest - prev_day) / prev_day * 100
    change_7d = (latest - df_filtered["Close"].iloc[-8]) / df_filtered["Close"].iloc[-8] * 100 \
                if len(df_filtered) > 8 else 0
    high_52w  = df_filtered["High"].tail(365).max()
    low_52w   = df_filtered["Low"].tail(365).min()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        delta_col = "positive" if change_1d >= 0 else "negative"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Current Price</div>
            <div class="metric-value">${latest:,.0f}</div>
            <div class="metric-delta {delta_col}">{change_1d:+.2f}% (24h)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        delta_col = "positive" if change_7d >= 0 else "negative"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">7-Day Change</div>
            <div class="metric-value">{change_7d:+.2f}%</div>
            <div class="metric-delta">vs 7 days ago</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">52W High</div>
            <div class="metric-value">${high_52w:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">52W Low</div>
            <div class="metric-value">${low_52w:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        fg_val = df_filtered["fear_greed_value"].iloc[-1] if "fear_greed_value" in df_filtered.columns else 50
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Fear & Greed</div>
            <div class="metric-value">{fg_val:.0f}/100</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Main Chart + Fear & Greed ─────────────────────────────────────────────
    col_chart, col_fg = st.columns([3, 1])
    with col_chart:
        st.plotly_chart(candlestick_chart(df_filtered, f"{symbol} Price Chart"),
                        use_container_width=True)
    with col_fg:
        if "fear_greed_value" in df_filtered.columns:
            fg_current = df_filtered["fear_greed_value"].iloc[-1]
            st.plotly_chart(fear_greed_gauge(fg_current), use_container_width=True)

            # Fear & Greed history
            fg_hist = go.Figure(go.Scatter(
                x=df_filtered.index, y=df_filtered["fear_greed_value"],
                fill="tozeroy", line=dict(color="#627EEA"),
                fillcolor="rgba(98,126,234,0.3)"
            ))
            fg_hist.update_layout(
                title="F&G History", height=200,
                paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                font_color="white", margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fg_hist, use_container_width=True)

    # ── Technical Indicators ───────────────────────────────────────────────
    st.markdown("### 📊 Technical Indicators")
    tab_rsi, tab_macd, tab_vol = st.tabs(["RSI", "MACD", "Volatility"])

    feat_df = load_feature_data(symbol)
    if feat_df is not None:
        fd = feat_df[feat_df.index >= cutoff]

        with tab_rsi:
            if "rsi" in fd.columns:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=fd.index, y=fd["rsi"], name="RSI",
                    line=dict(color="#F7931A", width=2)
                ))
                fig_rsi.add_hline(y=70, line_color="#FF4B4B", line_dash="dash",
                                   annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_color="#00C896", line_dash="dash",
                                   annotation_text="Oversold (30)")
                fig_rsi.update_layout(height=300, paper_bgcolor="#0E1117",
                                       plot_bgcolor="#0E1117", font_color="white")
                fig_rsi.update_xaxes(gridcolor="#1A1F2E")
                fig_rsi.update_yaxes(gridcolor="#1A1F2E", range=[0, 100])
                st.plotly_chart(fig_rsi, use_container_width=True)

        with tab_macd:
            if "macd" in fd.columns:
                fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                          row_heights=[0.6, 0.4])
                fig_macd.add_trace(go.Scatter(
                    x=fd.index, y=fd["macd"], name="MACD",
                    line=dict(color="#627EEA")), row=1, col=1)
                fig_macd.add_trace(go.Scatter(
                    x=fd.index, y=fd["macd_signal"], name="Signal",
                    line=dict(color="#F7931A")), row=1, col=1)
                colours_macd = ["#00C896" if v >= 0 else "#FF4B4B"
                                for v in fd["macd_hist"]]
                fig_macd.add_trace(go.Bar(
                    x=fd.index, y=fd["macd_hist"],
                    marker_color=colours_macd, name="Histogram"), row=2, col=1)
                fig_macd.update_layout(height=350, paper_bgcolor="#0E1117",
                                        plot_bgcolor="#0E1117", font_color="white")
                st.plotly_chart(fig_macd, use_container_width=True)

        with tab_vol:
            if "volatility_30d" in fd.columns:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=fd.index, y=fd["volatility_30d"] * 100,
                    fill="tozeroy", name="30d Volatility (%)",
                    line=dict(color="#F7931A"),
                    fillcolor="rgba(247,147,26,0.2)"
                ))
                fig_vol.update_layout(height=300, paper_bgcolor="#0E1117",
                                       plot_bgcolor="#0E1117", font_color="white",
                                       yaxis_title="Annualised Volatility (%)")
                st.plotly_chart(fig_vol, use_container_width=True)

    # ── Model Predictions ────────────────────────────────────────────────────
    st.markdown("### 🎯 Model Predictions")
    model, scaler = load_model_and_scaler(symbol)

    if model is None:
        st.info("ℹ️ Model not yet trained. Run `python run_pipeline.py` to train.")
    else:
        pred_results_path = f"{config.LOGS_DIR}/prediction_results.csv"
        if os.path.exists(pred_results_path):
            pred_df = pd.read_csv(pred_results_path, index_col="Date", parse_dates=True)
            fig_pred = prediction_vs_actual_chart(
                pred_df.index,
                pred_df["actual"].values,
                pred_df["predicted"].values
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Metrics
            from evaluation import compute_metrics
            metrics = compute_metrics(pred_df["actual"].values,
                                      pred_df["predicted"].values)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("RMSE",  f"${metrics['RMSE']:,.0f}")
            m2.metric("MAE",   f"${metrics['MAE']:,.0f}")
            m3.metric("MAPE",  f"{metrics['MAPE']:.2f}%")
            m4.metric("R²",    f"{metrics['R2']:.3f}")
            m5.metric("Dir. Accuracy", f"{metrics['Directional_Accuracy']*100:.1f}%")

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    if run_mc and model is not None and feat_df is not None:
        st.markdown("### 🎲 Monte Carlo Simulation")
        with st.spinner(f"Running {config.MONTE_CARLO_RUNS} simulation paths..."):
            from preprocessing import preprocess, inverse_transform_predictions
            from advanced_features import monte_carlo_simulation

            feat_df_proc = load_feature_data(symbol)
            result = preprocess(feat_df_proc, symbol)
            (_, _, _, _, _, _, X_test, _, _, scaler_loaded, meta) = result

            last_seq = X_test[-1:].copy()
            historical_real = inverse_transform_predictions(
                scaler_loaded, result[6].flatten(),
                meta["n_features"], meta["target_col_idx"])

            mc = monte_carlo_simulation(model, last_seq, scaler_loaded,
                                        meta["n_features"], meta["target_col_idx"])
            hist_dates = result[8]   # test dates

            fig_mc = monte_carlo_chart(mc, historical_real, hist_dates)
            st.plotly_chart(fig_mc, use_container_width=True)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("30d Median Forecast", f"${mc['median'][-1]:,.0f}")
            col_b.metric("Lower 5th Pct", f"${mc['lower_5'][-1]:,.0f}")
            col_c.metric("Upper 95th Pct", f"${mc['upper_95'][-1]:,.0f}")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#7F8C8D; font-size:12px;'>"
        "CryptOracle | Built with TensorFlow, Streamlit & Plotly | "
        "⚠️ Educational purposes only — not financial advice"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
