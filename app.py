"""
Real-Time Stock Price Analyzer Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import threading

from utils.data_fetcher import StockDataFetcher
from utils.indicators import TechnicalIndicators
from utils.trend_detector import TrendDetector
from models.predictor import StockPredictor
from alerts.alert_manager import AlertManager

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  :root {
    --bg: #0a0e1a;
    --surface: #111827;
    --border: #1f2937;
    --accent: #00d4ff;
    --green: #00ff88;
    --red: #ff4466;
    --yellow: #ffcc00;
    --text: #e2e8f0;
    --muted: #64748b;
  }

  html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
  }

  .metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
  }

  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1;
  }

  .metric-label {
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.25rem;
  }

  .bullish { color: var(--green) !important; }
  .bearish { color: var(--red) !important; }
  .neutral { color: var(--yellow) !important; }

  .signal-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
  }

  .badge-bullish { background: rgba(0,255,136,0.15); color: var(--green); border: 1px solid var(--green); }
  .badge-bearish { background: rgba(255,68,102,0.15); color: var(--red); border: 1px solid var(--red); }
  .badge-neutral { background: rgba(255,204,0,0.15); color: var(--yellow); border: 1px solid var(--yellow); }

  .stSidebar { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
  .stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
  }
  .stTextInput > div > input {
    background: var(--bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1rem !important;
  }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ────────────────────────────────────────────────────────
if "fetcher" not in st.session_state:
    st.session_state.fetcher = StockDataFetcher()
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Analyzer Pro")
    st.markdown("---")

    ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="AAPL, TSLA, INFY.NS...").upper().strip()

    st.markdown("### 📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("To", value=datetime.now())

    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m", "1wk"], index=0)

    st.markdown("### 📊 Moving Averages")
    sma_windows = st.multiselect("SMA Windows", [5, 10, 20, 50, 100, 200], default=[20, 50, 200])
    ema_windows = st.multiselect("EMA Windows", [5, 10, 20, 50, 100, 200], default=[12, 26])

    st.markdown("### 🔔 Alert Thresholds")
    price_alert_high = st.number_input("Price High Alert ($)", value=0.0, step=1.0)
    price_alert_low = st.number_input("Price Low Alert ($)", value=0.0, step=1.0)

    st.markdown("### ⚡ Auto-Refresh")
    auto_refresh = st.toggle("Enable Auto-Refresh", value=False)
    refresh_interval = st.slider("Interval (seconds)", 5, 60, 30) if auto_refresh else 30

    st.markdown("---")
    fetch_btn = st.button("🔄 Fetch Data", use_container_width=True)

    st.markdown("### 📧 Email Alerts")
    with st.expander("Configure"):
        email_to = st.text_input("Recipient Email")
        smtp_host = st.text_input("SMTP Host", value="smtp.gmail.com")
        smtp_user = st.text_input("SMTP User")
        smtp_pass = st.text_input("SMTP Password", type="password")

    st.markdown("### 🤖 Telegram Alerts")
    with st.expander("Configure"):
        tg_token = st.text_input("Bot Token")
        tg_chat_id = st.text_input("Chat ID")

# ─── Auto Refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown(f"# 📊 {ticker} — Real-Time Stock Analyzer")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# ─── Fetch & Process Data ─────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(ticker, start, end, interval):
    fetcher = StockDataFetcher()
    df = fetcher.fetch(ticker, str(start), str(end), interval)
    return df

try:
    with st.spinner(f"Fetching {ticker} data..."):
        df = load_data(ticker, start_date, end_date, interval)

    if df is None or df.empty:
        st.error(f"❌ No data found for **{ticker}**. Check the ticker symbol and try again.")
        st.stop()

    # Calculate indicators
    ti = TechnicalIndicators(df)
    df = ti.add_sma(sma_windows)
    df = ti.add_ema(ema_windows)
    df = ti.add_rsi()
    df = ti.add_macd()
    df = ti.add_bollinger_bands()
    df = ti.add_volume_ma()

    # Trend detection
    td = TrendDetector(df)
    trend_info = td.detect_trend(sma_windows)
    crossovers = td.detect_crossovers(sma_windows)
    signals = td.generate_signals()

    # Alert checks
    alert_mgr = AlertManager(
        email_config={"to": email_to, "host": smtp_host, "user": smtp_user, "password": smtp_pass} if email_to else None,
        telegram_config={"token": tg_token, "chat_id": tg_chat_id} if tg_token and tg_chat_id else None,
    )
    current_price = float(df["Close"].iloc[-1])
    triggered_alerts = alert_mgr.check_price_alerts(ticker, current_price, price_alert_high, price_alert_low)
    ma_alerts = alert_mgr.check_ma_crossover_alerts(ticker, crossovers)
    all_alerts = triggered_alerts + ma_alerts
    if all_alerts:
        st.session_state.alerts = all_alerts + st.session_state.alerts[:20]

    # ─── Top Metrics Row ──────────────────────────────────────────────────────
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
    pct_change = ((current_price - prev_close) / prev_close) * 100
    high_52w = float(df["High"].max())
    low_52w = float(df["Low"].min())
    vol = int(df["Volume"].iloc[-1])
    avg_vol = int(df["Volume"].mean())

    color_cls = "bullish" if pct_change > 0 else "bearish" if pct_change < 0 else "neutral"
    arrow = "▲" if pct_change > 0 else "▼"

    cols = st.columns(6)
    metrics = [
        ("Current Price", f"${current_price:,.2f}", color_cls),
        ("Change", f"{arrow} {abs(pct_change):.2f}%", color_cls),
        ("52W High", f"${high_52w:,.2f}", "bullish"),
        ("52W Low", f"${low_52w:,.2f}", "bearish"),
        ("Volume", f"{vol:,}", "neutral"),
        ("Avg Volume", f"{avg_vol:,}", "neutral"),
    ]
    for col, (label, value, cls) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value {cls}">{value}</div>
            </div>""", unsafe_allow_html=True)

    # ─── Trend Summary ────────────────────────────────────────────────────────
    st.markdown("---")
    trend_cols = st.columns([2, 1, 1, 1])
    with trend_cols[0]:
        badge_cls = "badge-bullish" if trend_info["overall"] == "BULLISH" else "badge-bearish" if trend_info["overall"] == "BEARISH" else "badge-neutral"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Overall Trend</div>
          <span class="signal-badge {badge_cls}">{trend_info['overall']}</span>
          <span style="margin-left:0.5rem;font-size:0.85rem;color:#94a3b8">{trend_info.get('description','')}</span>
        </div>""", unsafe_allow_html=True)
    with trend_cols[1]:
        rsi_val = float(df["RSI"].iloc[-1]) if "RSI" in df.columns and not pd.isna(df["RSI"].iloc[-1]) else 50
        rsi_cls = "bearish" if rsi_val > 70 else "bullish" if rsi_val < 30 else "neutral"
        rsi_label = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">RSI (14) — {rsi_label}</div>
          <div class="metric-value {rsi_cls}">{rsi_val:.1f}</div>
        </div>""", unsafe_allow_html=True)
    with trend_cols[2]:
        macd_val = float(df["MACD"].iloc[-1]) if "MACD" in df.columns and not pd.isna(df["MACD"].iloc[-1]) else 0
        macd_cls = "bullish" if macd_val > 0 else "bearish"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">MACD</div>
          <div class="metric-value {macd_cls}">{macd_val:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with trend_cols[3]:
        latest_signal = signals[-1] if signals else {"type": "HOLD", "reason": "No signal"}
        sig_cls = "badge-bullish" if latest_signal["type"] == "BUY" else "badge-bearish" if latest_signal["type"] == "SELL" else "badge-neutral"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Latest Signal</div>
          <span class="signal-badge {sig_cls}">{latest_signal['type']}</span>
          <div style="font-size:0.72rem;color:#94a3b8;margin-top:0.3rem">{latest_signal.get('reason','')}</div>
        </div>""", unsafe_allow_html=True)

    # ─── Crossover Alerts ─────────────────────────────────────────────────────
    if crossovers:
        recent = crossovers[-3:]
        for c in reversed(recent):
            icon = "🌟" if c["type"] == "GOLDEN_CROSS" else "💀"
            col_cls = "green" if c["type"] == "GOLDEN_CROSS" else "red"
            st.info(f"{icon} **{c['type'].replace('_',' ')}** on {c['date']} — {c['description']}")

    # ─── Main Price Chart ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📉 Price Chart with Moving Averages")

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.2, 0.15],
        subplot_titles=["", "Volume", "RSI (14)", "MACD"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing_line_color="#00ff88", decreasing_line_color="#ff4466",
        increasing_fillcolor="rgba(0,255,136,0.2)", decreasing_fillcolor="rgba(255,68,102,0.2)",
    ), row=1, col=1)

    # Bollinger Bands
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="rgba(100,150,255,0.5)", dash="dot", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color="rgba(100,150,255,0.5)", dash="dot", width=1),
            fill="tonexty", fillcolor="rgba(100,150,255,0.03)"), row=1, col=1)

    # SMAs
    sma_colors = ["#00d4ff", "#ffcc00", "#ff88cc", "#88ff00", "#ff8800"]
    for i, w in enumerate(sma_windows):
        col_name = f"SMA_{w}"
        if col_name in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col_name], name=f"SMA {w}",
                line=dict(color=sma_colors[i % len(sma_colors)], width=1.5)), row=1, col=1)

    # EMAs
    ema_colors = ["#ff6644", "#44ff66"]
    for i, w in enumerate(ema_windows):
        col_name = f"EMA_{w}"
        if col_name in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col_name], name=f"EMA {w}",
                line=dict(color=ema_colors[i % len(ema_colors)], width=1.5, dash="dash")), row=1, col=1)

    # Crossover markers
    for c in crossovers:
        marker_color = "#00ff88" if c["type"] == "GOLDEN_CROSS" else "#ff4466"
        marker_symbol = "triangle-up" if c["type"] == "GOLDEN_CROSS" else "triangle-down"
        price_at = df["Close"].get(c["date"], current_price)
        fig.add_trace(go.Scatter(
            x=[c["date"]], y=[float(price_at)],
            mode="markers+text",
            marker=dict(symbol=marker_symbol, size=14, color=marker_color, line=dict(width=2, color="white")),
            text=["GC" if c["type"] == "GOLDEN_CROSS" else "DC"],
            textposition="top center",
            textfont=dict(size=9, color=marker_color),
            name=c["type"].replace("_", " "),
            showlegend=False,
        ), row=1, col=1)

    # Volume
    vol_colors = ["rgba(0,255,136,0.5)" if c >= o else "rgba(255,68,102,0.5)"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
        marker_color=vol_colors, showlegend=False), row=2, col=1)
    if "Volume_MA" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Volume_MA"], name="Vol MA",
            line=dict(color="#ffcc00", width=1)), row=2, col=1)

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#00d4ff", width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ff4466", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00ff88", line_width=1, row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,68,102,0.05)", line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,136,0.05)", line_width=0, row=3, col=1)

    # MACD
    if "MACD" in df.columns and "MACD_signal" in df.columns:
        macd_colors = ["rgba(0,255,136,0.6)" if v >= 0 else "rgba(255,68,102,0.6)" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD Hist",
            marker_color=macd_colors, showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#00d4ff", width=1.5)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
            line=dict(color="#ff6644", width=1.5)), row=4, col=1)

    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font=dict(family="DM Sans", color="#94a3b8", size=11),
        legend=dict(bgcolor="rgba(17,24,39,0.9)", bordercolor="#1f2937", borderwidth=1,
                    font=dict(size=10), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor="#1f2937", showgrid=True, row=i, col=1, zeroline=False)
        fig.update_yaxes(gridcolor="#1f2937", showgrid=True, row=i, col=1, zeroline=False)

    st.plotly_chart(fig, use_container_width=True)

    # ─── ML Prediction ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 ML Price Prediction")

    pred_col1, pred_col2 = st.columns([3, 1])
    with pred_col2:
        model_type = st.selectbox("Model", ["Linear Regression", "Random Forest", "Gradient Boosting"])
        pred_days = st.slider("Predict N Days", 1, 30, 7)
        run_pred = st.button("Run Prediction", use_container_width=True)

    with pred_col1:
        if run_pred or "prediction_result" in st.session_state:
            with st.spinner("Training model..."):
                predictor = StockPredictor(df, model_type=model_type.lower().replace(" ", "_"))
                result = predictor.train_and_predict(future_days=pred_days)
                st.session_state.prediction_result = result

            result = st.session_state.prediction_result
            fig_pred = go.Figure()

            # Actual prices
            fig_pred.add_trace(go.Scatter(
                x=result["actual_dates"], y=result["actual"],
                name="Actual", line=dict(color="#00d4ff", width=2)))

            # Train predictions
            fig_pred.add_trace(go.Scatter(
                x=result["train_dates"], y=result["train_pred"],
                name="Fitted (Train)", line=dict(color="#ffcc00", width=1.5, dash="dot")))

            # Test predictions
            if result.get("test_dates"):
                fig_pred.add_trace(go.Scatter(
                    x=result["test_dates"], y=result["test_pred"],
                    name="Predicted (Test)", line=dict(color="#00ff88", width=2)))

            # Future predictions
            fig_pred.add_trace(go.Scatter(
                x=result["future_dates"], y=result["future_pred"],
                name=f"Forecast (+{pred_days}d)", line=dict(color="#ff6644", width=2, dash="dash"),
                fill="tozeroy", fillcolor="rgba(255,102,68,0.05)"))

            fig_pred.update_layout(
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
                font=dict(family="DM Sans", color="#94a3b8"),
                legend=dict(bgcolor="rgba(17,24,39,0.9)", bordercolor="#1f2937", borderwidth=1),
                xaxis=dict(gridcolor="#1f2937"), yaxis=dict(gridcolor="#1f2937"),
                height=350, margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("RMSE", f"${result['rmse']:.2f}")
            mc2.metric("MAE", f"${result['mae']:.2f}")
            mc3.metric("R² Score", f"{result['r2']:.3f}")
            mc4.metric("Next Day Pred.", f"${result['future_pred'][0]:.2f}",
                       delta=f"{result['future_pred'][0]-current_price:+.2f}")
        else:
            st.info("👈 Select a model and click **Run Prediction** to forecast future prices.")

    # ─── Alerts Panel ─────────────────────────────────────────────────────────
    if st.session_state.alerts:
        st.markdown("---")
        st.markdown("### 🔔 Recent Alerts")
        for alert in st.session_state.alerts[:5]:
            icon = "🟢" if alert["severity"] == "positive" else "🔴" if alert["severity"] == "negative" else "🟡"
            st.warning(f"{icon} **{alert['title']}** — {alert['message']} *(triggered at {alert['time']})*")

    # ─── Data Table ───────────────────────────────────────────────────────────
    with st.expander("📋 Raw Data Table"):
        display_cols = ["Open", "High", "Low", "Close", "Volume"] + \
                       [c for c in df.columns if c.startswith("SMA_") or c.startswith("EMA_") or c in ("RSI", "MACD")]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[display_cols].tail(100).style.format("{:.2f}").background_gradient(
                subset=["Close"], cmap="RdYlGn"),
            use_container_width=True,
        )

    # ─── Export ───────────────────────────────────────────────────────────────
    with st.expander("📥 Export Data"):
        csv = df.to_csv()
        st.download_button("Download CSV", csv, f"{ticker}_analysis.csv", "text/csv", use_container_width=True)

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.exception(e)
