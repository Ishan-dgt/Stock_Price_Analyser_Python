# 📈 Real-Time Stock Price Analyzer

A production-grade Python system for fetching, analyzing, and visualizing live stock market data — with technical indicators, trend detection, ML price prediction, and multi-channel alerts.

---

## 🗂 Project Structure

```
stock_analyzer/
├── app.py                  # Streamlit interactive dashboard
├── cli.py                  # Command-line interface
├── streamer.py             # Real-time terminal streaming loop
├── requirements.txt
│
├── utils/
│   ├── data_fetcher.py     # yfinance wrapper + data cleaning
│   ├── indicators.py       # SMA, EMA, RSI, MACD, Bollinger, ATR, OBV
│   └── trend_detector.py  # Trend analysis, crossovers, buy/sell signals
│
├── models/
│   └── predictor.py        # ML prediction (LinReg / RF / GBM)
│
└── alerts/
    └── alert_manager.py    # Email (SMTP) + Telegram Bot alerts
```

---

## ⚡ Quick Start

### 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### 2 — Launch the Dashboard (Streamlit)

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### 3 — Use the CLI

```bash
# Basic analysis
python cli.py --ticker AAPL

# Custom date range with prediction
python cli.py --ticker TSLA --days 365 --sma 20 50 200 --ema 12 26 --predict --predict-days 14

# Indian stocks
python cli.py --ticker INFY.NS --days 365

# With price alerts (prints to console)
python cli.py --ticker MSFT --alert-high 460 --alert-low 380

# Save charts to specific directory
python cli.py --ticker NVDA --output-dir ./charts
```

### 4 — Real-Time Terminal Stream

```bash
# Refresh every 30 seconds
python streamer.py --ticker AAPL

# Custom interval with alert thresholds
python streamer.py --ticker TSLA --interval 10 --alert-high 250 --alert-low 180
```

---

## 🧩 Features

### 📥 Data Fetching
- Powered by **yfinance** — no API key required
- Supports any global ticker: `AAPL`, `TSLA`, `INFY.NS`, `BTC-USD`, `^GSPC`, etc.
- Intervals: `1m`, `5m`, `15m`, `30m`, `1h`, `1d`, `1wk`
- Automatic date-range clamping for intraday data (yfinance limits)
- Built-in caching to avoid redundant requests

### 📊 Technical Indicators
| Indicator | Description |
|-----------|-------------|
| **SMA** | Simple Moving Average (any window) |
| **EMA** | Exponential Moving Average (any window) |
| **RSI** | Relative Strength Index (14-period) |
| **MACD** | MACD line, Signal line, Histogram |
| **Bollinger Bands** | Upper/Lower/Width/%B |
| **ATR** | Average True Range (volatility) |
| **OBV** | On-Balance Volume |
| **Volume MA** | 20-period volume moving average |

### 🔍 Trend Detection
- **Overall trend**: Bullish / Bearish / Neutral based on price vs SMA alignment
- **Golden Cross**: Short SMA crosses above long SMA → bullish
- **Death Cross**: Short SMA crosses below long SMA → bearish
- **Composite signals**: Combines RSI, MACD crossover, and price vs SMA 50

### 🤖 ML Price Prediction
Three models available:
- **Linear Regression** — fast baseline
- **Random Forest** — handles non-linearity, good default
- **Gradient Boosting** — highest accuracy on most datasets

Features used: lagged prices, log returns, rolling stats, RSI, MACD, BB%B, calendar features

Outputs: fitted vs actual chart, RMSE / MAE / R² metrics, N-day forecast

### 🔔 Alert System
- **Price threshold alerts**: trigger when price goes above/below a target
- **MA crossover alerts**: notify on Golden/Death Cross on today's date
- **Email delivery**: via SMTP (works with Gmail, Outlook, etc.)
- **Telegram delivery**: via Bot API (instant push notifications)

To enable email alerts in the dashboard, expand the "Email Alerts" section in the sidebar and fill in your SMTP credentials. For Gmail, use an **App Password** (not your regular password).

To enable Telegram alerts, create a bot via [@BotFather](https://t.me/botfather), get the token, and find your chat ID via [@userinfobot](https://t.me/userinfobot).

---

## 📧 Email Alert Setup (Gmail)

1. Enable 2FA on your Google account
2. Go to **Google Account → Security → App Passwords**
3. Create an app password for "Mail"
4. Use that 16-char password in the dashboard

---

## 📱 Telegram Alert Setup

```bash
# 1. Message @BotFather on Telegram → /newbot → copy token
# 2. Message @userinfobot → copy your Chat ID
# 3. Enter both in the dashboard sidebar
```

---

## 🛠 Extending

### Add a new indicator

```python
# utils/indicators.py
def add_stochastic(self, k_period=14, d_period=3) -> pd.DataFrame:
    low_min  = self.df["Low"].rolling(k_period).min()
    high_max = self.df["High"].rolling(k_period).max()
    self.df["%K"] = 100 * (self.df["Close"] - low_min) / (high_max - low_min)
    self.df["%D"] = self.df["%K"].rolling(d_period).mean()
    return self.df
```

### Add a new ML model

```python
# models/predictor.py  — add to MODEL_MAP
from sklearn.svm import SVR
MODEL_MAP = {
    ...
    "svr": SVR,
}
```

---

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**. It does not constitute financial advice. Past price data and ML model predictions are not a reliable indicator of future results. Always do your own research before making investment decisions.
