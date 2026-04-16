"""
Real-Time Streaming Simulator
Continuously polls yfinance for fresh 1-minute data and prints live updates.

Run: python streamer.py --ticker AAPL --interval 30
"""

import argparse
import os
import sys
import time
import signal
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_fetcher import StockDataFetcher
from utils.indicators import TechnicalIndicators
from utils.trend_detector import TrendDetector
from alerts.alert_manager import AlertManager


# ─── Colour helpers ───────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def colour(text, code):
    return f"{code}{text}{RESET}"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


# ─── Live Tick Printer ────────────────────────────────────────────────────────

def print_live_tick(ticker, df, prev_price, alert_mgr, high_threshold, low_threshold):
    current = float(df["Close"].iloc[-1])
    change  = current - prev_price
    pct     = change / prev_price * 100 if prev_price else 0
    arrow   = "▲" if change > 0 else "▼" if change < 0 else "─"
    col     = GREEN if change > 0 else RED if change < 0 else YELLOW

    td   = TrendDetector(df)
    trend = td.detect_trend([20, 50])
    sig   = td.generate_signals()[-1]

    rsi_val = "--"
    if "RSI" in df.columns:
        rsi_val = f"{df['RSI'].iloc[-1]:.1f}"

    macd_val = "--"
    if "MACD" in df.columns:
        macd_val = f"{df['MACD'].iloc[-1]:.4f}"

    sig_col = GREEN if sig["type"] == "BUY" else RED if sig["type"] == "SELL" else YELLOW

    print(colour(f"\n{'═'*55}", CYAN))
    print(colour(f"  📡  {ticker} — LIVE STREAM", BOLD))
    print(colour(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}", CYAN))
    print(colour(f"{'─'*55}", CYAN))
    print(f"  Price   : {colour(f'${current:,.2f}', col)}  "
          f"{colour(f'{arrow} {abs(change):.2f} ({abs(pct):.2f}%)', col)}")
    print(f"  Trend   : {colour(trend['overall'], GREEN if trend['overall']=='BULLISH' else RED if trend['overall']=='BEARISH' else YELLOW)}")
    print(f"  RSI     : {rsi_val}   MACD: {macd_val}")
    print(f"  Signal  : {colour(sig['type'], sig_col)}  — {sig['reason']}")
    print(colour(f"{'═'*55}", CYAN))

    # Check alerts
    triggered = alert_mgr.check_price_alerts(ticker, current, high_threshold, low_threshold)
    for a in triggered:
        col2 = GREEN if a["severity"] == "positive" else RED
        print(colour(f"\n  🔔 ALERT: {a['title']}", col2))
        print(colour(f"     {a['message']}", col2))

    return current


# ─── Main Loop ────────────────────────────────────────────────────────────────

def stream(args):
    fetcher    = StockDataFetcher()
    alert_mgr  = AlertManager()
    prev_price = None
    running    = True

    def handle_exit(sig, frame):
        nonlocal running
        print(colour("\n\n  👋 Stream stopped by user.\n", YELLOW))
        running = False
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    print(colour(f"\n  🚀 Starting live stream for {args.ticker}", BOLD))
    print(colour(f"  Refresh every {args.interval}s | Ctrl+C to stop\n", CYAN))

    while running:
        try:
            end   = datetime.now()
            start = end - timedelta(days=30)  # rolling 30-day window
            df = fetcher.fetch(args.ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "1d")

            if df is None or df.empty:
                print(colour(f"  ⚠️  No data for {args.ticker}. Retrying...", YELLOW))
                time.sleep(args.interval)
                continue

            # Add indicators
            ti = TechnicalIndicators(df)
            df = ti.add_sma([20, 50])
            df = ti.add_ema([12, 26])
            df = ti.add_rsi()
            df = ti.add_macd()

            if prev_price is None:
                prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else float(df["Close"].iloc[-1])

            prev_price = print_live_tick(
                args.ticker, df, prev_price,
                alert_mgr, args.alert_high, args.alert_low,
            )

        except Exception as e:
            print(colour(f"  ❌ Error: {e}", RED))

        time.sleep(args.interval)


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Stock Streamer")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds")
    parser.add_argument("--alert-high", type=float, default=0.0)
    parser.add_argument("--alert-low",  type=float, default=0.0)
    stream(parser.parse_args())
