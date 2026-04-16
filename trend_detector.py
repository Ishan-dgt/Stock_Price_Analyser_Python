"""
Trend Detector
Identifies market trends, crossover signals, and trade signals.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime


class TrendDetector:
    """
    Detects:
      • Overall trend (Bullish / Bearish / Neutral) from moving averages
      • Golden Cross / Death Cross crossover events
      • Buy / Sell / Hold signals from RSI, MACD, and MA crossovers
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.current_price = float(df["Close"].iloc[-1])

    # ─────────────────────────────────────────────────────────────────────────
    # Trend
    # ─────────────────────────────────────────────────────────────────────────

    def detect_trend(self, sma_windows: List[int] = None) -> Dict[str, Any]:
        """
        Determine overall trend by counting how many SMAs are below the
        current closing price.

        Returns a dict with keys: overall, description, score, details
        """
        sma_windows = sorted(sma_windows or [20, 50, 200])
        available = [w for w in sma_windows if f"SMA_{w}" in self.df.columns]

        if not available:
            return {"overall": "NEUTRAL", "description": "No SMAs available", "score": 0, "details": []}

        bullish_count = 0
        details = []
        for w in available:
            sma_val = float(self.df[f"SMA_{w}"].iloc[-1])
            is_above = self.current_price > sma_val
            direction = "above" if is_above else "below"
            details.append({
                "window": w,
                "sma": round(sma_val, 4),
                "price_vs_sma": direction,
                "bullish": is_above,
            })
            if is_above:
                bullish_count += 1

        score = bullish_count / len(available)

        if score >= 0.67:
            overall = "BULLISH"
            description = f"Price above {bullish_count}/{len(available)} SMAs — uptrend"
        elif score <= 0.33:
            overall = "BEARISH"
            description = f"Price below {len(available)-bullish_count}/{len(available)} SMAs — downtrend"
        else:
            overall = "NEUTRAL"
            description = "Mixed signals — consolidation or trend reversal"

        return {
            "overall": overall,
            "description": description,
            "score": round(score, 2),
            "details": details,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Crossovers
    # ─────────────────────────────────────────────────────────────────────────

    def detect_crossovers(self, sma_windows: List[int] = None) -> List[Dict[str, Any]]:
        """
        Detect Golden Cross and Death Cross events between consecutive SMA pairs.

        Golden Cross: shorter SMA crosses above longer SMA → bullish
        Death Cross:  shorter SMA crosses below longer SMA → bearish
        """
        sma_windows = sorted(sma_windows or [50, 200])
        available = [w for w in sma_windows if f"SMA_{w}" in self.df.columns]

        if len(available) < 2:
            return []

        crossovers = []
        pairs = [(available[i], available[i + 1]) for i in range(len(available) - 1)]

        for short_w, long_w in pairs:
            short_col = f"SMA_{short_w}"
            long_col = f"SMA_{long_w}"

            short_sma = self.df[short_col]
            long_sma = self.df[long_col]

            # Current position: 1 if short is above long, else -1
            above = (short_sma > long_sma).astype(int)
            crossings = above.diff()

            golden_dates = self.df.index[crossings == 1]
            death_dates = self.df.index[crossings == -1]

            for dt in golden_dates:
                crossovers.append({
                    "type": "GOLDEN_CROSS",
                    "date": str(dt)[:10],
                    "short_ma": short_w,
                    "long_ma": long_w,
                    "price": round(float(self.df.loc[dt, "Close"]), 4),
                    "description": f"SMA {short_w} crossed above SMA {long_w}",
                })
            for dt in death_dates:
                crossovers.append({
                    "type": "DEATH_CROSS",
                    "date": str(dt)[:10],
                    "short_ma": short_w,
                    "long_ma": long_w,
                    "price": round(float(self.df.loc[dt, "Close"]), 4),
                    "description": f"SMA {short_w} crossed below SMA {long_w}",
                })

        # Sort chronologically
        crossovers.sort(key=lambda x: x["date"])
        return crossovers

    # ─────────────────────────────────────────────────────────────────────────
    # Composite Signals
    # ─────────────────────────────────────────────────────────────────────────

    def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generate BUY / SELL / HOLD signals based on:
          • RSI overbought/oversold
          • MACD crossover (MACD line crosses Signal line)
          • Price vs SMA 50 (if available)
        """
        signals = []
        df = self.df

        # ── RSI signals ───────────────────────────────────────────────────────
        if "RSI" in df.columns:
            rsi_series = df["RSI"].dropna()
            if len(rsi_series) >= 2:
                # Oversold reversal → BUY
                prev_rsi = float(rsi_series.iloc[-2])
                curr_rsi = float(rsi_series.iloc[-1])
                if prev_rsi < 30 and curr_rsi >= 30:
                    signals.append({"type": "BUY", "reason": "RSI exiting oversold (< 30)",
                                    "date": str(df.index[-1])[:10], "confidence": 0.6})
                elif prev_rsi > 70 and curr_rsi <= 70:
                    signals.append({"type": "SELL", "reason": "RSI exiting overbought (> 70)",
                                    "date": str(df.index[-1])[:10], "confidence": 0.6})
                elif curr_rsi < 30:
                    signals.append({"type": "BUY", "reason": f"RSI oversold ({curr_rsi:.1f})",
                                    "date": str(df.index[-1])[:10], "confidence": 0.5})
                elif curr_rsi > 70:
                    signals.append({"type": "SELL", "reason": f"RSI overbought ({curr_rsi:.1f})",
                                    "date": str(df.index[-1])[:10], "confidence": 0.5})

        # ── MACD crossover signals ────────────────────────────────────────────
        if "MACD" in df.columns and "MACD_signal" in df.columns:
            macd = df["MACD"].dropna()
            sig = df["MACD_signal"].dropna()
            common_idx = macd.index.intersection(sig.index)
            if len(common_idx) >= 2:
                prev_macd, curr_macd = float(macd[common_idx[-2]]), float(macd[common_idx[-1]])
                prev_sig, curr_sig = float(sig[common_idx[-2]]), float(sig[common_idx[-1]])
                if prev_macd <= prev_sig and curr_macd > curr_sig:
                    signals.append({"type": "BUY", "reason": "MACD crossed above Signal line",
                                    "date": str(df.index[-1])[:10], "confidence": 0.65})
                elif prev_macd >= prev_sig and curr_macd < curr_sig:
                    signals.append({"type": "SELL", "reason": "MACD crossed below Signal line",
                                    "date": str(df.index[-1])[:10], "confidence": 0.65})

        # ── Price vs SMA50 ────────────────────────────────────────────────────
        if "SMA_50" in df.columns:
            price = float(df["Close"].iloc[-1])
            sma50 = float(df["SMA_50"].iloc[-1])
            prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else price
            prev_sma50 = float(df["SMA_50"].iloc[-2]) if len(df) > 1 else sma50

            if prev_price <= prev_sma50 and price > sma50:
                signals.append({"type": "BUY", "reason": "Price crossed above SMA 50",
                                "date": str(df.index[-1])[:10], "confidence": 0.55})
            elif prev_price >= prev_sma50 and price < sma50:
                signals.append({"type": "SELL", "reason": "Price crossed below SMA 50",
                                "date": str(df.index[-1])[:10], "confidence": 0.55})

        # ── Fallback ──────────────────────────────────────────────────────────
        if not signals:
            signals.append({"type": "HOLD", "reason": "No actionable signal at this time",
                             "date": str(df.index[-1])[:10], "confidence": 0.0})

        return signals

    # ─────────────────────────────────────────────────────────────────────────
    # Support / Resistance
    # ─────────────────────────────────────────────────────────────────────────

    def find_support_resistance(self, window: int = 20) -> Dict[str, float]:
        """
        Simple S/R detection using rolling min (support) and max (resistance).
        """
        support = float(self.df["Low"].rolling(window).min().iloc[-1])
        resistance = float(self.df["High"].rolling(window).max().iloc[-1])
        return {
            "support": round(support, 4),
            "resistance": round(resistance, 4),
            "current": round(self.current_price, 4),
            "pct_to_resistance": round((resistance - self.current_price) / self.current_price * 100, 2),
            "pct_to_support": round((self.current_price - support) / self.current_price * 100, 2),
        }
