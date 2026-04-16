"""
Technical Indicators
Adds SMA, EMA, RSI, MACD, Bollinger Bands, and Volume MA to a DataFrame.
"""

import pandas as pd
import numpy as np
from typing import List


class TechnicalIndicators:
    """
    Computes common technical indicators and appends them as columns
    to the OHLCV DataFrame passed at construction time.
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        required = {"Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {missing}")
        self.df = df.copy()

    # ── Moving Averages ───────────────────────────────────────────────────────

    def add_sma(self, windows: List[int]) -> pd.DataFrame:
        """Add Simple Moving Average columns: SMA_<window>."""
        for w in windows:
            self.df[f"SMA_{w}"] = self.df["Close"].rolling(window=w, min_periods=1).mean()
        return self.df

    def add_ema(self, windows: List[int]) -> pd.DataFrame:
        """Add Exponential Moving Average columns: EMA_<window>."""
        for w in windows:
            self.df[f"EMA_{w}"] = self.df["Close"].ewm(span=w, adjust=False).mean()
        return self.df

    # ── Momentum ─────────────────────────────────────────────────────────────

    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) column.
        RSI = 100 − 100 / (1 + RS) where RS = avg_gain / avg_loss
        """
        delta = self.df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.df["RSI"] = (100 - (100 / (1 + rs))).round(4)
        return self.df

    def add_macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """
        Add MACD, MACD_signal, and MACD_hist columns.
        MACD Line  = EMA(fast) − EMA(slow)
        Signal     = EMA(MACD, signal)
        Histogram  = MACD − Signal
        """
        ema_fast = self.df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df["Close"].ewm(span=slow, adjust=False).mean()
        self.df["MACD"] = (ema_fast - ema_slow).round(6)
        self.df["MACD_signal"] = self.df["MACD"].ewm(span=signal, adjust=False).mean().round(6)
        self.df["MACD_hist"] = (self.df["MACD"] - self.df["MACD_signal"]).round(6)
        return self.df

    # ── Volatility ────────────────────────────────────────────────────────────

    def add_bollinger_bands(self, window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Band columns: BB_middle, BB_upper, BB_lower, BB_width, BB_%B.
        """
        rolling = self.df["Close"].rolling(window=window, min_periods=1)
        middle = rolling.mean()
        std = rolling.std(ddof=0)

        self.df["BB_middle"] = middle.round(4)
        self.df["BB_upper"] = (middle + std_dev * std).round(4)
        self.df["BB_lower"] = (middle - std_dev * std).round(4)

        band_width = self.df["BB_upper"] - self.df["BB_lower"]
        self.df["BB_width"] = band_width.round(6)

        # %B = (Close − Lower) / (Upper − Lower)
        self.df["BB_%B"] = ((self.df["Close"] - self.df["BB_lower"]) /
                             band_width.replace(0, np.nan)).round(4)
        return self.df

    def add_atr(self, period: int = 14) -> pd.DataFrame:
        """Average True Range (ATR) — proxy for volatility."""
        high_low = self.df["High"] - self.df["Low"]
        high_pc = (self.df["High"] - self.df["Close"].shift()).abs()
        low_pc = (self.df["Low"] - self.df["Close"].shift()).abs()
        true_range = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
        self.df["ATR"] = true_range.ewm(span=period, adjust=False).mean().round(4)
        return self.df

    # ── Volume ────────────────────────────────────────────────────────────────

    def add_volume_ma(self, window: int = 20) -> pd.DataFrame:
        """Add a simple moving average of volume."""
        self.df["Volume_MA"] = self.df["Volume"].rolling(window=window, min_periods=1).mean().astype(int)
        return self.df

    def add_obv(self) -> pd.DataFrame:
        """On-Balance Volume: cumulative measure of buy/sell pressure."""
        direction = np.sign(self.df["Close"].diff().fillna(0))
        self.df["OBV"] = (direction * self.df["Volume"]).cumsum()
        return self.df

    # ── All at once ───────────────────────────────────────────────────────────

    def add_all(
        self,
        sma_windows: List[int] = None,
        ema_windows: List[int] = None,
    ) -> pd.DataFrame:
        """Convenience method to calculate every indicator."""
        sma_windows = sma_windows or [20, 50, 200]
        ema_windows = ema_windows or [12, 26]
        self.add_sma(sma_windows)
        self.add_ema(ema_windows)
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_atr()
        self.add_volume_ma()
        self.add_obv()
        return self.df
