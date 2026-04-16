"""
Stock Data Fetcher
Wraps yfinance for clean, reliable data retrieval.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Fetches historical and near-real-time stock data via yfinance.
    """

    VALID_INTERVALS = {
        "1m", "2m", "5m", "15m", "30m", "60m", "90m",
        "1h", "1d", "5d", "1wk", "1mo", "3mo",
    }

    # yfinance limits on how far back intraday data goes
    INTRADAY_LIMITS = {
        "1m": 7, "2m": 60, "5m": 60, "15m": 60,
        "30m": 60, "60m": 730, "90m": 60, "1h": 730,
    }

    def __init__(self):
        self._cache: dict = {}

    def fetch(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for `ticker` between `start` and `end`.

        Args:
            ticker:      Stock symbol, e.g. "AAPL", "TSLA", "INFY.NS"
            start:       ISO date string, e.g. "2023-01-01"
            end:         ISO date string, e.g. "2024-01-01"
            interval:    Candle interval; one of VALID_INTERVALS
            auto_adjust: Adjust OHLC for splits/dividends (default True)

        Returns:
            Cleaned DataFrame or None on failure.
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Choose from: {self.VALID_INTERVALS}")

        # Clamp intraday start dates to yfinance limits
        if interval in self.INTRADAY_LIMITS:
            max_days = self.INTRADAY_LIMITS[interval]
            earliest_allowed = datetime.now() - timedelta(days=max_days)
            start_dt = datetime.fromisoformat(str(start))
            if start_dt < earliest_allowed:
                start = earliest_allowed.strftime("%Y-%m-%d")
                logger.warning(
                    f"Interval '{interval}' limited to {max_days} days. "
                    f"Adjusting start to {start}."
                )

        cache_key = f"{ticker}_{start}_{end}_{interval}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            tkr = yf.Ticker(ticker)
            df = tkr.history(start=start, end=end, interval=interval, auto_adjust=auto_adjust)

            if df.empty:
                logger.error(f"No data returned for {ticker}")
                return None

            df = self._clean(df)
            self._cache[cache_key] = df
            logger.info(f"Fetched {len(df)} rows for {ticker} [{interval}]")
            return df

        except Exception as exc:
            logger.error(f"Failed to fetch {ticker}: {exc}")
            raise

    def fetch_latest(self, ticker: str, n_days: int = 1) -> Optional[pd.DataFrame]:
        """Fetch the last `n_days` of daily data."""
        end = datetime.now()
        start = end - timedelta(days=n_days + 5)  # buffer for weekends/holidays
        return self.fetch(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "1d")

    def get_info(self, ticker: str) -> dict:
        """Return company metadata dict."""
        try:
            return yf.Ticker(ticker).info
        except Exception:
            return {}

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Standardise columns, drop nulls, and sort by date."""
        # Rename columns to consistent names
        rename_map = {
            "Open": "Open", "High": "High", "Low": "Low",
            "Close": "Close", "Volume": "Volume",
        }
        df = df.rename(columns=rename_map)

        # Keep only OHLCV
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()

        # Remove timezone info for plotting convenience
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Forward-fill then drop remaining NaNs
        df = df.ffill().dropna()

        # Ensure chronological order
        df = df.sort_index()

        # Cast types
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(int)

        return df
