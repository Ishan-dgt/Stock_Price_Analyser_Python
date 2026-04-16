"""
Stock Price Predictor
Supports Linear Regression, Random Forest, and Gradient Boosting models.
Trains on engineered features and forecasts future prices.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Any, Tuple

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


class StockPredictor:
    """
    Trains a supervised regression model on technical features derived from
    historical OHLCV data and predicts future closing prices.

    Features used:
      • Lagged closes (1, 2, 3, 5, 10 days)
      • Log returns (1, 5 days)
      • Rolling mean / std (5, 10, 20 days)
      • RSI, MACD, BB_%B (if present in df)
      • Day-of-week, month (calendar)
    """

    MODEL_MAP = {
        "linear_regression": LinearRegression,
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
    }

    def __init__(self, df: pd.DataFrame, model_type: str = "random_forest"):
        if model_type not in self.MODEL_MAP:
            raise ValueError(
                f"Unknown model '{model_type}'. Choose from: {list(self.MODEL_MAP.keys())}"
            )
        self.df = df.copy()
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols: list = []

    # ─────────────────────────────────────────────────────────────────────────
    # Feature Engineering
    # ─────────────────────────────────────────────────────────────────────────

    def _build_features(self) -> pd.DataFrame:
        df = self.df.copy()
        close = df["Close"]

        # Lagged closes
        for lag in [1, 2, 3, 5, 10]:
            df[f"lag_{lag}"] = close.shift(lag)

        # Log returns
        df["log_ret_1"] = np.log(close / close.shift(1))
        df["log_ret_5"] = np.log(close / close.shift(5))

        # Rolling statistics
        for win in [5, 10, 20]:
            df[f"roll_mean_{win}"] = close.rolling(win).mean()
            df[f"roll_std_{win}"] = close.rolling(win).std()

        # Price range
        if "High" in df.columns and "Low" in df.columns:
            df["hl_range"] = df["High"] - df["Low"]

        # Calendar
        if hasattr(df.index, "dayofweek"):
            df["dayofweek"] = df.index.dayofweek
            df["month"] = df.index.month

        # Technical indicators already in df
        for col in ["RSI", "MACD", "BB_%B", "ATR"]:
            if col in df.columns:
                df[col] = df[col]  # already present

        df = df.dropna()
        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        exclude = {"Open", "High", "Low", "Close", "Volume",
                   "BB_upper", "BB_lower", "BB_middle", "BB_width",
                   "MACD_signal", "MACD_hist", "Volume_MA", "OBV"}
        # Also exclude SMA/EMA columns to avoid leakage (they depend on Close)
        cols = [
            c for c in df.columns
            if c not in exclude
            and not c.startswith("SMA_")
            and not c.startswith("EMA_")
        ]
        return cols

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Extract X (features) and y (next-day close) arrays."""
        feat_df = df[self.feature_cols]
        y_raw = df["Close"].shift(-1).dropna()  # predict next day's close
        common = y_raw.index.intersection(feat_df.dropna().index)
        X = feat_df.loc[common].values
        y = y_raw.loc[common].values
        return X, y, common

    def train_and_predict(self, future_days: int = 7, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Full pipeline: feature engineering → train/test split → fit → evaluate → forecast.

        Returns a result dict compatible with the Streamlit dashboard.
        """
        feature_df = self._build_features()
        self.feature_cols = self._get_feature_columns(feature_df)

        X, y, idx = self._prepare_xy(feature_df)

        if len(X) < 30:
            raise ValueError("Not enough data to train. Need at least 30 rows after feature engineering.")

        # Split
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        idx_train, idx_test = idx[:split], idx[split:]

        # Scale
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Build and train model
        kwargs = {}
        if self.model_type in ("random_forest", "gradient_boosting"):
            kwargs = {"n_estimators": 100, "random_state": 42}

        self.model = self.MODEL_MAP[self.model_type](**kwargs)
        self.model.fit(X_train_s, y_train)

        # In-sample predictions
        train_pred = self.model.predict(X_train_s)
        test_pred = self.model.predict(X_test_s)

        # Metrics on test set
        rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
        mae = float(mean_absolute_error(y_test, test_pred))
        r2 = float(r2_score(y_test, test_pred))

        # ── Future Forecast ──────────────────────────────────────────────────
        future_preds = self._forecast_future(feature_df, future_days)
        last_date = feature_df.index[-1]
        freq = self._infer_freq(feature_df.index)
        future_dates = [last_date + (i + 1) * freq for i in range(future_days)]

        return {
            "actual": list(feature_df["Close"].values),
            "actual_dates": list(feature_df.index),
            "train_pred": list(train_pred),
            "train_dates": list(idx_train),
            "test_pred": list(test_pred),
            "test_dates": list(idx_test),
            "future_pred": [round(p, 4) for p in future_preds],
            "future_dates": future_dates,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "model": self.model_type,
            "n_features": len(self.feature_cols),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Iterative Future Forecasting
    # ─────────────────────────────────────────────────────────────────────────

    def _forecast_future(self, feature_df: pd.DataFrame, n_days: int) -> list:
        """
        Iteratively append predicted closes to the working DataFrame and
        re-engineer features to forecast `n_days` into the future.
        """
        working = feature_df.copy()
        preds = []

        for i in range(n_days):
            # Rebuild features on extended window
            extended = self._build_features_incremental(working)
            if extended is None or len(extended) == 0:
                break

            last_row = extended.iloc[-1:]
            X_last = last_row[self.feature_cols].values
            if np.any(np.isnan(X_last)):
                # Fill NaN with column mean as fallback
                col_means = extended[self.feature_cols].mean()
                last_row = last_row.fillna(col_means)
                X_last = last_row[self.feature_cols].values

            X_scaled = self.scaler.transform(X_last)
            pred = float(self.model.predict(X_scaled)[0])
            preds.append(pred)

            # Append synthetic row with predicted Close
            new_date = working.index[-1] + self._infer_freq(working.index)
            new_row = pd.DataFrame({"Close": [pred], "Open": [pred], "High": [pred],
                                    "Low": [pred], "Volume": [int(working["Volume"].mean())]},
                                   index=[new_date])
            working = pd.concat([working, new_row])

        return preds if preds else [float(feature_df["Close"].iloc[-1])]

    def _build_features_incremental(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lightweight feature rebuild for forecasting loop."""
        try:
            return self._build_features_on(df)
        except Exception:
            return None

    def _build_features_on(self, df: pd.DataFrame) -> pd.DataFrame:
        """Same as _build_features but accepts an arbitrary DataFrame."""
        close = df["Close"]
        out = df.copy()
        for lag in [1, 2, 3, 5, 10]:
            out[f"lag_{lag}"] = close.shift(lag)
        out["log_ret_1"] = np.log(close / close.shift(1))
        out["log_ret_5"] = np.log(close / close.shift(5))
        for win in [5, 10, 20]:
            out[f"roll_mean_{win}"] = close.rolling(win, min_periods=1).mean()
            out[f"roll_std_{win}"] = close.rolling(win, min_periods=1).std()
        if "High" in out.columns and "Low" in out.columns:
            out["hl_range"] = out["High"] - out["Low"]
        if hasattr(out.index, "dayofweek"):
            out["dayofweek"] = out.index.dayofweek
            out["month"] = out.index.month
        return out.fillna(method="bfill").fillna(0)

    @staticmethod
    def _infer_freq(index: pd.DatetimeIndex) -> timedelta:
        if len(index) < 2:
            return timedelta(days=1)
        diffs = pd.Series(index).diff().dropna()
        median_diff = diffs.median()
        return median_diff if pd.notna(median_diff) else timedelta(days=1)
