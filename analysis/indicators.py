"""Technical indicator helpers for the stock dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)
    return normalized


def _extract_close(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.astype(float)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend, momentum, and risk indicators to price data."""

    if df is None or df.empty:
        return df

    df = _normalize_columns(df)
    close = _extract_close(df)
    returns = close.pct_change().fillna(0.0)

    df["RSI"] = _rsi(close, window=14)
    df["MA20"] = close.rolling(window=20, min_periods=1).mean()
    df["MA50"] = close.rolling(window=50, min_periods=1).mean()
    df["MA200"] = close.rolling(window=200, min_periods=1).mean()
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["Momentum21"] = close.pct_change(21).mul(100).fillna(0.0)
    df["Volatility21"] = returns.rolling(window=21, min_periods=2).std().mul(np.sqrt(TRADING_DAYS) * 100).fillna(0.0)
    df["High252"] = close.rolling(window=TRADING_DAYS, min_periods=1).max()
    df["Drawdown252"] = ((close / df["High252"]) - 1.0).mul(100).fillna(0.0)
    df["DistanceMA50"] = ((close / df["MA50"]) - 1.0).mul(100).fillna(0.0)

    return df
