"""テクニカル指標の計算モジュール。"""

from __future__ import annotations

import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) を計算します。"""

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """指定した株価データフレームにインジケーター列を追加します。

    追加される列:
    - RSI
    - MA50
    - MA200

    Args:
        df: 日次株価データ (yfinance 形式)

    Returns:
        df にインジケーター列を追加した DataFrame
    """

    if df is None or df.empty:
        return df

    # 動作を安定させるため、インデックスを日付として扱う
    df = df.copy()

    # yfinance は単一銘柄でも MultiIndex カラムを返すことがあるため、Close シリーズを取り出す
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        # 例: MultiIndex (Price, Ticker) 形式
        close = close.iloc[:, 0]

    df["RSI"] = _rsi(close, window=14)
    df["MA50"] = close.rolling(window=50, min_periods=1).mean()
    df["MA200"] = close.rolling(window=200, min_periods=1).mean()

    return df
