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

    # yfinance は環境やバージョンにより列構造が変わるため、Close を確実に 1 本の Series で取得する
    close: pd.Series | pd.DataFrame

    if "Close" in df.columns:
        close = df["Close"]
    elif isinstance(df.columns, pd.MultiIndex) and "Close" in df.columns.get_level_values(-1):
        # 例: (Ticker, Price) 形式
        close = df.xs("Close", axis=1, level=-1)
    else:
        raise ValueError("Close 列が見つからないため、インジケーターを計算できません。")

    if isinstance(close, pd.DataFrame):
        # 例: MultiIndex (Price, Ticker) 形式や重複列のケース
        close = close.iloc[:, 0]

    # 後続処理が常に単一列名 "Close" で扱えるよう正規化
    df["Close"] = close

    df["RSI"] = _rsi(close, window=14)
    df["MA50"] = close.rolling(window=50, min_periods=1).mean()
    df["MA200"] = close.rolling(window=200, min_periods=1).mean()

    return df
