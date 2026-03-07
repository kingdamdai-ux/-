"""データ取得ロジック（yfinance を使った実装）。"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def get_data(symbol: str) -> pd.DataFrame:
    """指定したシンボルの過去1年分の株価データを取得します。

    Args:
        symbol: 株式シンボル（例: AAPL, 7203.T）

    Returns:
        pandas.DataFrame: 日次株価データ（取得できない場合は空の DataFrame）
    """

    try:
        df = yf.download(tickers=symbol, period="1y", interval="1d", progress=False)

        # yfinance はシンボルが存在しない場合に空の DataFrame を返す
        if df is None or df.empty:
            return pd.DataFrame()

        # 日付をインデックスにしたまま返す
        return df

    except Exception:
        return pd.DataFrame()


# 旧 API 互換のためのラッパー
def fetch_stock_data(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """互換性のために古いインターフェースを維持します。"""

    # TODO: start/end をサポートする実装に拡張
    return get_data(symbol)
