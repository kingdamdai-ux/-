"""データ取得ロジック（yfinance を使った実装）。"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """データ取得に失敗した場合に投げる例外。"""

    def __init__(self, message: str, attempts: int = 0):
        super().__init__(message)
        self.attempts = attempts


def get_data(
    symbol: str,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    return_attempts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, int]:
    """指定したシンボルの過去1年分の株価データを取得します。

    Args:
        symbol: 株式シンボル（例: AAPL, 7203.T）
        max_retries: 取得に失敗した場合のリトライ回数
        backoff_seconds: リトライ時の待機秒数（指数バックオフ）
        return_attempts: True の場合、(df, attempts) を返す

    Returns:
        pandas.DataFrame: 日次株価データ

    Raises:
        DataFetchError: データ取得に失敗した場合。
    """

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(tickers=symbol, period="1y", interval="1d", progress=False)

            # yfinance はシンボルが存在しない場合に空の DataFrame を返す
            if df is None or df.empty:
                raise DataFetchError(f"{symbol} のデータが取得できませんでした。", attempts=attempt)

            # 日付をインデックスにしたまま返す
            return (df, attempt) if return_attempts else df

        except Exception as exc:
            logger.exception("%s のデータ取得に失敗しました。（試行 %s/%s）", symbol, attempt, max_retries)

            if attempt >= max_retries:
                raise DataFetchError(
                    f"{symbol} のデータ取得中にエラーが発生しました（{attempt} 回試行）：{exc}",
                    attempts=attempt,
                ) from exc

            # Exponential backoff
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))


# 旧 API 互換のためのラッパー
def fetch_stock_data(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """互換性のために古いインターフェースを維持します。"""

    # TODO: start/end をサポートする実装に拡張
    return get_data(symbol)
