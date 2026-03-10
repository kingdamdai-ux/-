"""Data fetching helpers backed by yfinance."""

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf


logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Raised when stock data could not be fetched."""

    def __init__(self, message: str, attempts: int = 0):
        super().__init__(message)
        self.attempts = attempts


def normalize_symbol(symbol: object) -> str | None:
    """Normalize a ticker symbol and reject unusable placeholder values."""

    if not isinstance(symbol, str):
        return None

    normalized = symbol.strip().upper()
    if not normalized or normalized in {"NONE", "NAN", "NULL"}:
        return None

    return normalized


def get_data(
    symbol: str,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    return_attempts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, int]:
    """Fetch 1 year of daily OHLCV data for a single ticker."""

    normalized_symbol = normalize_symbol(symbol)
    if normalized_symbol is None:
        raise DataFetchError("有効な銘柄コードが指定されていません。", attempts=0)

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(tickers=normalized_symbol, period="1y", interval="1d", progress=False)

            if df is None or df.empty:
                raise DataFetchError(f"{normalized_symbol} のデータが取得できませんでした。", attempts=attempt)

            return (df, attempt) if return_attempts else df

        except Exception as exc:
            logger.exception("%s のデータ取得に失敗しました。（試行 %s/%s）", normalized_symbol, attempt, max_retries)

            if attempt >= max_retries:
                raise DataFetchError(
                    f"{normalized_symbol} のデータ取得中にエラーが発生しました（{attempt} 回試行）：{exc}",
                    attempts=attempt,
                ) from exc

            time.sleep(backoff_seconds * (2 ** (attempt - 1)))


def fetch_stock_data(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Compatibility wrapper for the legacy API."""

    return get_data(symbol)
