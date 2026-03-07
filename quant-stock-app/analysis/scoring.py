"""クオンツスコア計算モジュール。"""

from __future__ import annotations

import pandas as pd


def score_stock(df: pd.DataFrame) -> float:
    """最新データを使ってシンプルなクオンツスコアを計算します。

    ルール:
      - RSI < 40  -> +30
      - MA50 > MA200 -> +40

    Args:
        df: インジケーター列が追加された日次株価データ

    Returns:
        float: スコア
    """

    if df is None or df.empty:
        return 0.0

    last = df.iloc[-1]

    score = 0.0

    rsi = float(last.get("RSI", 0)) if "RSI" in df.columns else 0.0
    ma50 = float(last.get("MA50", 0)) if "MA50" in df.columns else 0.0
    ma200 = float(last.get("MA200", 0)) if "MA200" in df.columns else 0.0

    if rsi and rsi < 40:
        score += 30

    if ma50 and ma200 and ma50 > ma200:
        score += 40

    return score
