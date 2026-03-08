"""クオンツスコア計算モジュール。"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ScoreConfig:
    """スコア計算の設定値。

    Attributes:
        rsi_threshold: RSI がこの値以下なら加点対象
        rsi_score: RSI 条件にマッチした場合の加点
        ma_cross_score: MA50 > MA200 の場合の加点
    """

    rsi_threshold: float = 40.0
    rsi_score: float = 30.0
    ma_cross_score: float = 40.0


def score_stock(df: pd.DataFrame, config: ScoreConfig | None = None) -> float:
    """最新データを使ってシンプルなクオンツコアを計算します。

    Args:
        df: インジケーター列が追加された日次株価データ
        config: スコア計算の設定（未指定時はデフォルト設定を使用）

    Returns:
        float: スコア
    """

    if config is None:
        config = ScoreConfig()

    if df is None or df.empty:
        return 0.0

    last = df.iloc[-1]

    score = 0.0

    rsi = float(last.get("RSI", 0)) if "RSI" in df.columns else 0.0
    ma50 = float(last.get("MA50", 0)) if "MA50" in df.columns else 0.0
    ma200 = float(last.get("MA200", 0)) if "MA200" in df.columns else 0.0

    if rsi and rsi < config.rsi_threshold:
        score += config.rsi_score

    if ma50 and ma200 and ma50 > ma200:
        score += config.ma_cross_score

    return score
