"""Scoring helpers for broad market screening and buy recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreConfig:
    name: str = "中期"
    rsi_lower_bound: float = 40.0
    rsi_upper_bound: float = 65.0
    trend_score: float = 30.0
    momentum_score: float = 25.0
    pullback_score: float = 20.0
    risk_score: float = 15.0
    relative_strength_score: float = 10.0
    max_volatility: float = 35.0
    breakout_drawdown_floor: float = -12.0
    breakout_drawdown_ceiling: float = -2.0
    strong_buy_threshold: float = 70.0
    buy_threshold: float = 55.0
    watch_threshold: float = 40.0


@dataclass(frozen=True)
class ScoreResult:
    total: float
    trend: float
    momentum: float
    pullback: float
    risk: float
    relative_strength: float
    recommendation: str


HORIZON_ORDER = ("短期", "中期", "長期")


def _scalarize(value: object, default: float) -> float:
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return default
        value = value.iloc[0, 0]
    elif isinstance(value, pd.Series):
        if value.empty:
            return default
        value = value.iloc[0]
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    elif isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]

    if pd.isna(value):
        return default
    return float(value)


def _to_float(last: pd.Series, key: str, default: float = 0.0) -> float:
    return _scalarize(last.get(key, default), default)


def get_score_config(horizon: str) -> ScoreConfig:
    if horizon == "短期":
        return ScoreConfig(
            name="短期",
            rsi_lower_bound=45.0,
            rsi_upper_bound=72.0,
            trend_score=20.0,
            momentum_score=35.0,
            pullback_score=15.0,
            risk_score=10.0,
            relative_strength_score=20.0,
            max_volatility=45.0,
            breakout_drawdown_floor=-8.0,
            breakout_drawdown_ceiling=0.0,
            strong_buy_threshold=72.0,
            buy_threshold=58.0,
            watch_threshold=42.0,
        )
    if horizon == "長期":
        return ScoreConfig(
            name="長期",
            rsi_lower_bound=45.0,
            rsi_upper_bound=62.0,
            trend_score=40.0,
            momentum_score=15.0,
            pullback_score=15.0,
            risk_score=20.0,
            relative_strength_score=10.0,
            max_volatility=30.0,
            breakout_drawdown_floor=-18.0,
            breakout_drawdown_ceiling=-4.0,
            strong_buy_threshold=68.0,
            buy_threshold=54.0,
            watch_threshold=38.0,
        )
    return ScoreConfig(name="中期")


def score_stock_details(df: pd.DataFrame, config: ScoreConfig | None = None) -> ScoreResult:
    if config is None:
        config = ScoreConfig()

    if df is None or df.empty:
        return ScoreResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "No Data")

    last = df.iloc[-1]

    rsi = _to_float(last, "RSI", 50.0)
    ma20 = _to_float(last, "MA20")
    ma50 = _to_float(last, "MA50")
    ma200 = _to_float(last, "MA200")
    ema20 = _to_float(last, "EMA20")
    ema50 = _to_float(last, "EMA50")
    momentum21 = _to_float(last, "Momentum21")
    volatility21 = _to_float(last, "Volatility21")
    drawdown252 = _to_float(last, "Drawdown252")
    distance_ma50 = _to_float(last, "DistanceMA50")

    trend = 0.0
    if ma50 > ma200:
        trend += config.trend_score * 0.5
    if ema20 > ema50:
        trend += config.trend_score * 0.25
    if ma20 >= ma50 and distance_ma50 > -3.0:
        trend += config.trend_score * 0.25

    momentum = 0.0
    if config.rsi_lower_bound <= rsi <= config.rsi_upper_bound:
        momentum += config.momentum_score * 0.5
    elif 35.0 <= rsi < config.rsi_lower_bound:
        momentum += config.momentum_score * 0.25
    if momentum21 > 3.0:
        momentum += config.momentum_score * 0.5
    elif momentum21 > 0.0:
        momentum += config.momentum_score * 0.25

    pullback = 0.0
    if config.breakout_drawdown_floor <= drawdown252 <= config.breakout_drawdown_ceiling:
        pullback += config.pullback_score * 0.7
    elif -20.0 <= drawdown252 < config.breakout_drawdown_floor:
        pullback += config.pullback_score * 0.35
    if -5.0 <= distance_ma50 <= 8.0:
        pullback += config.pullback_score * 0.3

    risk = 0.0
    if 0.0 < volatility21 <= config.max_volatility:
        risk += config.risk_score
    elif volatility21 == 0.0:
        risk += config.risk_score * 0.5
    elif volatility21 <= config.max_volatility * 1.35:
        risk += config.risk_score * 0.4

    relative_strength = 0.0
    if ma20 > ma50 > ma200:
        relative_strength += config.relative_strength_score * 0.6
    if momentum21 > 8.0:
        relative_strength += config.relative_strength_score * 0.4
    elif momentum21 > 4.0:
        relative_strength += config.relative_strength_score * 0.2

    total = round(trend + momentum + pullback + risk + relative_strength, 1)

    if total >= config.strong_buy_threshold:
        recommendation = "Strong Buy"
    elif total >= config.buy_threshold:
        recommendation = "Buy"
    elif total >= config.watch_threshold:
        recommendation = "Watch"
    else:
        recommendation = "Avoid"

    return ScoreResult(
        total=total,
        trend=round(trend, 1),
        momentum=round(momentum, 1),
        pullback=round(pullback, 1),
        risk=round(risk, 1),
        relative_strength=round(relative_strength, 1),
        recommendation=recommendation,
    )


def score_stock(df: pd.DataFrame, config: ScoreConfig | None = None) -> float:
    return score_stock_details(df, config=config).total
