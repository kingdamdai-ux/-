import pandas as pd

from analysis.scoring import HORIZON_ORDER, get_score_config, score_stock, score_stock_details


def test_score_stock_details_returns_buy_for_strong_setup():
    df = pd.DataFrame(
        {
            "RSI": [55.0],
            "MA20": [115.0],
            "MA50": [110.0],
            "MA200": [100.0],
            "EMA20": [114.0],
            "EMA50": [108.0],
            "Momentum21": [8.5],
            "Volatility21": [22.0],
            "Drawdown252": [-6.0],
            "DistanceMA50": [3.0],
        }
    )

    result = score_stock_details(df, config=get_score_config("中期"))

    assert result.total >= 55.0
    assert result.recommendation in {"Buy", "Strong Buy"}


def test_score_stock_respects_custom_config_by_horizon():
    df = pd.DataFrame(
        {
            "RSI": [58.0],
            "MA20": [102.0],
            "MA50": [101.0],
            "MA200": [100.0],
            "EMA20": [102.0],
            "EMA50": [101.0],
            "Momentum21": [4.0],
            "Volatility21": [20.0],
            "Drawdown252": [-4.0],
            "DistanceMA50": [1.0],
        }
    )

    short_score = score_stock(df, config=get_score_config("短期"))
    long_score = score_stock(df, config=get_score_config("長期"))

    assert short_score != long_score
    assert short_score > 0.0
    assert long_score > 0.0


def test_horizon_order_and_configs_are_available():
    assert HORIZON_ORDER == ("短期", "中期", "長期")
    assert get_score_config("短期").name == "短期"
    assert get_score_config("中期").name == "中期"
    assert get_score_config("長期").name == "長期"


def test_score_stock_empty_dataframe_returns_zero():
    assert score_stock(pd.DataFrame()) == 0.0
