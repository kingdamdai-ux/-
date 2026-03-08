import pandas as pd

from analysis.scoring import ScoreConfig, score_stock


def test_score_stock_default_config():
    # RSI が 40 以下、MA50 > MA200 のときに全加点されること
    df = pd.DataFrame(
        {
            "RSI": [30.0],
            "MA50": [120.0],
            "MA200": [100.0],
        }
    )

    assert score_stock(df) == 70.0


def test_score_stock_custom_config():
    # カスタム設定値でスコアが変化すること
    df = pd.DataFrame(
        {
            "RSI": [50.0],
            "MA50": [120.0],
            "MA200": [100.0],
        }
    )

    config = ScoreConfig(rsi_threshold=60.0, rsi_score=10.0, ma_cross_score=20.0)

    assert score_stock(df, config=config) == 30.0


def test_score_stock_empty_dataframe_returns_zero():
    assert score_stock(pd.DataFrame()) == 0.0
