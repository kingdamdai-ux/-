import pandas as pd

from analysis.indicators import add_indicators


def test_add_indicators_returns_expected_columns_and_values():
    # 3 日分の株価データを作ってインジケータが追加されるか確認
    df = pd.DataFrame(
        {
            "Close": [100.0, 105.0, 110.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    result = add_indicators(df)

    assert "RSI" in result.columns
    assert "MA50" in result.columns
    assert "MA200" in result.columns

    # 移動平均は最初の値はそのままになる
    assert result["MA50"].iloc[0] == 100.0
    assert result["MA200"].iloc[0] == 100.0

    # 終値の上昇が続いているので RSI は 50 を超えているはず
    assert result["RSI"].iloc[-1] > 50.0
