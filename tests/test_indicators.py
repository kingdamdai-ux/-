import pandas as pd

from analysis.indicators import add_indicators


def test_add_indicators_returns_expected_columns_and_non_nan_values():
    df = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 103.0, 104.0, 106.0, 108.0, 110.0, 109.0, 111.0, 114.0,
                      116.0, 118.0, 117.0, 119.0, 122.0, 124.0, 126.0, 128.0, 129.0, 131.0,
                      133.0, 135.0, 136.0, 138.0, 140.0],
        },
        index=pd.date_range("2024-01-01", periods=25, freq="D"),
    )

    result = add_indicators(df)

    for column in ["RSI", "MA20", "MA50", "MA200", "EMA20", "Momentum21", "Volatility21", "Drawdown252"]:
        assert column in result.columns

    assert result["MA50"].iloc[0] == 100.0
    assert result["MA200"].iloc[0] == 100.0
    assert pd.notna(result["RSI"].iloc[-1])
    assert result["Momentum21"].iloc[-1] > 0.0
