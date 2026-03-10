import pandas as pd
import pytest

from fetch.fetch_data import DataFetchError, get_data, normalize_symbol


def test_normalize_symbol_rejects_empty_and_placeholder_values():
    assert normalize_symbol(None) is None
    assert normalize_symbol("") is None
    assert normalize_symbol("   ") is None
    assert normalize_symbol("None") is None
    assert normalize_symbol("nan") is None
    assert normalize_symbol(" aapl ") == "AAPL"


def test_get_data_rejects_invalid_symbol_before_yfinance_call(monkeypatch):
    called = False

    def fake_download(*args, **kwargs):
        nonlocal called
        called = True
        return pd.DataFrame()

    monkeypatch.setattr("fetch.fetch_data.yf.download", fake_download)

    with pytest.raises(DataFetchError, match="有効な銘柄コード"):
        get_data(None)

    assert called is False
