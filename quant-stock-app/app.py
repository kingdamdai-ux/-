"""Streamlit app entry point for Quant Stock Ranking."""

from __future__ import annotations

import concurrent.futures
import datetime

import numpy as np
import pandas as pd
import streamlit as st

from analysis.indicators import add_indicators
from analysis.scoring import ScoreConfig, score_stock
from fetch.fetch_data import DataFetchError, get_data


# --- キャッシュ管理 ----------------------------------------------------------------

_DATA_CACHE: dict[str, pd.DataFrame] = {}
_CACHE_META: dict[str, dict] = {}


def clear_cache(symbol: str | None = None) -> None:
    """キャッシュをクリアします。

    Args:
        symbol: 指定したシンボルのみクリア。None の場合は全クリア。
    """

    if symbol:
        _DATA_CACHE.pop(symbol, None)
        _CACHE_META.pop(symbol, None)
    else:
        _DATA_CACHE.clear()
        _CACHE_META.clear()


def _get_cached_data(symbol: str, force_refresh: bool = False, max_retries: int = 3, backoff_seconds: float = 1.0) -> tuple[pd.DataFrame, int, str | None]:
    """キャッシュ付きでデータを取得し、メタ情報を返します。"""

    if not force_refresh and symbol in _DATA_CACHE:
        meta = _CACHE_META.get(symbol, {})
        return _DATA_CACHE[symbol], meta.get("attempts", 1), None

    try:
        df, attempts = get_data(
            symbol, max_retries=max_retries, backoff_seconds=backoff_seconds, return_attempts=True
        )
        _DATA_CACHE[symbol] = df
        _CACHE_META[symbol] = {"attempts": attempts, "fetched_at": datetime.datetime.utcnow()}
        return df, attempts, None
    except DataFetchError as exc:
        clear_cache(symbol)
        return pd.DataFrame(), getattr(exc, "attempts", 0) or 0, str(exc)


# --- モックデータ生成 ----------------------------------------------------------------


def _generate_mock_data(symbol: str, days: int = 252) -> pd.DataFrame:
    """モックデータ（ランダムウォーク）を生成します。"""

    today = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=today, periods=days)

    # random walk
    np.random.seed(abs(hash(symbol)) % (2**32))
    returns = np.random.normal(loc=0.0002, scale=0.02, size=len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"Close": prices}, index=dates)
    return df


# --- UI / ロジック ------------------------------------------------------------------


def _get_symbol_options() -> list[str]:
    """サイドバーに表示する銘柄セットを決定します。"""

    import data.symbols as symbols

    option = st.sidebar.radio(
        "銘柄セット",
        options=["全銘柄", "日本株", "米国株", "カスタム"],
        index=0,
    )

    if option == "日本株":
        return symbols.JP_STOCKS
    if option == "米国株":
        return symbols.US_STOCKS

    if option == "カスタム":
        raw = st.sidebar.text_input("カスタムシンボル (カンマ区切り)", value="")
        symbols_list = [s.strip().upper() for s in raw.split(",") if s.strip()]
        return symbols_list

    return symbols.ALL_SYMBOLS


def _render_symbol_chart(symbol: str, source: str) -> None:
    """選択した銘柄の株価と指標をチャート表示する。"""

    if source == "yfinance":
        df, _, error = _get_cached_data(symbol)
    else:
        df = _generate_mock_data(symbol)
        error = None

    if error or df is None or df.empty:
        st.warning(f"{symbol} のデータが取得できませんでした。{error or ''}")
        return

    df = add_indicators(df)

    st.subheader(f"{symbol} の推移")
    st.caption("終値 + MA50/MA200")
    st.line_chart(df[["Close", "MA50", "MA200"]])

    if "RSI" in df.columns:
        st.caption("RSI (14日)  (70 が過買い、30 が売られすぎの目安)")
        st.line_chart(df[["RSI"]])


def _score_symbol(symbol: str, source: str, config: ScoreConfig, force_refresh: bool) -> dict:
    """単一シンボルについてデータ取得・指標計算・スコアリングを行います。"""

    if source == "yfinance":
        df, attempts, error = _get_cached_data(symbol, force_refresh=force_refresh)
    else:
        df = _generate_mock_data(symbol)
        attempts = 1
        error = None

    if error or df is None or df.empty:
        return {
            "symbol": symbol,
            "score": None,
            "rsi": None,
            "ma50": None,
            "ma200": None,
            "attempts": attempts,
            "status": "error",
            "error": error,
        }

    df = add_indicators(df)
    score = score_stock(df, config=config)

    return {
        "symbol": symbol,
        "score": score,
        "rsi": float(df["RSI"].iloc[-1]) if "RSI" in df.columns else None,
        "ma50": float(df["MA50"].iloc[-1]) if "MA50" in df.columns else None,
        "ma200": float(df["MA200"].iloc[-1]) if "MA200" in df.columns else None,
        "attempts": attempts,
        "status": "ok",
        "error": None,
    }


def main() -> None:
    st.set_page_config(page_title="Quant Stock Ranking Dashboard", layout="wide")

    st.title("📈 Quant Stock Ranking Dashboard")
    st.markdown(
        """
        日米株を対象にクオンツスコアを算出し、銘柄をランキング表示します。

        1年分の終値データを取得し、RSI と移動平均をもとにスコアリングします。
        """
    )

    selected_symbols = _get_symbol_options()

    data_source = st.sidebar.radio("データソース", ["yfinance", "モック（オフライン）"])

    cache_symbol = st.sidebar.text_input("キャッシュをクリアする銘柄（空で全クリア）", value="")
    if st.sidebar.button("キャッシュをクリア"):
        clear_cache(cache_symbol.strip().upper() or None)
        st.sidebar.success("キャッシュをクリアしました。次回更新で最新取得します。")

    if not selected_symbols:
        st.sidebar.warning("少なくとも 1 つの銘柄を指定してください。")

    with st.sidebar.expander("スコア設定", expanded=False):
        rsi_threshold = st.number_input("RSI 閾値 (以下で加点)", value=40.0, min_value=0.0, max_value=100.0, step=1.0)
        rsi_score = st.number_input("RSI 加点", value=30.0, min_value=0.0, step=1.0)
        ma_score = st.number_input("MA50 > MA200 加点", value=40.0, min_value=0.0, step=1.0)

    score_config = ScoreConfig(
        rsi_threshold=rsi_threshold,
        rsi_score=rsi_score,
        ma_cross_score=ma_score,
    )

    force_refresh = st.sidebar.checkbox("強制再取得 (キャッシュを無視)", value=False)

    if st.sidebar.button("ランクを更新"):
        if not selected_symbols:
            st.sidebar.error("表示する銘柄がありません。銘柄セットを確認してください。")
            return

        st.sidebar.info("データを取得・計算中です... 少々お待ちください")

        rows: list[dict] = []

        progress = st.sidebar.progress(0)
        with st.spinner("銘柄データを並列取得中..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(_score_symbol, symbol, data_source, score_config, force_refresh): symbol
                    for symbol in selected_symbols
                }

                for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    rows.append(future.result())
                    progress.progress(i / max(len(selected_symbols), 1))

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)

            st.dataframe(result_df, use_container_width=True)

            error_df = result_df[result_df["status"] == "error"]
            if not error_df.empty:
                with st.expander("取得エラーのある銘柄"):
                    st.table(error_df[["symbol", "attempts", "error"]])

            symbol_to_plot = st.selectbox(
                "チャート表示する銘柄を選択",
                options=result_df["symbol"].tolist(),
                index=0,
            )
            _render_symbol_chart(symbol_to_plot, data_source)
        else:
            st.info("データが見つかりませんでした。")


if __name__ == "__main__":
    main()
