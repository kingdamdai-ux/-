"""Streamlit app entry point for Quant Stock Ranking."""

from __future__ import annotations

import concurrent.futures
import datetime
import io
import time

import numpy as np
import pandas as pd
import streamlit as st

from analysis.indicators import add_indicators
from analysis.scoring import HORIZON_ORDER, ScoreConfig, get_score_config, score_stock_details
from fetch.fetch_data import DataFetchError, get_data, normalize_symbol


_DATA_CACHE: dict[str, pd.DataFrame] = {}
_CACHE_META: dict[str, dict] = {}
_RANK_MAP = {"Strong Buy": 3, "Buy": 2, "Watch": 1, "Avoid": 0, "No Data": -1}
_HORIZON_PREFIX = {"短期": "short", "中期": "medium", "長期": "long"}
_RECOMMENDATION_LABELS = {
    "Strong Buy": "強い買い",
    "Buy": "買い",
    "Watch": "様子見",
    "Avoid": "回避",
    "No Data": "データなし",
}
_STATUS_LABELS = {"ok": "正常", "error": "エラー"}
_RANK_FILTER_OPTIONS = {
    "強い買い以上": "Strong Buy",
    "買い以上": "Buy",
    "様子見以上": "Watch",
    "回避以上": "Avoid",
}

def _sanitize_symbols(symbols: list[object]) -> list[str]:
    sanitized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        normalized = normalize_symbol(symbol)
        if normalized is None or normalized in seen:
            continue
        sanitized.append(normalized)
        seen.add(normalized)
    return sanitized


def clear_cache(symbol: str | None = None) -> None:
    normalized_symbol = normalize_symbol(symbol) if symbol is not None else None
    if normalized_symbol:
        _DATA_CACHE.pop(normalized_symbol, None)
        _CACHE_META.pop(normalized_symbol, None)
    else:
        _DATA_CACHE.clear()
        _CACHE_META.clear()


def _get_cached_data(
    symbol: str,
    force_refresh: bool = False,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
) -> tuple[pd.DataFrame, int, str | None]:
    normalized_symbol = normalize_symbol(symbol)
    if normalized_symbol is None:
        return pd.DataFrame(), 0, "有効な銘柄コードが指定されていません。"

    if not force_refresh and normalized_symbol in _DATA_CACHE:
        meta = _CACHE_META.get(normalized_symbol, {})
        return _DATA_CACHE[normalized_symbol], meta.get("attempts", 1), None

    try:
        df, attempts = get_data(
            normalized_symbol,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            return_attempts=True,
        )
        _DATA_CACHE[normalized_symbol] = df
        _CACHE_META[normalized_symbol] = {
            "attempts": attempts,
            "fetched_at": datetime.datetime.utcnow(),
        }
        return df, attempts, None
    except DataFetchError as exc:
        clear_cache(normalized_symbol)
        return pd.DataFrame(), getattr(exc, "attempts", 0) or 0, str(exc)


def _generate_mock_data(symbol: str, days: int = 252) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=today, periods=days)
    np.random.seed(abs(hash(symbol)) % (2**32))
    returns = np.random.normal(loc=0.0003, scale=0.018, size=len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"Close": prices}, index=dates)


def _get_symbol_options() -> list[str]:
    import data.symbols as symbols

    universe_name = st.sidebar.radio(
        "スクリーニング対象",
        options=list(symbols.SYMBOL_SETS.keys()) + ["カスタム"],
        index=0,
    )

    if universe_name == "カスタム":
        raw = st.sidebar.text_area("カスタム銘柄 (カンマ区切り)", value="AAPL, MSFT, NVDA")
        return _sanitize_symbols(raw.split(","))

    return _sanitize_symbols(symbols.SYMBOL_SETS[universe_name])


def _safe_last_float(last: pd.Series, key: str) -> float | None:
    value = last.get(key, np.nan)
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return None
        value = value.iloc[0, 0]
    elif isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    elif isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    return float(value) if pd.notna(value) else None


def _render_symbol_chart(symbol: str, source: str) -> None:
    normalized_symbol = normalize_symbol(symbol)
    if normalized_symbol is None:
        st.warning("有効な銘柄コードが指定されていません。")
        return

    if source == "yfinance":
        df, _, error = _get_cached_data(normalized_symbol)
    else:
        df = _generate_mock_data(normalized_symbol)
        error = None

    if error or df is None or df.empty:
        st.warning(f"{normalized_symbol} のデータが取得できませんでした。{error or ''}")
        return

    df = add_indicators(df)

    st.subheader(f"{normalized_symbol} のチャート")
    st.caption("終値 / 20日移動平均 / 50日移動平均 / 200日移動平均")
    st.line_chart(df[["Close", "MA20", "MA50", "MA200"]])

    indicator_cols = [c for c in ["RSI", "Momentum21", "Drawdown252"] if c in df.columns]
    if indicator_cols:
        st.caption("RSI / 21日モメンタム / 52週高値からの下落率")
        st.line_chart(df[indicator_cols])


def _build_horizon_payload(prefix: str, score_result: object) -> dict[str, object]:
    return {
        f"{prefix}_score": score_result.total,
        f"{prefix}_recommendation": score_result.recommendation,
        f"{prefix}_trend_score": score_result.trend,
        f"{prefix}_momentum_score": score_result.momentum,
        f"{prefix}_pullback_score": score_result.pullback,
        f"{prefix}_risk_score": score_result.risk,
        f"{prefix}_relative_strength_score": score_result.relative_strength,
    }


def _score_symbol(symbol: str, source: str, configs: dict[str, ScoreConfig], force_refresh: bool) -> dict:
    normalized_symbol = normalize_symbol(symbol)
    base_result = {
        "symbol": normalized_symbol or symbol,
        "rsi": None,
        "momentum21": None,
        "drawdown252": None,
        "volatility21": None,
        "ma20": None,
        "ma50": None,
        "ma200": None,
        "attempts": 0,
        "status": "error",
        "error": "有効な銘柄コードが指定されていません。" if normalized_symbol is None else None,
    }
    for horizon in HORIZON_ORDER:
        prefix = _horizon_prefix(horizon)
        base_result.update(
            {
                f"{prefix}_score": None,
                f"{prefix}_recommendation": "No Data",
                f"{prefix}_trend_score": None,
                f"{prefix}_momentum_score": None,
                f"{prefix}_pullback_score": None,
                f"{prefix}_risk_score": None,
                f"{prefix}_relative_strength_score": None,
            }
        )

    if normalized_symbol is None:
        return base_result

    if source == "yfinance":
        df, attempts, error = _get_cached_data(normalized_symbol, force_refresh=force_refresh)
    else:
        df = _generate_mock_data(normalized_symbol)
        attempts = 1
        error = None

    if error or df is None or df.empty:
        base_result.update({"attempts": attempts, "error": error})
        return base_result

    df = add_indicators(df)
    last = df.iloc[-1]
    result = {
        **base_result,
        "symbol": normalized_symbol,
        "rsi": _safe_last_float(last, "RSI"),
        "momentum21": _safe_last_float(last, "Momentum21"),
        "drawdown252": _safe_last_float(last, "Drawdown252"),
        "volatility21": _safe_last_float(last, "Volatility21"),
        "ma20": _safe_last_float(last, "MA20"),
        "ma50": _safe_last_float(last, "MA50"),
        "ma200": _safe_last_float(last, "MA200"),
        "attempts": attempts,
        "status": "ok",
        "error": None,
    }

    for horizon in HORIZON_ORDER:
        prefix = _horizon_prefix(horizon)
        score_result = score_stock_details(df, config=configs[horizon])
        result.update(_build_horizon_payload(prefix, score_result))

    return result


def _style_recommendations(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _fmt(value: float | None) -> str:
        return f"{float(value):.1f}" if pd.notna(value) else "-"

    fmt_map = {}
    for column in df.columns:
        if column not in {"Symbol", "Recommendation", "Status", "銘柄", "推奨", "状態"}:
            fmt_map[column] = _fmt
    styler = df.style.format(fmt_map)

    if "推奨" in df.columns:
        styler = styler.format({"推奨": lambda value: _RECOMMENDATION_LABELS.get(str(value), str(value))})
    elif "Recommendation" in df.columns:
        styler = styler.format({"Recommendation": lambda value: _RECOMMENDATION_LABELS.get(str(value), str(value))})

    if "状態" in df.columns:
        styler = styler.format({"状態": lambda value: _STATUS_LABELS.get(str(value), str(value))})
    elif "Status" in df.columns:
        styler = styler.format({"Status": lambda value: _STATUS_LABELS.get(str(value), str(value))})

    def _highlight_reco(col: pd.Series) -> list[str]:
        styles = []
        for value in col:
            if value == "Strong Buy":
                styles.append("background-color: #d9f7be")
            elif value == "Buy":
                styles.append("background-color: #fff3bf")
            elif value == "Watch":
                styles.append("background-color: #e7f5ff")
            else:
                styles.append("")
        return styles

    if "推奨" in df.columns:
        styler = styler.apply(_highlight_reco, subset=["推奨"])
    elif "Recommendation" in df.columns:
        styler = styler.apply(_highlight_reco, subset=["Recommendation"])
    return styler


def _horizon_prefix(horizon: str) -> str:
    return _HORIZON_PREFIX[horizon]


def _build_horizon_table(result_df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    prefix = _horizon_prefix(horizon)
    return result_df[
        [
            "symbol",
            f"{prefix}_recommendation",
            f"{prefix}_score",
            f"{prefix}_trend_score",
            f"{prefix}_momentum_score",
            f"{prefix}_pullback_score",
            f"{prefix}_risk_score",
            f"{prefix}_relative_strength_score",
            "rsi",
            "momentum21",
            "drawdown252",
            "volatility21",
            "vs_benchmark",
            "status",
        ]
    ].rename(
        columns={
            "symbol": "銘柄",
            f"{prefix}_recommendation": "推奨",
            f"{prefix}_score": "総合スコア",
            f"{prefix}_trend_score": "トレンド",
            f"{prefix}_momentum_score": "モメンタム",
            f"{prefix}_pullback_score": "押し目",
            f"{prefix}_risk_score": "リスク",
            f"{prefix}_relative_strength_score": "相対強さ",
            "rsi": "RSI",
            "momentum21": "21日モメンタム",
            "drawdown252": "52週高値比",
            "volatility21": "21日ボラティリティ",
            "vs_benchmark": "対ベンチマーク",
            "status": "状態",
        }
    )


def main() -> None:
    st.set_page_config(page_title="クオンツ株ランキングダッシュボード", layout="wide")

    if "score_history" not in st.session_state:
        st.session_state.score_history = {horizon: [] for horizon in HORIZON_ORDER}
    if "selected_symbol_for_chart" not in st.session_state:
        st.session_state.selected_symbol_for_chart = None
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 5

    st.title("クオンツ株ランキングダッシュボード")
    st.markdown(
        """
        広い市場ユニバースを対象に、短期・中期・長期それぞれの視点で
        トレンド、モメンタム、押し目、リスクを評価して買い候補を抽出します。
        """
    )

    selected_symbols = _get_symbol_options()
    data_source = st.sidebar.radio("データソース", ["yfinance", "モックデータ"], index=0)

    cache_symbol = st.sidebar.text_input("キャッシュ削除する銘柄 (空欄で全件)", value="")
    if st.sidebar.button("キャッシュを削除"):
        clear_cache(cache_symbol or None)
        st.sidebar.success("キャッシュを削除しました。")

    with st.sidebar.expander("期間別表示設定", expanded=False):
        default_horizon = st.selectbox("初期表示期間", options=list(HORIZON_ORDER), index=1)
        recommendation_filter_label = st.selectbox("最低推奨ランク", options=list(_RANK_FILTER_OPTIONS.keys()), index=1)
        top_n = st.slider("表示件数", min_value=5, max_value=min(max(len(selected_symbols), 5), 60), value=min(max(len(selected_symbols), 5), 15))

    with st.sidebar.expander("ベンチマーク", expanded=False):
        enable_benchmark = st.checkbox("ベンチマーク比較を有効化", value=True)
        benchmark_symbol = st.selectbox("ベンチマーク銘柄", options=["SPY", "QQQ", "1570.T", "1321.T"], index=0)

    force_refresh = st.sidebar.checkbox("強制再取得 (キャッシュ無視)", value=False)
    auto_refresh = st.sidebar.checkbox("自動更新を有効化", value=st.session_state.auto_refresh)
    if auto_refresh:
        refresh_interval = st.sidebar.selectbox("更新間隔 (分)", options=[1, 3, 5, 10, 15, 30], index=2, format_func=lambda x: f"{x} 分")
    else:
        refresh_interval = st.session_state.refresh_interval
    st.session_state.auto_refresh = auto_refresh
    st.session_state.refresh_interval = refresh_interval

    configs = {horizon: get_score_config(horizon) for horizon in HORIZON_ORDER}

    if not selected_symbols:
        st.warning("少なくとも 1 つの有効な銘柄を指定してください。")
        return

    if st.sidebar.button("ランキングを更新"):
        rows: list[dict] = []
        benchmark_scores: dict[str, float | None] = {horizon: None for horizon in HORIZON_ORDER}

        if enable_benchmark:
            benchmark_row = _score_symbol(benchmark_symbol, data_source, configs, force_refresh)
            for horizon in HORIZON_ORDER:
                benchmark_scores[horizon] = benchmark_row.get(f"{_horizon_prefix(horizon)}_score")

        progress = st.sidebar.progress(0)
        with st.spinner("市場全体をスクリーニングしています..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(_score_symbol, symbol, data_source, configs, force_refresh): symbol
                    for symbol in selected_symbols
                }
                for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    result = future.result()
                    if isinstance(result, dict):
                        rows.append(result)
                    progress.progress(i / max(len(selected_symbols), 1))

        result_df = pd.DataFrame(rows)
        if result_df.empty or "short_score" not in result_df.columns:
            st.info("スクリーニング結果が取得できませんでした。")
            return

        for horizon in HORIZON_ORDER:
            prefix = _horizon_prefix(horizon)
            benchmark_score = benchmark_scores[horizon]
            if enable_benchmark and benchmark_score is not None:
                result_df[f"{prefix}_vs_benchmark"] = result_df[f"{prefix}_score"].apply(
                    lambda s: round(float(s) - float(benchmark_score), 1) if pd.notna(s) else None
                )
            else:
                result_df[f"{prefix}_vs_benchmark"] = None

        initial_prefix = _horizon_prefix(default_horizon)
        result_df = result_df.sort_values([f"{initial_prefix}_score", "symbol"], ascending=[False, True], na_position="last").reset_index(drop=True)

        tabs = st.tabs(list(HORIZON_ORDER))
        for tab, horizon in zip(tabs, HORIZON_ORDER):
            prefix = _horizon_prefix(horizon)
            with tab:
                horizon_df = result_df.sort_values([f"{prefix}_score", "symbol"], ascending=[False, True], na_position="last").reset_index(drop=True)
                threshold = _RANK_MAP[_RANK_FILTER_OPTIONS[recommendation_filter_label]]
                candidate_df = horizon_df[horizon_df[f"{prefix}_recommendation"].map(_RANK_MAP).fillna(-1) >= threshold].copy().head(top_n)

                snapshot = {"timestamp": datetime.datetime.now().strftime("%H:%M:%S")}
                for _, row in horizon_df.iterrows():
                    if row["status"] == "ok" and pd.notna(row[f"{prefix}_score"]):
                        snapshot[row["symbol"]] = row[f"{prefix}_score"]
                st.session_state.score_history[horizon].append(snapshot)
                st.session_state.score_history[horizon] = st.session_state.score_history[horizon][-20:]

                strong_buy_count = int((horizon_df[f"{prefix}_recommendation"] == "Strong Buy").sum())
                buy_count = int((horizon_df[f"{prefix}_recommendation"] == "Buy").sum())
                ok_count = int((horizon_df["status"] == "ok").sum())
                col1, col2, col3 = st.columns(3)
                col1.metric(f"{horizon} 解析銘柄数", ok_count)
                col2.metric("強い買い", strong_buy_count)
                col3.metric("買い", buy_count)

                st.subheader(f"{horizon} 買い候補")
                if candidate_df.empty:
                    st.info("条件に合う買い候補はありませんでした。")
                else:
                    candidate_view = candidate_df[
                        [
                            "symbol",
                            f"{prefix}_recommendation",
                            f"{prefix}_score",
                            f"{prefix}_trend_score",
                            f"{prefix}_momentum_score",
                            f"{prefix}_pullback_score",
                            f"{prefix}_risk_score",
                            "rsi",
                            "momentum21",
                            "drawdown252",
                            "volatility21",
                            f"{prefix}_vs_benchmark",
                        ]
                    ].rename(
                        columns={
                            "symbol": "銘柄",
                            f"{prefix}_recommendation": "推奨",
                            f"{prefix}_score": "総合スコア",
                            f"{prefix}_trend_score": "トレンド",
                            f"{prefix}_momentum_score": "モメンタム",
                            f"{prefix}_pullback_score": "押し目",
                            f"{prefix}_risk_score": "リスク",
                            "rsi": "RSI",
                            "momentum21": "21日モメンタム",
                            "drawdown252": "52週高値比",
                            "volatility21": "21日ボラティリティ",
                            f"{prefix}_vs_benchmark": "対ベンチマーク",
                        }
                    )
                    st.dataframe(_style_recommendations(candidate_view), use_container_width=True)

                with st.expander(f"{horizon} 全銘柄ランキング", expanded=(horizon == default_horizon)):
                    ranking_view = _build_horizon_table(horizon_df.assign(vs_benchmark=horizon_df[f"{prefix}_vs_benchmark"]), horizon)
                    st.dataframe(_style_recommendations(ranking_view), use_container_width=True)

                export_col1, export_col2 = st.columns([1, 5])
                with export_col1:
                    csv = horizon_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label=f"{horizon} CSV ダウンロード",
                        data=csv,
                        file_name=f"screening_{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key=f"csv_{prefix}",
                    )
                with export_col2:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        horizon_df.to_excel(writer, index=False, sheet_name="スクリーニング")
                        candidate_df.to_excel(writer, index=False, sheet_name="買い候補")
                    st.download_button(
                        label=f"{horizon} Excel ダウンロード",
                        data=buffer.getvalue(),
                        file_name=f"screening_{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"xlsx_{prefix}",
                    )

                error_df = horizon_df[horizon_df["status"] == "error"]
                if not error_df.empty:
                    with st.expander("取得エラーのあった銘柄", expanded=False):
                        st.table(
                            error_df[["symbol", "attempts", "error"]].rename(
                                columns={"symbol": "銘柄", "attempts": "試行回数", "error": "エラー内容"}
                            )
                        )

                symbols = candidate_df["symbol"].dropna().tolist() or horizon_df["symbol"].dropna().tolist()
                if symbols:
                    default_symbol = st.session_state.get("selected_symbol_for_chart") or symbols[0]
                    if default_symbol not in symbols:
                        default_symbol = symbols[0]
                    symbol_to_plot = st.selectbox(
                        f"{horizon} でチャート表示する銘柄",
                        options=symbols,
                        index=symbols.index(default_symbol),
                        key=f"selected_symbol_for_chart_{prefix}",
                    )
                    st.session_state.selected_symbol_for_chart = symbol_to_plot
                    _render_symbol_chart(symbol_to_plot, data_source)

                if len(st.session_state.score_history[horizon]) >= 2:
                    with st.expander(f"{horizon} スコア履歴", expanded=False):
                        history_df = pd.DataFrame(st.session_state.score_history[horizon]).set_index("timestamp")
                        st.line_chart(history_df)

    if st.session_state.auto_refresh:
        st.sidebar.info(f"次回更新まで {st.session_state.refresh_interval} 分")
        time.sleep(st.session_state.refresh_interval * 60)
        st.rerun()


if __name__ == "__main__":
    main()

