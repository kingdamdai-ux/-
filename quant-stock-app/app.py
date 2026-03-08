"""Streamlit app entry point for Quant Stock Ranking."""

from __future__ import annotations

import concurrent.futures
import datetime
import time

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
            "rsi_score": None,
            "ma_score": None,
            "rsi": None,
            "ma50": None,
            "ma200": None,
            "attempts": attempts,
            "status": "error",
            "error": error,
        }

    df = add_indicators(df)
    score = score_stock(df, config=config)

    # --- スコア内訳を計算 ---
    rsi_val = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else None
    ma50_val = float(df["MA50"].iloc[-1]) if "MA50" in df.columns else None
    ma200_val = float(df["MA200"].iloc[-1]) if "MA200" in df.columns else None

    rsi_score_val = (
        config.rsi_score if (rsi_val is not None and rsi_val <= config.rsi_threshold) else 0.0
    )
    ma_score_val = (
        config.ma_cross_score
        if (ma50_val is not None and ma200_val is not None and ma50_val > ma200_val)
        else 0.0
    )

    return {
        "symbol": symbol,
        "score": score,
        "rsi_score": rsi_score_val,
        "ma_score": ma_score_val,
        "rsi": rsi_val,
        "ma50": ma50_val,
        "ma200": ma200_val,
        "attempts": attempts,
        "status": "ok",
        "error": None,
    }


def main() -> None:
    st.set_page_config(page_title="Quant Stock Ranking Dashboard", layout="wide")

    # --- セッション状態初期化 ---
    if "score_history" not in st.session_state:
        st.session_state.score_history = []  # list[dict]  各要素: {"timestamp": ..., "SYMBOL": score, ...}
    if "alert_history" not in st.session_state:
        st.session_state.alert_history = []  # list[dict]  各要素: {"timestamp": ..., "alerts": [...]}
    if "alerts_hidden" not in st.session_state:
        st.session_state.alerts_hidden = False
    if "selected_symbol_for_chart" not in st.session_state:
        st.session_state.selected_symbol_for_chart = None
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    if "refresh_interval" not in st.session_state:
        st.session_state.refresh_interval = 5

    st.title("📈 Quant Stock Ranking Dashboard")
    if st.session_state.auto_refresh:
        st.caption(f"🔄 自動更新モード: {st.session_state.refresh_interval} 分ごとに更新")
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

    if st.sidebar.button("🗑️ 履歴をクリア"):
        st.session_state.score_history = []
        st.sidebar.success("スコア履歴をクリアしました。")

    if st.sidebar.button("🗑️ アラート履歴をクリア"):
        st.session_state.alert_history = []
        st.sidebar.success("アラート履歴をクリアしました。")

    if not selected_symbols:
        st.sidebar.warning("少なくとも 1 つの銘柄を指定してください。")

    with st.sidebar.expander("スコア設定", expanded=False):
        rsi_threshold = st.number_input("RSI 閾値 (以下で加点)", value=40.0, min_value=0.0, max_value=100.0, step=1.0)
        rsi_score = st.number_input("RSI 加点", value=30.0, min_value=0.0, step=1.0)
        ma_score = st.number_input("MA50 > MA200 加点", value=40.0, min_value=0.0, step=1.0)

    with st.sidebar.expander("📊 ベンチマーク設定", expanded=False):
        enable_benchmark = st.checkbox("ベンチマーク比較を有効化", value=False)
        benchmark_symbol = st.selectbox(
            "ベンチマーク銘柄",
            options=["SPY", "QQQ", "1570.T", "1321.T"],
            index=0,
        )

    score_config = ScoreConfig(
        rsi_threshold=rsi_threshold,
        rsi_score=rsi_score,
        ma_cross_score=ma_score,
    )

    with st.sidebar.expander("🔔 アラート設定", expanded=False):
        alert_oversold_rsi = st.number_input(
            "売られすぎ RSI 閾値 (以下で警告)", value=30.0, min_value=0.0, max_value=100.0, step=1.0
        )
        alert_overbought_rsi = st.number_input(
            "買われすぎ RSI 閾値 (以上で警告)", value=70.0, min_value=0.0, max_value=100.0, step=1.0
        )
        alert_enable_ma = st.checkbox("デッドクロス警告を有効化 (MA50 < MA200)", value=True)

    force_refresh = st.sidebar.checkbox("強制再取得 (キャッシュを無視)", value=False)

    # --- 自動更新設定 ---
    auto_refresh = st.sidebar.checkbox("🔄 自動更新を有効化", value=st.session_state.auto_refresh)
    if auto_refresh:
        refresh_interval = st.sidebar.selectbox(
            "更新間隔",
            options=[1, 3, 5, 10, 15, 30],
            index=2,
            format_func=lambda x: f"{x} 分",
        )
    else:
        refresh_interval = st.session_state.refresh_interval

    st.session_state.auto_refresh = auto_refresh
    st.session_state.refresh_interval = refresh_interval

    def _dismiss_alerts() -> None:
        st.session_state.alerts_hidden = True

    def _select_symbol(symbol: str) -> None:
        st.session_state.selected_symbol_for_chart = symbol

    if st.sidebar.button("ランクを更新"):
        # 更新時はアラート閉じをリセット
        st.session_state.alerts_hidden = False

        if not selected_symbols:
            st.sidebar.error("表示する銘柄がありません。銘柄セットを確認してください。")
            return

        st.sidebar.info("データを取得・計算中です... 少々お待ちください")

        rows: list[dict] = []

        # --- ベンチマークスコアを取得 ---
        benchmark_score: float | None = None
        if enable_benchmark:
            bench_result = _score_symbol(benchmark_symbol, data_source, score_config, force_refresh)
            benchmark_score = bench_result.get("score")
            if benchmark_score is not None:
                st.sidebar.info(f"ベンチマーク ({benchmark_symbol}) スコア: {benchmark_score:.1f}")
            else:
                st.sidebar.warning(f"ベンチマーク ({benchmark_symbol}) のデータ取得に失敗しました。")

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

            # --- 超過スコア列を追加 ---
            if enable_benchmark and benchmark_score is not None:
                result_df["vs_benchmark"] = result_df["score"].apply(
                    lambda s: round(s - benchmark_score, 1) if s is not None else None
                )
            else:
                result_df["vs_benchmark"] = None

            # --- 履歴に追記 ---
            snapshot = {"timestamp": datetime.datetime.now().strftime("%H:%M:%S")}
            for _, row in result_df.iterrows():
                if row["status"] == "ok" and row["score"] is not None:
                    snapshot[row["symbol"]] = row["score"]
            st.session_state.score_history.append(snapshot)
            # 履歴は最新 20 件だけ保持
            if len(st.session_state.score_history) > 20:
                st.session_state.score_history = st.session_state.score_history[-20:]

            # --- 表示列の定義と列名の日本語化 ---
            display_cols = {
                "symbol": "銘柄",
                "score": "合計スコア",
                "rsi_score": "RSI点",
                "ma_score": "MA点",
                "vs_benchmark": "ベンチマーク超過",
                "rsi": "RSI",
                "ma50": "MA50",
                "ma200": "MA200",
                "status": "状態",
            }
            available_cols = [c for c in display_cols.keys() if c in result_df.columns]
            display_df = result_df[available_cols].rename(columns=display_cols)

            def _style_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
                styler = df.style.format(
                    {
                        "合計スコア": lambda x: f"{x:.1f}" if x is not None else "-",
                        "RSI点": lambda x: f"{x:.1f}" if x is not None else "-",
                        "MA点": lambda x: f"{x:.1f}" if x is not None else "-",
                        "RSI": lambda x: f"{x:.1f}" if x is not None else "-",
                        "MA50": lambda x: f"{x:.1f}" if x is not None else "-",
                        "MA200": lambda x: f"{x:.1f}" if x is not None else "-",
                    "ベンチマーク超過": lambda x: f"{x:.1f}" if x is not None else "-",
                    }
                )

                if "合計スコア" in df.columns:
                    styler = styler.background_gradient(subset=["合計スコア"], cmap="RdYlGn")

                if "ベンチマーク超過" in df.columns:
                    vals = df["ベンチマーク超過"].dropna()
                    if not vals.empty:
                        max_abs = max(abs(vals.min()), abs(vals.max()))
                        styler = styler.background_gradient(
                            subset=["ベンチマーク超過"],
                            cmap="RdYlGn",
                            vmin=-max_abs,
                            vmax=max_abs,
                        )

                if "状態" in df.columns:
                    def _highlight_status(row: pd.Series) -> list[str]:
                        if row.get("状態") == "error":
                            return ["background-color: #ffe5e5"] * len(row)
                        return [""] * len(row)

                    styler = styler.apply(_highlight_status, axis=1)

                return styler

            st.dataframe(_style_dataframe(display_df), use_container_width=True)

            # --- アラート判定・表示 ---
            ok_df = result_df[result_df["status"] == "ok"].copy()

            # 必要なときだけアラートを評価（取得失敗銘柄を除外）
            alert_history_entry: dict = {"timestamp": datetime.datetime.now().strftime("%H:%M:%S"), "alerts": []}
            alerts: list[tuple[str, str, str]] = []  # (level, message, symbol)

            if not ok_df.empty:
                # 売られすぎ（RSI が閾値以下）
                oversold = ok_df[ok_df["rsi"].notna() & (ok_df["rsi"] <= alert_oversold_rsi)]
                for symbol in oversold["symbol"].tolist():
                    msg = f"🔻 売られすぎ (RSI ≤ {alert_oversold_rsi}): **{symbol}**"
                    alerts.append(("warning", msg, symbol))
                    alert_history_entry["alerts"].append({"level": "warning", "symbol": symbol, "message": msg})

                # 買われすぎ（RSI が閾値以上）
                overbought = ok_df[ok_df["rsi"].notna() & (ok_df["rsi"] >= alert_overbought_rsi)]
                for symbol in overbought["symbol"].tolist():
                    msg = f"🔺 買われすぎ (RSI ≥ {alert_overbought_rsi}): **{symbol}**"
                    alerts.append(("warning", msg, symbol))
                    alert_history_entry["alerts"].append({"level": "warning", "symbol": symbol, "message": msg})

                # デッドクロス（MA50 < MA200）
                if alert_enable_ma:
                    dead_cross = ok_df[
                        ok_df["ma50"].notna() & ok_df["ma200"].notna() & (ok_df["ma50"] < ok_df["ma200"])
                    ]
                    for symbol in dead_cross["symbol"].tolist():
                        msg = f"☠️ デッドクロス (MA50 < MA200): **{symbol}**"
                        alerts.append(("error", msg, symbol))
                        alert_history_entry["alerts"].append({"level": "error", "symbol": symbol, "message": msg})

            # 履歴に追加（アラートが無くても記録）
            st.session_state.alert_history.append(alert_history_entry)
            if len(st.session_state.alert_history) > 20:
                st.session_state.alert_history = st.session_state.alert_history[-20:]

            if alerts and not st.session_state.alerts_hidden:
                with st.expander("### 🔔 アラート", expanded=True):
                    st.button("✕ 閉じる", key="dismiss_alerts", on_click=_dismiss_alerts)
                    for level, message, symbol in alerts:
                        if level == "error":
                            st.error(message)
                        else:
                            st.warning(message)
                        st.button(
                            f"▶ {symbol} をチャート表示",
                            key=f"select_{symbol}",
                            on_click=_select_symbol,
                            args=(symbol,),
                        )
            else:
                st.success("✅ 現在アラート対象の銘柄はありません。")

            # --- アラート履歴表示 ---
            with st.expander("🕒 アラート履歴（最新20件）", expanded=False):
                history_rows = []
                for entry in reversed(st.session_state.alert_history):
                    ts = entry.get("timestamp")
                    for alert in entry.get("alerts", []):
                        history_rows.append(
                            {
                                "timestamp": ts,
                                "level": alert.get("level"),
                                "symbol": alert.get("symbol"),
                                "message": alert.get("message"),
                            }
                        )
                if history_rows:
                    history_df = pd.DataFrame(history_rows)
                    st.table(history_df)
                else:
                    st.info("アラート履歴はまだありません。")

            # ✅ ここを追加 -------------------------------------------
            col1, col2 = st.columns([1, 5])
            with col1:
                csv = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 CSVダウンロード",
                    data=csv,
                    file_name=f"ranking_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
            with col2:
                try:
                    import io

                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        result_df.to_excel(writer, index=False, sheet_name="Ranking")
                    st.download_button(
                        label="📊 Excelダウンロード",
                        data=buffer.getvalue(),
                        file_name=f"ranking_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except ImportError:
                    st.caption("`openpyxl` をインストールするとExcel出力も可能です。")
            # ✅ ここまで -----------------------------------------------

            # --- スコア履歴グラフ ---
            if len(st.session_state.score_history) >= 2:
                with st.expander("📈 スコア履歴トレンド", expanded=True):
                    history_df = pd.DataFrame(st.session_state.score_history).set_index("timestamp")
                    st.caption("ランク更新のたびに履歴が蓄積されます（最大20件）")
                    st.line_chart(history_df)
            elif len(st.session_state.score_history) == 1:
                st.info("📊 履歴グラフは「ランクを更新」を2回以上実行すると表示されます。")

            error_df = result_df[result_df["status"] == "error"]
            if not error_df.empty:
                with st.expander("取得エラーのある銘柄"):
                    st.table(error_df[["symbol", "attempts", "error"]])

            # 選択中の銘柄をセッション状態に保存し、アラートからのジャンプで選択を切り替えられるようにする
            symbols = result_df["symbol"].tolist()
            default_symbol = st.session_state.get("selected_symbol_for_chart") or symbols[0]
            if default_symbol not in symbols:
                default_symbol = symbols[0]

            default_index = symbols.index(default_symbol)
            symbol_to_plot = st.selectbox(
                "チャート表示する銘柄を選択",
                options=symbols,
                index=default_index,
                key="selected_symbol_for_chart",
            )
            _render_symbol_chart(symbol_to_plot, data_source)
        else:
            st.info("データが見つかりませんでした。")

        # --- 自動更新トリガー ---
        if st.session_state.auto_refresh:
            st.sidebar.info(f"次回更新まで {st.session_state.refresh_interval} 分")
            time.sleep(st.session_state.refresh_interval * 60)
            st.rerun()


if __name__ == "__main__":
    main()
