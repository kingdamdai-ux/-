"""Streamlit app entry point for Quant Stock Ranking."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data.symbols import ALL_SYMBOLS
from fetch.fetch_data import get_data
from analysis.indicators import add_indicators
from analysis.scoring import score_stock


def main() -> None:
    st.set_page_config(page_title="Quant Stock Ranking Dashboard", layout="wide")

    st.title("📈 Quant Stock Ranking Dashboard")
    st.markdown(
        """
        日米株を対象にクオンツスコアを算出し、銘柄をランキング表示します。

        1年分の終値データを取得し、RSI と移動平均をもとにスコアリングします。
        """
    )

    if st.sidebar.button("ランクを更新"):
        st.sidebar.info("データを取得・計算中です... 少々お待ちください")

        rows: list[dict] = []

        for symbol in ALL_SYMBOLS:
            df = get_data(symbol)
            df = add_indicators(df)

            score = score_stock(df)

            rows.append(
                {
                    "symbol": symbol,
                    "score": score,
                    "rsi": float(df["RSI"].iloc[-1]) if not df.empty and "RSI" in df.columns else None,
                    "ma50": float(df["MA50"].iloc[-1]) if not df.empty and "MA50" in df.columns else None,
                    "ma200": float(df["MA200"].iloc[-1]) if not df.empty and "MA200" in df.columns else None,
                }
            )

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)
            st.dataframe(result_df, use_container_width=True)
        else:
            st.info("データが見つかりませんでした。")


if __name__ == "__main__":
    main()
