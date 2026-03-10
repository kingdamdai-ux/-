"""Ticker symbol universes used by the Streamlit dashboard."""

from __future__ import annotations

JP_CORE = [
    "7203.T",  # Toyota Motor
    "6758.T",  # Sony Group
    "8306.T",  # Mitsubishi UFJ Financial Group
    "9984.T",  # SoftBank Group
    "6861.T",  # Keyence
    "8035.T",  # Tokyo Electron
    "4063.T",  # Shin-Etsu Chemical
    "6501.T",  # Hitachi
    "6098.T",  # Recruit Holdings
    "4519.T",  # Chugai Pharmaceutical
    "7974.T",  # Nintendo
    "9432.T",  # NTT
]

US_CORE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "AVGO",
    "TSM",
    "AMD",
    "NFLX",
    "COST",
    "JPM",
    "V",
    "LLY",
    "XOM",
]

US_INDEX_ETFS = [
    "SPY",
    "VOO",
    "IVV",
    "QQQ",
    "DIA",
    "IWM",
    "VTI",
    "RSP",
]

JP_INDEX_ETFS = [
    "1306.T",
    "1321.T",
    "1570.T",
    "2558.T",
    "2630.T",
]

SECTOR_ETFS = [
    "XLK",
    "XLF",
    "XLI",
    "XLE",
    "XLV",
    "XLP",
    "XLY",
    "XLC",
    "XLU",
    "SMH",
]

GROWTH_LEADERS = [
    "PLTR",
    "CRWD",
    "PANW",
    "NOW",
    "SHOP",
    "MELI",
    "UBER",
    "SNOW",
    "ADBE",
    "INTU",
]

DIVIDEND_QUALITY = [
    "KO",
    "PEP",
    "PG",
    "JNJ",
    "ABBV",
    "HD",
    "MCD",
    "WMT",
    "MS",
    "BLK",
]

SYMBOL_SETS = {
    "広域スクリーニング": US_INDEX_ETFS + JP_INDEX_ETFS + US_CORE + JP_CORE + SECTOR_ETFS,
    "米国大型株": US_CORE,
    "日本大型株": JP_CORE,
    "指数 ETF": US_INDEX_ETFS + JP_INDEX_ETFS,
    "セクター ETF": SECTOR_ETFS,
    "成長リーダー": GROWTH_LEADERS,
    "高配当・品質": DIVIDEND_QUALITY,
}

ALL_SYMBOLS = SYMBOL_SETS["広域スクリーニング"] + GROWTH_LEADERS + DIVIDEND_QUALITY
