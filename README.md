# Quant Stock App

日本株・米国株を対象に、テクニカル指標ベースで銘柄をスクリーニングする Streamlit アプリです。

## 主な機能

- 短期・中期・長期の期間別ランキング表示
- 推奨ランク（強い買い / 買い / 様子見 / 回避）の可視化
- ベンチマーク比較
- CSV / Excel エクスポート
- 候補銘柄のチャート表示

## セットアップ

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 構成

- `app.py`: Streamlit アプリ本体
- `data/symbols.py`: 銘柄ユニバース定義
- `fetch/fetch_data.py`: データ取得処理
- `analysis/indicators.py`: テクニカル指標計算
- `analysis/scoring.py`: スコアリングロジック
