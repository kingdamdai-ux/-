# Quant Stock App

**日本株 + 米国株のクオンツ分析ダッシュボード（Streamlit）**

このリポジトリは、株価データを取得してクオンツスコアを計算し、
ランキングするための基盤となる構造を提供します。

## 🧱 目次

- `app.py` - Streamlitダッシュボードのエントリーポイント
- `data/symbols.py` - 分析対象銘柄リスト
- `fetch/fetch_data.py` - データ取得のための関数群
- `analysis/indicators.py` - テクニカル指標の計算
- `analysis/scoring.py` - クオンツスコアの算出

## 🚀 起動方法

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧭 今後の拡張

- バックテスト機能の追加
- 多銘柄同時分析
- インタラクティブなチャートと比較ビュー
