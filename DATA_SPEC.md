# DATA_SPEC.md — データ仕様（案）

## 合成系列（Phase0 で実装）
1日1行。列は次のとおり。

| 列 | 意味 |
|---|---|
| date | 日付 |
| clear_kwh | 晴天時の理論発電量（季節の正弦波で変化、容量は約4.5kWpを想定） |
| weather_class | 天気分類（晴/曇/雨の3値、季節性のある確率で生成） |
| sunshine_ratio | 日照率（0〜1、天気分類と相関） |
| gsr | 全天日射量の代理量（MJ/m²） |
| gen_kwh | 観測発電量 = clear_kwh × 天気係数 × (1+ノイズ) × 異常係数 |
| is_anomaly | 異常日なら1（正解。検知器には渡さない） |
| anomaly_type | none / outage / degrade / shade / stuck |

## 異常の注入（案）
- **outage**: 1〜3日、発電量を0〜10%に。
- **degrade**: 数十日かけて最大5〜15%まで緩やかに低下（開始日以降すべてを異常とする）。
- **shade**: 一定期間、発電量を20〜40%低下（影を想定）。
- **stuck**: 数日間、前日と同じ値が続く（計測固着）。

## 分割
- 時間順。train（正常のみ・異常を注入しない）／ val（較正用、異常を含む）／ test（最終評価、異常を含む）。
- `split_seed`: train 1M / val 2M / test 3M。

## 実データ（Phase6）
- 自宅の日次発電量（`ml_forecast/data/hatuden_daily.csv`、`hatuden_amd.csv`）。**読み取り専用で参照し、コピー・コミットしない**。
