# anomaly_lab

**異常検知**を、太陽光発電量（日次）の時系列を題材に学ぶプロジェクト。
「いつもと違う」をどう定義し、どう見つけ、**どれだけ当たったかをどう検証するか**を、小さなシステムとして組み立てる。

詳しくは [MANUAL.md](MANUAL.md) / [DATA_SPEC.md](DATA_SPEC.md) / [EVALUATION.md](EVALUATION.md) / [PLAN.md](PLAN.md) を参照。
各Notebookの**実行結果（図・表・読み取れること）のまとめ**は [notebooks/README.md](notebooks/README.md) にある（Notebookを開く前の予習・実行後の答え合わせに使える）。pc_docs運用マニュアルのコピーは [anomaly-lab.md](anomaly-lab.md)。

> 🔒 **学習は合成データだけで行う**（公開リポジトリ向け）。自宅の発電量など**実データは手元のみで適用**し、
> 生データ・結果の画像はコミットしない。詳しくは [CLAUDE.md](CLAUDE.md) を参照。

## なぜこのプロジェクトか

[ai_core](../ai_core/PROJECT_MAP.md) で「未経験」だった分野の1つが**異常検知**。
異常検知は、教師あり分類と違って**「異常のラベルがほとんど無い」**のが最大の難しさ。
このため「何を正常とみなすか」「評価をどう作るか」の設計がそのまま学びになる。

## 題材

日次の太陽光発電量（kWh）。晴れれば多く、曇り・雨で少ない**季節と天気に依存する系列**。
「少ない」こと自体は異常ではない（雨の日は少なくて当然）。**天気で説明できない落ち込み**が異常。

| 異常の種類 | 例 | 検知の難しさ |
|---|---|---|
| 突発的な落ち込み | 停電・パワコン停止で発電0 | 易しい |
| 緩やかな劣化 | パネルの汚れ・劣化で数%ずつ低下 | 難しい（日々の変動に埋もれる） |
| 部分的な低下 | 影（新築・樹木）で一定時間帯のみ低下 | 中 |
| 計測欠損・固着 | 同じ値が続く、0が続く | 中（データの異常） |

## 作るもの（パイプライン）

```
合成データ生成（正常＋異常を注入） ─▶ ①非AIのベースライン（しきい値・移動平均）
  ─▶ ②期待値との残差で検知（天気から期待発電量を予測） ─▶ ③教師なし手法（Isolation Forest等）
  ─▶ ④オートエンコーダ ─▶ ⑤評価（検知率・誤警報・検知までの日数） ─▶ ⑥実データへ適用（手元のみ）
```

## 現状

**Phase0（合成データ生成器）・Phase1（非AIのベースライン）・Phase2（天気から期待発電量を予測し、残差で検知）・Phase3（教師なし手法 Isolation Forest/LOF）・Phase4（オートエンコーダ）・Phase5（組み合わせ・検知限界・最終評価）・Phase6（自宅の実データへの適用。手元のみ）完了。** 最終レポートは [REPORT.md](REPORT.md)。計画した全フェーズが終了。進め方は [PLAN.md](PLAN.md)。

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pandas scikit-learn matplotlib ipykernel torch
.venv/bin/python -m unittest discover -s tests   # テスト
.venv/bin/python src/make_data.py                # data/*.csv と outputs/phase0_series.png を作る
.venv/bin/python src/run_phase1.py               # ベースライン4つ+素朴ルールの評価
.venv/bin/python src/run_phase2.py               # 残差法6つ（期待値モデル3種×平滑化2種）の評価
.venv/bin/python src/run_phase3.py               # 教師なし6つ（IF/LOF×特徴量A/B/C）の評価
.venv/bin/python src/run_phase4.py               # オートエンコーダ6つ（窓・code・汚染）×3シードの評価
.venv/bin/python src/run_phase5.py               # 組み合わせ・劣化の検知限界・影の感度分析
.venv/bin/python src/final_eval.py --split val   # 最終評価の予行演習（testは使用済み）
.venv/bin/python src/run_phase6.py               # 実データ（自宅・手元のみ）。ml_forecast/data を読み取り専用で参照。結果は outputs/real/（git管理外）
.venv/bin/python src/run_inject.py               # 実データに合成の異常を注入して検知限界を測る（手元のみ）
```

Notebook は `notebooks/`（カーネル名「Python (anomaly_lab)」）。

## Notebook を読むとき
はじめての方は、まず [notebooks/00_basics.ipynb](notebooks/00_basics.ipynb)（基礎知識・用語集・読む順）から。各Notebookの冒頭には「やさしい導入」があります。

## 更新履歴（セキュリティ・依存更新）

| 日付 | 内容 |
|------|------|
| 2026-10-24 | 実行環境（venv）の依存パッケージを脆弱性点検に基づき更新（pillow 12.x → 13.0.0、pip・setuptools・certifi・filelock・urllib3 など）。単体テスト60件が全て通過。リポジトリのコード変更なし。週次の自動点検は別の非公開プロジェクトで管理している。 |
