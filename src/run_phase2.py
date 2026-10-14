"""Phase2: 天気から期待発電量を予測し、残差で検知する。Phase1と同じ評価条件。

- 期待値モデルは train（正常のみ）で学習。しきい値は train の年ごとのhold-out残差の上位 RATE。
- 評価は val の index=1〜10（testは使わない）。
- 条件を1つずつ変える: ①期待値モデル（gsrのみ線形/全特徴線形/RandomForest） ②残差の平滑化（なし/7日）

使い方: .venv/bin/python src/run_phase2.py
出力:   outputs/phase2_results.csv / outputs/phase2_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anomaly_eval import alarm_by_normal_quantile, auprc, evaluate_events, false_alarms_per_month  # noqa: E402
from anomaly_synth import make_split  # noqa: E402
from baselines import DETECTOR_COLUMNS  # noqa: E402
from residual import fit_expected, out_of_fold_residual, relative_residual, smooth  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RATE = 0.03
EVAL_INDEXES = range(1, 11)
CONFIGS = [  # (名前, モデル, 平滑化の窓)
    ("線形(gsrのみ)・日次", "linear_gsr", 1),
    ("線形(全特徴)・日次", "linear_all", 1),
    ("RandomForest・日次", "rf", 1),
    ("線形(gsrのみ)・7日平均", "linear_gsr", 7),
    ("線形(全特徴)・7日平均", "linear_all", 7),
    ("RandomForest・7日平均", "rf", 7),
]


def run(rate: float = RATE, indexes=EVAL_INDEXES, verbose: bool = True):
    train = make_split("train")
    tr = train[DETECTOR_COLUMNS]
    models = {k: fit_expected(tr, k) for k in {c[1] for c in CONFIGS}}
    oof = {k: out_of_fold_residual(tr, k) for k in models}
    thr = {(n, k, w): alarm_by_normal_quantile(smooth(oof[k], w), rate) for n, k, w in CONFIGS}

    ev_rows, sum_rows = [], []
    for idx in indexes:
        df = make_split("val", idx)
        x = df[DETECTOR_COLUMNS]
        for n, k, w in CONFIGS:
            score = smooth(relative_residual(x, models[k], k), w).to_numpy()
            alarm = score > thr[(n, k, w)]
            ev = evaluate_events(df, alarm)
            ev["detector"], ev["index"] = n, idx
            ev_rows.append(ev)
            sum_rows.append(dict(detector=n, index=idx, fa_per_month=false_alarms_per_month(df, alarm),
                                 fa_grace7=false_alarms_per_month(df, alarm, grace=7),
                                 auprc=auprc(df, score),
                                 precision=(alarm & (df["is_anomaly"] == 1).to_numpy()).sum() / max(1, alarm.sum())))
    ev_all = pd.concat(ev_rows, ignore_index=True)
    summ = pd.DataFrame(sum_rows)
    res = (ev_all.groupby(["detector", "type"])
           .agg(events=("detected", "size"), detected=("detected", "sum"), delay_mean=("delay", "mean"),
                days=("days", "sum"), alarm_days=("alarm_days", "sum")).reset_index())
    res["recall"] = (res["detected"] / res["events"]).round(2)
    res["day_recall"] = (res["alarm_days"] / res["days"]).round(2)
    res["delay_mean"] = res["delay_mean"].round(1)
    first = ev_all[ev_all["detector"] == CONFIGS[0][0]]
    chance = first.groupby("type")["days"].apply(lambda s: float((1 - (1 - rate) ** s).mean())).round(2)
    agg = summ.groupby("detector").agg(
        fa_mean=("fa_per_month", "mean"), fa_sd=("fa_per_month", "std"), fa_grace7=("fa_grace7", "mean"),
        precision=("precision", "mean"), auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std")).round(3)
    agg = agg.loc[[c[0] for c in CONFIGS]]

    if verbose:
        pd.set_option("display.width", 170)
        order = [c[0] for c in CONFIGS]
        print("== 検知器ごと（10系列の平均±標準偏差）==")
        print(agg)
        for col, title in [("recall", "検知率（イベント単位）"), ("day_recall", "日単位の検知率"),
                           ("delay_mean", "検知までの日数（検知できたイベントの平均）")]:
            print(f"\n== {title} ==")
            print(res.pivot(index="detector", columns="type", values=col).loc[order])
        print("\n偶然の水準:", chance.to_dict())
    return res, agg, chance


if __name__ == "__main__":
    res, agg, _ = run()
    (ROOT / "outputs").mkdir(exist_ok=True)
    res.to_csv(ROOT / "outputs" / "phase2_results.csv", index=False)
    agg.to_csv(ROOT / "outputs" / "phase2_summary.csv")
