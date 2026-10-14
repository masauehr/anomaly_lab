"""Phase5: 最終評価。構成は val で決めた後に固定し、test は1回だけ使う。

構成（test を見る前に決めたもの）:
  - 最終構成: 日次の残差(3%) ＋ 固着ルール
  - 劣化重視: 日次の残差(2%) ＋ 7日平均の残差(1%) ＋ 固着ルール
  - 参照: 素朴ルール（3kWh未満）、季節平均との比（Phase1）
しきい値はすべて train（正常のみ）で決める。test の結果で構成やしきい値を選ばない。

使い方:
  .venv/bin/python src/final_eval.py --split val    # 予行演習（val の index=11〜20。何度でも可）
  .venv/bin/python src/final_eval.py --split test   # 本番（test の index=0〜9。1回だけ。使用済みだと拒否する）
出力: outputs/final/final_{split}.csv / final_{split}_summary.csv（test は TEST_USED.json も作る）
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anomaly_eval import alarm_by_normal_quantile, evaluate_events, false_alarms_per_month  # noqa: E402
from anomaly_synth import make_split  # noqa: E402
from baselines import DETECTOR_COLUMNS, score_low_value, score_seasonal_ratio  # noqa: E402
from combined import CombinedDetector  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "final"
INDEXES = {"val": range(11, 21), "test": range(0, 10)}
RATE = 0.03


def build(train: pd.DataFrame) -> dict:
    """名前 → 警報関数（df → bool配列）"""
    final = CombinedDetector(0.03, 0.0, True).fit(train)
    deg = CombinedDetector(0.02, 0.01, True).fit(train)
    tr = train[DETECTOR_COLUMNS]
    thr_season = alarm_by_normal_quantile(score_seasonal_ratio(tr, tr), RATE)
    return {
        "最終構成（日次+固着）": final.alarm,
        "劣化重視（日次2%+7日1%+固着）": deg.alarm,
        "参照: 素朴ルール（3kWh未満）": lambda d: score_low_value(d[DETECTOR_COLUMNS]).to_numpy() > -3.0,
        "参照: 季節平均との比": lambda d: score_seasonal_ratio(d[DETECTOR_COLUMNS], tr).to_numpy() > thr_season,
    }


def evaluate(split: str):
    train = make_split("train")
    detectors = build(train)
    ev_rows, sum_rows = [], []
    for name, fn in detectors.items():
        for idx in INDEXES[split]:
            df = make_split(split, idx)
            al = np.asarray(fn(df), dtype=bool)
            ev = evaluate_events(df, al)
            ev["detector"], ev["index"] = name, idx
            ev_rows.append(ev)
            sum_rows.append(dict(detector=name, index=idx, fa=false_alarms_per_month(df, al),
                                 fa_grace7=false_alarms_per_month(df, al, grace=7),
                                 precision=(al & (df["is_anomaly"] == 1).to_numpy()).sum() / max(1, al.sum())))
    ev = pd.concat(ev_rows, ignore_index=True)
    res = (ev.groupby(["detector", "type"]).agg(events=("detected", "size"), detected=("detected", "sum"),
                                                alarm_days=("alarm_days", "sum"), days=("days", "sum"),
                                                delay=("delay", "mean")).reset_index())
    res["recall"] = (res["detected"] / res["events"]).round(2)
    res["day_recall"] = (res["alarm_days"] / res["days"]).round(2)
    res["delay"] = res["delay"].round(1)
    s = pd.DataFrame(sum_rows).groupby("detector").agg(
        fa=("fa", "mean"), fa_sd=("fa", "std"), fa_grace7=("fa_grace7", "mean"),
        precision=("precision", "mean")).round(2).loc[list(detectors)]
    return res, s, list(detectors)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    a = ap.parse_args()
    lock = OUT / "TEST_USED.json"
    if a.split == "test" and lock.exists():
        sys.exit(f"testは使用済みです（{lock}）。再評価が必要なら、新しいシードのtestセット（index=1000以降）を作ってください。")
    res, s, order = evaluate(a.split)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"final_{a.split}.csv", index=False)
    s.to_csv(OUT / f"final_{a.split}_summary.csv")
    pd.set_option("display.width", 170)
    print(f"== {a.split}（index={INDEXES[a.split].start}〜{INDEXES[a.split].stop - 1}）==")
    print(s)
    for col in ["recall", "day_recall", "delay"]:
        print(f"\n-- {col} --")
        print(res.pivot(index="detector", columns="type", values=col).loc[order])
    if a.split == "test":
        lock.write_text(json.dumps({"used_at": datetime.now().isoformat(timespec="seconds"),
                                    "indexes": [INDEXES["test"].start, INDEXES["test"].stop - 1],
                                    "note": "testは最終評価で使用済み。構成はvalで決めて固定した"}, ensure_ascii=False, indent=2))
        print("\nTEST_USED.json を作成しました（以後 --split test は拒否されます）")


if __name__ == "__main__":
    main()
