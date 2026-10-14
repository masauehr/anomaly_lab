"""Phase5: 組み合わせの比較・劣化の検知限界・感度分析（すべて val。testは使わない）。

1. compare(): 組み合わせ構成の比較（val の index=1〜10）
2. degrade_limit(): 劣化が「何%低下」から検知できるか（劣化だけの40系列、最終低下率20%、index=200〜239）
3. shade_sensitivity(): 影の強さを変えたときの検知率（index=300〜339）

使い方: .venv/bin/python src/run_phase5.py
出力:   outputs/phase5_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anomaly_eval import evaluate_events, false_alarms_per_month  # noqa: E402
from anomaly_synth import make_split  # noqa: E402
from combined import CombinedDetector  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL_INDEXES = range(1, 11)
# 構成名 → (日次の率, 7日平均の率, 固着ルール)。最終構成はvalで決めた「日次+固着」
CONFIGS = {
    "日次のみ": (0.03, 0.0, False),
    "日次+固着（最終構成）": (0.03, 0.0, True),
    "日次2%+7日1%+固着（劣化重視）": (0.02, 0.01, True),
}
LOSS_BINS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20001]


def compare(verbose: bool = True):
    train = make_split("train")
    rows, ev_rows = [], []
    for name, (rd, r7, st) in CONFIGS.items():
        det = CombinedDetector(rd, r7, st).fit(train)
        for idx in EVAL_INDEXES:
            df = make_split("val", idx)
            al = det.alarm(df)
            ev = evaluate_events(df, al)
            ev["config"], ev["index"] = name, idx
            ev_rows.append(ev)
            rows.append(dict(config=name, index=idx, fa=false_alarms_per_month(df, al),
                             fa_grace7=false_alarms_per_month(df, al, grace=7),
                             precision=(al & (df["is_anomaly"] == 1).to_numpy()).sum() / max(1, al.sum())))
    ev = pd.concat(ev_rows, ignore_index=True)
    summ = pd.DataFrame(rows).groupby("config").agg(fa=("fa", "mean"), fa_grace7=("fa_grace7", "mean"),
                                                   precision=("precision", "mean")).round(2).loc[list(CONFIGS)]
    res = (ev.groupby(["config", "type"]).agg(events=("detected", "size"), detected=("detected", "sum"),
                                              alarm_days=("alarm_days", "sum"), days=("days", "sum"),
                                              delay=("delay", "mean")).reset_index())
    res["recall"] = (res["detected"] / res["events"]).round(2)
    res["day_recall"] = (res["alarm_days"] / res["days"]).round(2)
    res["delay"] = res["delay"].round(1)
    if verbose:
        pd.set_option("display.width", 160)
        print(summ)
        for col in ["recall", "day_recall", "delay"]:
            print(f"\n== {col} ==")
            print(res.pivot(index="config", columns="type", values=col).loc[list(CONFIGS)])
    return summ, res


def degrade_limit(verbose: bool = True):
    """劣化の日ごとの低下率（1−実測/本来の値）ごとに、警報が出た割合を見る"""
    train = make_split("train")
    det = CombinedDetector(0.02, 0.01, True).fit(train)  # 日次2%+7日平均1%（部品別に見る）
    rows = []
    normal_alarm = []
    for idx in range(200, 240):
        df = make_split("val", idx, events={"degrade": 1}, degrade_depth=(0.20, 0.20))
        comp = det.components(df)
        d = df["anomaly_type"] == "degrade"
        loss = 1 - df["gen_kwh"] / df["gen_clean_kwh"]
        normal_alarm.append(comp[~d].mean())
        for lo, hi in zip(LOSS_BINS[:-1], LOSS_BINS[1:]):
            m = d & (loss >= lo) & (loss < hi)
            rows.append(dict(bin=f"{int(lo*100)}〜{int(round(min(hi, 0.2)*100))}%", lo=lo, n=int(m.sum()),
                             daily=comp.loc[m, "daily"].sum(), avg7=comp.loc[m, "avg7"].sum(),
                             any_=comp.loc[m].any(axis=1).sum()))
    t = pd.DataFrame(rows).groupby(["lo", "bin"]).sum().reset_index().sort_values("lo")
    for c in ["daily", "avg7", "any_"]:
        t[c + "_rate"] = (t[c] / t["n"]).round(2)
    base = pd.concat(normal_alarm, axis=1).mean(axis=1).round(3)
    out = t[["bin", "n", "daily_rate", "avg7_rate", "any__rate"]].rename(columns={"any__rate": "any_rate"})
    if verbose:
        print("劣化の低下率ごとの警報率（劣化以外の日の警報率は", base.to_dict(), "）")
        print(out.to_string(index=False))
    return out, base.to_dict()


def shade_sensitivity(verbose: bool = True):
    """影の強さ（発電量が本来の何割になるか）を変えたときの検知"""
    train = make_split("train")
    det = CombinedDetector(0.03, 0.0, True).fit(train)
    rows = []
    for f in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.4):
        evs, dr = [], []
        for idx in range(300, 340):
            df = make_split("val", idx, events={"shade": 3}, shade_factor=(f, f))
            al = det.alarm(df)
            evs.append(evaluate_events(df, al))
        ev = pd.concat(evs)
        rows.append(dict(低下率=f"{int(round((1-f)*100))}%", イベント検知率=round(ev["detected"].mean(), 2),
                         日単位検知率=round(ev["alarm_days"].sum() / ev["days"].sum(), 2)))
    out = pd.DataFrame(rows)
    if verbose:
        print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    print("=== 1. 組み合わせの比較（val 10系列）===")
    s, r = compare()
    s.to_csv(ROOT / "outputs" / "phase5_compare_summary.csv"); r.to_csv(ROOT / "outputs" / "phase5_compare.csv", index=False)
    print("\n=== 2. 劣化の検知限界（劣化だけ40系列）===")
    d, _ = degrade_limit(); d.to_csv(ROOT / "outputs" / "phase5_degrade_limit.csv", index=False)
    print("\n=== 3. 影の強さの感度分析（影だけ40系列）===")
    shade_sensitivity().to_csv(ROOT / "outputs" / "phase5_shade.csv", index=False)
