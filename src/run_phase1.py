"""Phase1: AIを使わないベースラインの評価。

しきい値は train（正常のみ）のスコア分位点で決める（正常日の警報率 RATE）。
評価は val の index=1〜10 の10系列（index=0 は使わない。testは使わない）。

使い方: .venv/bin/python src/run_phase1.py
出力:   outputs/phase1_results.csv（検知器×異常種類） / outputs/phase1_summary.csv（検知器ごと）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anomaly_eval import alarm_by_normal_quantile, auprc, evaluate_events, false_alarms_per_month  # noqa: E402
from anomaly_synth import make_split  # noqa: E402
from baselines import (DETECTOR_COLUMNS, score_low_value, score_moving_average_drop,  # noqa: E402
                       score_seasonal_ratio, score_stuck_run)

ROOT = Path(__file__).resolve().parent.parent
RATE = 0.03  # 正常日の警報率の目標（約1か月に1回）
EVAL_INDEXES = range(1, 11)


def build_detectors(train: pd.DataFrame) -> dict:
    """検知器の名前 → スコア関数（検知器が見てよい列だけを受け取る）"""
    return {
        "固定しきい値(低い値)": score_low_value,
        "移動平均からの落ち込み": score_moving_average_drop,
        "季節平均との比": lambda d: score_seasonal_ratio(d, train[DETECTOR_COLUMNS]),
        "固着ルール(同値連続)": score_stuck_run,
    }


def main() -> None:
    train = make_split("train")
    detectors = build_detectors(train)
    thresholds = {name: alarm_by_normal_quantile(fn(train[DETECTOR_COLUMNS]), RATE)
                  for name, fn in detectors.items()}
    # 「3kWh未満なら警報」という素朴な固定ルールも比較に入れる（較正なし）
    naive_thr = -3.0

    ev_rows, sum_rows = [], []
    for idx in EVAL_INDEXES:
        df = make_split("val", idx)
        x = df[DETECTOR_COLUMNS]
        for name, fn in detectors.items():
            score = fn(x).to_numpy()
            alarm = score > thresholds[name]
            ev = evaluate_events(df, alarm)
            ev["detector"], ev["index"] = name, idx
            ev_rows.append(ev)
            sum_rows.append(dict(detector=name, index=idx, fa_per_month=false_alarms_per_month(df, alarm),
                                 auprc=auprc(df, score), detected=ev["detected"].mean()))
        # 素朴な固定ルール
        alarm = score_low_value(x).to_numpy() > naive_thr
        ev = evaluate_events(df, alarm)
        ev["detector"], ev["index"] = "素朴ルール(3kWh未満)", idx
        ev_rows.append(ev)
        normal_rain = ((df["is_anomaly"] == 0) & (df["weather_class"] == 2)).sum()
        fa_rain = (alarm & (df["is_anomaly"] == 0) & (df["weather_class"] == 2)).sum()
        sum_rows.append(dict(detector="素朴ルール(3kWh未満)", index=idx, fa_per_month=false_alarms_per_month(df, alarm),
                             auprc=auprc(df, score_low_value(x)), detected=ev["detected"].mean(),
                             rain_fa_share=fa_rain / max(1, (alarm & (df["is_anomaly"] == 0)).sum()),
                             normal_rain_alarmed=fa_rain / normal_rain))

    ev_all = pd.concat(ev_rows, ignore_index=True)
    summ = pd.DataFrame(sum_rows)
    (ROOT / "outputs").mkdir(exist_ok=True)

    # 検知器×異常種類の検知率（イベント単位、10系列の合計）と、検知までの日数の平均
    res = (ev_all.groupby(["detector", "type"])
           .agg(events=("detected", "size"), detected=("detected", "sum"), delay_mean=("delay", "mean"),
                days=("days", "sum"), alarm_days=("alarm_days", "sum"))
           .reset_index())
    res["day_recall"] = (res["alarm_days"] / res["days"]).round(2)
    res["recall"] = (res["detected"] / res["events"]).round(2)
    res["delay_mean"] = res["delay_mean"].round(1)
    # 偶然の水準: 正常日の警報率RATEのまま、長さnのイベント中に1回以上警報が出る確率 = 1-(1-RATE)^n
    ev_len = ev_all[ev_all["detector"] == "固着ルール(同値連続)"]
    chance = ev_len.groupby("type")["days"].apply(lambda s: float((1 - (1 - RATE) ** s).mean())).round(2)
    res.to_csv(ROOT / "outputs" / "phase1_results.csv", index=False)

    agg = summ.groupby("detector").agg(
        fa_mean=("fa_per_month", "mean"), fa_sd=("fa_per_month", "std"),
        auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std")).round(3)
    agg.to_csv(ROOT / "outputs" / "phase1_summary.csv")

    pd.set_option("display.width", 160)
    print("しきい値:", {k: round(v, 3) for k, v in thresholds.items()})
    print("\n== 検知器ごと（10系列の平均±標準偏差）==")
    print(agg)
    print("\n== 検知率（イベント単位・10系列合計） ==")
    print(res.pivot(index="detector", columns="type", values="recall"))
    print("\n== 日単位の検知率（異常日のうち警報が出た割合・10系列合計） ==")
    print(res.pivot(index="detector", columns="type", values="day_recall"))
    print("\n== 偶然の水準（警報がランダムに出ても検知に数えられる確率） ==")
    print(chance.to_dict())
    print("\n== 検知までの日数（検知できたイベントの平均） ==")
    print(res.pivot(index="detector", columns="type", values="delay_mean"))
    naive = summ[summ["detector"] == "素朴ルール(3kWh未満)"]
    print("\n素朴ルールの誤警報のうち雨の日の割合: %.2f / 雨の正常日に警報が出る割合: %.2f"
          % (naive["rain_fa_share"].mean(), naive["normal_rain_alarmed"].mean()))
    print("異常イベント数（種類別）:", ev_all[ev_all["detector"] == "固着ルール(同値連続)"]["type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
