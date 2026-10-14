"""Phase4: オートエンコーダの評価。Phase1〜3と同じ評価条件（val の index=1〜10、しきい値は学習に使っていない年の上位3%）。

条件を1つずつ変える: ①圧縮の細さ（code） ②入力の窓（1日/7日） ③学習データに異常が混ざる（汚染）。
乱数の影響を見るため、各設定を3つのシード（0,1,2）で学習し、10系列×3シードの30通りで集計する。

使い方: .venv/bin/python src/run_phase4.py
出力:   outputs/phase4_results.csv / outputs/phase4_summary.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ae import AEDetector, out_of_fold_score  # noqa: E402
from anomaly_eval import alarm_by_normal_quantile, auprc, evaluate_events, false_alarms_per_month  # noqa: E402
from anomaly_synth import make_split  # noqa: E402
from baselines import DETECTOR_COLUMNS  # noqa: E402
from residual import fit_expected, out_of_fold_residual, relative_residual  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RATE = 0.03
EVAL_INDEXES = range(1, 11)
SEEDS = (0, 1, 2)
# (名前, 窓, code, 学習データ)  学習データ: clean=正常のみ / dirty=異常が約15%混ざる
CONFIGS = [
    ("AE・1日・code2", 1, 2, "clean"),
    ("AE・1日・code4", 1, 4, "clean"),
    ("AE・1日・code16(太い)", 1, 16, "clean"),
    ("AE・7日窓・code4", 7, 4, "clean"),
    ("AE・7日窓・code16(太い)", 7, 16, "clean"),
    ("AE・1日・code4・汚染学習", 1, 4, "dirty"),
]
REF = "（参考）残差法・線形gsr・日次"


def dirty_train() -> pd.DataFrame:
    """異常が混ざった2年分の学習データ（val系列 index=100 と 101。2本目は日付を366日ずらして2025年にする）"""
    a = make_split("val", 100)
    b = make_split("val", 101)
    b["date"] = b["date"] + pd.Timedelta(days=366)
    return pd.concat([a, b], ignore_index=True)[DETECTOR_COLUMNS]


def run(rate: float = RATE, indexes=EVAL_INDEXES, seeds=SEEDS, verbose: bool = True):
    clean = make_split("train")[DETECTOR_COLUMNS]
    dirty = dirty_train()
    val_sets = {idx: make_split("val", idx) for idx in indexes}

    scorers = []  # (名前, シード, スコア関数, しきい値)
    for name, window, code, data in CONFIGS:
        tr = clean if data == "clean" else dirty
        for seed in seeds:
            cfg = dict(window=window, code=code, seed=seed)
            det = AEDetector(**cfg).fit(tr)
            scorers.append((name, seed, det.score, alarm_by_normal_quantile(out_of_fold_score(tr, **cfg), rate)))
    model = fit_expected(clean, "linear_gsr")
    scorers.append((REF, 0, lambda d: relative_residual(d, model, "linear_gsr"),
                    alarm_by_normal_quantile(out_of_fold_residual(clean, "linear_gsr"), rate)))

    ev_rows, sum_rows = [], []
    for name, seed, fn, thr in scorers:
        for idx, df in val_sets.items():
            score = fn(df[DETECTOR_COLUMNS]).to_numpy()
            alarm = score > thr
            ev = evaluate_events(df, alarm)
            ev["detector"], ev["index"], ev["seed"] = name, idx, seed
            ev_rows.append(ev)
            sum_rows.append(dict(detector=name, index=idx, seed=seed,
                                 fa_per_month=false_alarms_per_month(df, alarm),
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
    chance = ev_all[ev_all["detector"] == REF].groupby("type")["days"].apply(
        lambda s: float((1 - (1 - rate) ** s).mean())).round(2)
    order = [c[0] for c in CONFIGS] + [REF]
    agg = summ.groupby("detector").agg(
        fa_mean=("fa_per_month", "mean"), fa_sd=("fa_per_month", "std"), fa_grace7=("fa_grace7", "mean"),
        precision=("precision", "mean"), auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std")).round(3).loc[order]
    # シード間のばらつき: 同じ系列・同じ設定でシードだけ変えたときのauprcの標準偏差（系列平均）
    seed_sd = summ.groupby(["detector", "index"])["auprc"].std().groupby("detector").mean().round(3)
    agg["auprc_seed_sd"] = seed_sd.reindex(agg.index)
    if verbose:
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 20)
        print("== 検知器ごと（10系列×3シードの平均と標準偏差）==")
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
    res.to_csv(ROOT / "outputs" / "phase4_results.csv", index=False)
    agg.to_csv(ROOT / "outputs" / "phase4_summary.csv")
