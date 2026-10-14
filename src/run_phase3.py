"""Phase3: 教師なし手法（Isolation Forest・LOF）× 特徴量セット(A/B/C)の評価。Phase1・2と同じ評価条件。

条件を1つずつ変える: ①手法（IF/LOF） ②特徴量セット（A/B/C）。比較の基準にPhase2の残差法（線形gsrのみ・日次）も評価する。

使い方: .venv/bin/python src/run_phase3.py
出力:   outputs/phase3_results.csv / outputs/phase3_summary.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anomaly_eval import alarm_by_normal_quantile, auprc, evaluate_events, false_alarms_per_month  # noqa: E402
from anomaly_synth import make_split  # noqa: E402
from baselines import DETECTOR_COLUMNS  # noqa: E402
from residual import fit_expected, out_of_fold_residual, relative_residual  # noqa: E402
from unsup import UnsupDetector, out_of_fold_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RATE = 0.03
EVAL_INDEXES = range(1, 11)
CONFIGS = [(f"{'IsolationForest' if k == 'iforest' else 'LOF'}・特徴{f}", k, f)
           for k in ("iforest", "lof") for f in ("A", "B", "C")]
REF = "（参考）残差法・線形gsr・日次"


def run(rate: float = RATE, indexes=EVAL_INDEXES, verbose: bool = True):
    train = make_split("train")[DETECTOR_COLUMNS]
    scorers = {}  # 名前 → (スコア関数, しきい値)
    for name, kind, fset in CONFIGS:
        det = UnsupDetector(kind, fset).fit(train)
        scorers[name] = (det.score, alarm_by_normal_quantile(out_of_fold_score(train, kind, fset), rate))
    model = fit_expected(train, "linear_gsr")
    scorers[REF] = (lambda d: relative_residual(d, model, "linear_gsr"),
                    alarm_by_normal_quantile(out_of_fold_residual(train, "linear_gsr"), rate))

    ev_rows, sum_rows = [], []
    for idx in indexes:
        df = make_split("val", idx)
        x = df[DETECTOR_COLUMNS]
        for name, (fn, thr) in scorers.items():
            score = fn(x).to_numpy()
            alarm = score > thr
            ev = evaluate_events(df, alarm)
            ev["detector"], ev["index"] = name, idx
            ev_rows.append(ev)
            sum_rows.append(dict(detector=name, index=idx, fa_per_month=false_alarms_per_month(df, alarm),
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
    if verbose:
        pd.set_option("display.width", 170)
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
    res.to_csv(ROOT / "outputs" / "phase3_results.csv", index=False)
    agg.to_csv(ROOT / "outputs" / "phase3_summary.csv")
