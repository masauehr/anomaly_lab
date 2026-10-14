"""Phase6: 自宅の実データへの適用（手元のみ）。

🔒 結果（検知した日付・個別の数値・図）は outputs/real/（git管理外）にだけ書く。mdには概況（件数・傾向）だけを書く。

1. 実データの残差のばらつきを、合成データと比べる（月ごとのhold-outで、学習に使っていない月の残差を使う）
2. 実データ並みのノイズにした合成データで、検知限界（影の強さ・劣化の低下率）がどう変わるかを調べる
3. 実データで「期待より大きく少ない日」を候補として書き出す（正解はないので、目視で確認する）

使い方: .venv/bin/python src/run_phase6.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anomaly_eval import evaluate_events  # noqa: E402
from anomaly_synth import make_split  # noqa: E402
from baselines import DETECTOR_COLUMNS  # noqa: E402
from combined import CombinedDetector  # noqa: E402
from real_data import load_real  # noqa: E402
from residual import fit_expected, out_of_fold_residual, relative_residual  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "real"  # git管理外（outputs/* は .gitignore 済み）


def mad(s) -> float:
    """外れ値に強いばらつき（正規分布なら標準偏差と一致するよう補正した、中央値からの絶対偏差の中央値）"""
    s = np.asarray(s, dtype=float)
    return float(1.4826 * np.median(np.abs(s - np.median(s))))


def real_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """実データの残差。月ごとのhold-out（その月を学習に使わない）。異常が混ざっていても効くようHuber回帰を使う"""
    y = df["gen_kwh"].to_numpy()
    x = df[["gsr"]].to_numpy()
    month = df["date"].dt.to_period("M")
    pred = np.zeros(len(df))
    for m in month.unique():
        te = (month == m).to_numpy()
        pred[te] = HuberRegressor(max_iter=500).fit(x[~te], y[~te]).predict(x[te])
    pred = np.clip(pred, 1.0, None)
    out = df.copy()
    out["pred"], out["short"] = pred, pred - y            # short: 期待より少ない量(kWh)
    out["rel"] = out["short"] / pred                      # 相対残差
    out["z_rel"] = (out["rel"] - out["rel"].median()) / mad(out["rel"])
    # 期待発電量の水準ごとに、絶対残差のばらつきを見て標準化する（雨の日は小さく、晴れの日は大きく揺れるため）
    out["level"] = pd.cut(out["pred"], [0, 5, 10, 15, 20, 100])
    sd = out.groupby("level", observed=True)["short"].transform(mad)
    out["z_abs"] = (out["short"] - out.groupby("level", observed=True)["short"].transform("median")) / sd
    return out


def synthetic_noise_calibration() -> pd.DataFrame:
    """合成データの日射計測誤差を増やして、実データ並みの相対残差のばらつきになる値を探す"""
    rows = []
    for noise in (0.04, 0.10, 0.15, 0.20, 0.25):
        tr = make_split("train", gsr_noise=noise)[DETECTOR_COLUMNS]
        r = out_of_fold_residual(tr, "linear_gsr")
        rows.append(dict(gsr_noise=noise, rel_mad=round(mad(r), 3)))
    return pd.DataFrame(rows)


def noisy_shade(noise: float, n_series: int = 40) -> pd.DataFrame:
    """日射の計測誤差が大きいとき、影の強さごとの検知率"""
    train = make_split("train", gsr_noise=noise)
    det = CombinedDetector(0.03, 0.0, True).fit(train)
    rows = []
    for f in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2):
        evs = [evaluate_events(d, det.alarm(d)) for d in
               (make_split("val", i, events={"shade": 3}, shade_factor=(f, f), gsr_noise=noise)
                for i in range(300, 300 + n_series))]
        ev = pd.concat(evs)
        rows.append(dict(低下率=f"{int(round((1 - f) * 100))}%", イベント検知率=round(ev["detected"].mean(), 2),
                         日単位検知率=round(ev["alarm_days"].sum() / ev["days"].sum(), 2)))
    return pd.DataFrame(rows)


def noisy_degrade(noise: float, n_series: int = 40) -> pd.DataFrame:
    """日射の計測誤差が大きいとき、劣化の低下率ごとの警報率（日次の残差＋7日平均の残差）"""
    train = make_split("train", gsr_noise=noise)
    det = CombinedDetector(0.02, 0.01, True).fit(train)
    bins = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40001]
    rows = []
    for i in range(200, 200 + n_series):
        df = make_split("val", i, events={"degrade": 1}, degrade_depth=(0.40, 0.40), gsr_noise=noise)
        comp = det.components(df)
        loss = 1 - df["gen_kwh"] / df["gen_clean_kwh"]
        d = df["anomaly_type"] == "degrade"
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = d & (loss >= lo) & (loss < hi)
            rows.append(dict(lo=lo, bin=f"{int(round(lo * 100))}〜{int(round(min(hi, 0.4) * 100))}%", n=int(m.sum()),
                             daily=int(comp.loc[m, "daily"].sum()), avg7=int(comp.loc[m, "avg7"].sum())))
    t = pd.DataFrame(rows).groupby(["lo", "bin"]).sum().reset_index().sort_values("lo")
    t["日次の警報率"], t["7日平均の警報率"] = (t["daily"] / t["n"]).round(2), (t["avg7"] / t["n"]).round(2)
    return t[["bin", "n", "日次の警報率", "7日平均の警報率"]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 170)
    df = load_real()
    r = real_residuals(df)
    print(f"実データ: {len(df)}日")

    print("\n== 1. 残差のばらつき（相対残差のMAD）==")
    cal = synthetic_noise_calibration()
    print("実データ:", round(mad(r["rel"]), 3), "（合成データの日射計測誤差を変えたときの値は下）")
    print(cal.to_string(index=False))
    by = r.groupby("level", observed=True).agg(日数=("rel", "size"), 相対MAD=("rel", mad), 絶対MAD_kWh=("short", mad)).round(2)
    print("\n期待発電量の水準ごと（実データ）:")
    print(by)

    near = cal.iloc[(cal["rel_mad"] - mad(r["rel"])).abs().argmin()]["gsr_noise"]
    print(f"\n== 2. 実データ並みのノイズ（日射計測誤差{near}）での検知限界（合成データ）==")
    shade, deg = noisy_shade(float(near)), noisy_degrade(float(near))
    print("影の強さと検知率:"); print(shade.to_string(index=False))
    print("劣化の低下率ごとの警報率:"); print(deg.to_string(index=False))

    print("\n== 3. 実データの候補（期待より大きく少ない日）==")
    cand = r[(r["z_abs"] > 3) | (r["z_rel"] > 3)].sort_values("z_abs", ascending=False)
    cols = ["date", "weather_class", "gsr", "gen_kwh", "pred", "short", "rel", "z_rel", "z_abs"]
    num = [c for c in cols if c != "date"]
    cand[cols].round({c: 2 for c in num}).to_csv(OUT / "candidates.csv", index=False)  # ← 手元のみ
    r[cols].round({c: 3 for c in num}).to_csv(OUT / "residuals.csv", index=False)       # ← 手元のみ
    print(f"候補（z_abs>3 または z_rel>3）: {len(cand)}日 / 発電量が0の日: {int((df['gen_kwh'] == 0).sum())}日")
    print("  うち雨(天気分類=2): %d日、晴天側(0): %d日" % ((cand["weather_class"] == 2).sum(), (cand["weather_class"] == 0).sum()))
    print("  候補の月別件数:", cand["date"].dt.month.value_counts().sort_index().to_dict())
    print("  詳細は outputs/real/candidates.csv（git管理外）")
    shade.to_csv(OUT / "noisy_shade.csv", index=False)
    deg.to_csv(OUT / "noisy_degrade.csv", index=False)


if __name__ == "__main__":
    main()
