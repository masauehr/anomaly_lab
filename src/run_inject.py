"""Phase6b: 実データに合成の異常を注入して、実データのノイズ構造のまま検知限界を測る（手元のみ）。

🔒 実データ（自宅の発電量）は ml_forecast/data/ を読み取り専用で参照する。結果は outputs/real/（git管理外）にだけ書く。
mdには概況（検知限界の数字）だけを書く。日付や個別の値は書かない。

考え方:
  - 実データ514日は、所有者の確認（候補15日はすべて正常）から、異常のない系列とみなす。
  - その発電量の一部を書き換えて異常（影・停電・劣化・固着）を作り、検知できるか測る。正解は自分で注入したので分かる。
  - 期待値モデル（月ごとのhold-outのHuber回帰）としきい値は、注入前のきれいなデータで決める。
    注入した値が学習に混ざる影響（汚染）は無視している（近似）。
  - 2つのスコアを比べる: 相対残差(rel) と、期待発電量の水準ごとに標準化した残差(z)。
  - 警報: 日次のスコアが、きれいなデータの上位3%を超えた日 ＋ 直近7日平均のスコアが上位1%を超えた日。

使い方: .venv/bin/python src/run_inject.py
出力:   outputs/real/inject_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_data import load_real  # noqa: E402
from run_phase6 import mad, real_residuals  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "real"
N_TRIALS = 300
LEVELS = [0, 5, 10, 15, 20, 100]
LEVEL_NAMES = ["〜5", "5〜10", "10〜15", "15〜20", "20〜"]
MIN_TRIALS = 10  # 水準ごとの試行数がこれ未満の欄は信頼できないので空欄にする


class Scorer:
    """注入前のきれいな実データから、期待値・水準別の統計・しきい値を決める"""

    def __init__(self, df: pd.DataFrame):
        r = real_residuals(df)
        self.pred = r["pred"].to_numpy()
        self.y = df["gen_kwh"].to_numpy()
        self.level = np.digitize(self.pred, LEVELS[1:-1])          # 水準の番号 0..4
        short = self.pred - self.y
        self.med = np.array([np.median(short[self.level == k]) for k in range(5)])
        self.sd = np.array([mad(short[self.level == k]) for k in range(5)])
        self.thr = {}
        for kind in ("rel", "z"):
            s = self.score(self.y, kind)
            self.thr[kind] = (np.quantile(s, 0.97), np.quantile(self.avg7(s), 0.99))

    def score(self, y: np.ndarray, kind: str) -> np.ndarray:
        short = self.pred - y
        if kind == "rel":
            return short / self.pred
        return (short - self.med[self.level]) / self.sd[self.level]

    @staticmethod
    def avg7(s: np.ndarray) -> np.ndarray:
        return pd.Series(s).rolling(7, min_periods=1).mean().to_numpy()

    def alarm(self, y: np.ndarray, kind: str) -> np.ndarray:
        s = self.score(y, kind)
        td, t7 = self.thr[kind]
        return (s > td) | (self.avg7(s) > t7)

    def daily_alarm(self, y: np.ndarray, kind: str) -> np.ndarray:
        return self.score(y, kind) > self.thr[kind][0]


def trial_positions(n: int, length: int, rng, count: int) -> np.ndarray:
    return rng.integers(7, n - length - 1, size=count)


def run() -> dict:
    df = load_real()
    sc = Scorer(df)
    n, y = len(df), sc.y
    rng = np.random.default_rng(0)
    out = {}

    # 注入前のきれいなデータでの警報率（誤警報の目安）
    out["clean_alarm_rate"] = {k: float(sc.alarm(y, k).mean()) for k in ("rel", "z")}

    # 1. 影: 5日間、発電量を f 倍にする
    rows = []
    for f in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2):
        for kind in ("rel", "z"):
            ev_hit, day_hit, by_level = [], [], {k: [] for k in range(5)}
            for s0 in trial_positions(n, 5, np.random.default_rng(int(f * 100)), N_TRIALS):
                yy = y.copy(); yy[s0:s0 + 5] *= f
                al = sc.alarm(yy, kind)[s0:s0 + 5]
                ev_hit.append(al.any()); day_hit.append(al.mean())
                by_level[int(np.median(sc.level[s0:s0 + 5]))].append(al.mean())
            row = dict(強さ=f"{int(round((1 - f) * 100))}%低下", スコア=kind, イベント検知率=round(float(np.mean(ev_hit)), 2),
                       日単位検知率=round(float(np.mean(day_hit)), 2))
            for k in range(5):
                row[f"日単位_期待{LEVEL_NAMES[k]}kWh"] = round(float(np.mean(by_level[k])), 2) if len(by_level[k]) >= MIN_TRIALS else np.nan
            rows.append(row)
    out["shade"] = pd.DataFrame(rows)

    # 2. 停電: 2日間、発電量を本来の5%にする
    rows = []
    for kind in ("rel", "z"):
        ev_hit, by_level = [], {k: [] for k in range(5)}
        for s0 in trial_positions(n, 2, np.random.default_rng(7), N_TRIALS):
            yy = y.copy(); yy[s0:s0 + 2] *= 0.05
            al = sc.alarm(yy, kind)[s0:s0 + 2]
            ev_hit.append(al.any()); by_level[int(np.median(sc.level[s0:s0 + 2]))].append(al.any())
        row = dict(スコア=kind, イベント検知率=round(float(np.mean(ev_hit)), 2))
        for k in range(5):
            row[f"期待{LEVEL_NAMES[k]}kWh"] = round(float(np.mean(by_level[k])), 2) if len(by_level[k]) >= MIN_TRIALS else np.nan
        rows.append(row)
    out["outage"] = pd.DataFrame(rows)

    # 3. 劣化: 40日間で直線的に最終低下率 depth まで悪化。その日の低下率の帯ごとに、警報が出た割合
    bins = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.6001]
    names = ["2〜5%", "5〜10%", "10〜20%", "20〜30%", "30〜40%", "40〜50%", "50%〜"]
    rows = []
    for kind in ("rel", "z"):
        cnt = np.zeros(len(names)); hit_d = np.zeros(len(names)); hit_a = np.zeros(len(names))
        for s0 in trial_positions(n, 40, np.random.default_rng(11), N_TRIALS):
            loss = np.zeros(n); loss[s0:s0 + 40] = 0.6 * np.arange(1, 41) / 40  # 最終60%低下まで
            yy = y * (1 - loss)
            al, dl = sc.alarm(yy, kind), sc.daily_alarm(yy, kind)
            for b in range(len(names)):
                m = (loss >= bins[b]) & (loss < bins[b + 1])
                cnt[b] += m.sum(); hit_d[b] += dl[m].sum(); hit_a[b] += al[m].sum()
        for b in range(len(names)):
            rows.append(dict(スコア=kind, 低下率=names[b], 日数=int(cnt[b]),
                             日次の警報率=round(hit_d[b] / max(cnt[b], 1), 2), 日次か7日平均の警報率=round(hit_a[b] / max(cnt[b], 1), 2)))
    out["degrade"] = pd.DataFrame(rows)

    # 4. 固着: 4日間、直前の日の値で固定
    hit = []
    for s0 in trial_positions(n, 4, np.random.default_rng(13), N_TRIALS):
        yy = y.copy(); yy[s0:s0 + 4] = y[s0 - 1]
        same = pd.Series(yy).eq(pd.Series(yy).shift(1)).to_numpy()
        hit.append(same[s0:s0 + 4].any())
    out["stuck"] = pd.DataFrame([dict(固着ルールのイベント検知率=round(float(np.mean(hit)), 2),
                                    実データの同値連続の日数=int(pd.Series(y).eq(pd.Series(y).shift(1)).sum()))])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    res = run()
    print("注入前の実データの警報率（誤警報の目安）:", {k: round(v, 3) for k, v in res["clean_alarm_rate"].items()})
    for name in ("shade", "outage", "degrade", "stuck"):
        print(f"\n== {name} ==")
        print(res[name].to_string(index=False))
        res[name].to_csv(OUT / f"inject_{name}.csv", index=False)


if __name__ == "__main__":
    main()
