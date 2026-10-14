"""合成データ（train/val/test）をCSVに書き出し、確認用の図を作る。

使い方: .venv/bin/python src/make_data.py
出力:   data/{train,val,test}.csv / outputs/phase0_series.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anomaly_synth import make_split  # noqa: E402

plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "DejaVu Sans"]
ROOT = Path(__file__).resolve().parent.parent
COLORS = {"outage": "red", "degrade": "orange", "shade": "purple", "stuck": "green"}


def main() -> None:
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    for ax, split in zip(axes, ["train", "val", "test"]):
        df = make_split(split)
        df.to_csv(ROOT / "data" / f"{split}.csv", index=False)
        ax.plot(df["date"], df["gen_kwh"], lw=0.8, color="gray", label="発電量")
        for kind, c in COLORS.items():
            d = df[df["anomaly_type"] == kind]
            ax.scatter(d["date"], d["gen_kwh"], s=14, color=c, label=kind, zorder=3)
        ax.set(title=f"{split}（異常 {int(df['is_anomaly'].sum())} 日）", ylabel="kWh")
        print(split, len(df), "日", "異常", int(df["is_anomaly"].sum()), df["anomaly_type"].value_counts().to_dict())
    axes[0].legend(ncol=5, loc="upper right")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "phase0_series.png", dpi=100)


if __name__ == "__main__":
    main()
