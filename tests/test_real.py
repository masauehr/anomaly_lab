import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from anomaly_synth import make_split  # noqa: E402
from anomaly_synth.generator import ORACLE_COLUMNS  # noqa: E402
from baselines import DETECTOR_COLUMNS  # noqa: E402
from real_data import REAL_DIR, WEATHER_MAP, load_real  # noqa: E402
from run_phase6 import OUT, mad, real_residuals  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def dummy_real(d: Path, n: int = 40):
    """実データと同じ形式のダミー（個人データは使わない）"""
    dates = pd.date_range("2025-05-01", periods=n)
    rng = np.random.default_rng(0)
    gsr = rng.uniform(3, 25, n)
    pd.DataFrame({"date": dates, "generation_kwh": (0.8 * gsr).round(2)}).to_csv(d / "hatuden_daily.csv", index=False)
    pd.DataFrame({
        "date": dates, "GSR": gsr.round(2), "GSR_cli": 15.0, "SSD": 5.0, "TMP_max": 28.0, "TMP_mean": 24.0,
        "possible_sunshine": 12.0, "sunshine_ratio": rng.uniform(0, 100, n).round(1),
        "weather_class": rng.choice(list(WEATHER_MAP), n)}).to_csv(d / "hatuden_amd.csv", index=False)


class TestRealData(unittest.TestCase):
    def test_読み込み形式と変換(self):
        with tempfile.TemporaryDirectory() as t:
            dummy_real(Path(t))
            df = load_real(Path(t))
        self.assertTrue(set(DETECTOR_COLUMNS) <= set(df.columns))
        self.assertFalse(set(df.columns) & set(ORACLE_COLUMNS))
        self.assertTrue(df["sunshine_ratio"].between(0, 1).all())     # %表記を0〜1に変換
        self.assertTrue(set(df["weather_class"]) <= {0, 1, 2})
        self.assertTrue(df["date"].is_monotonic_increasing)

    def test_未知の天気分類は拒否する(self):
        with tempfile.TemporaryDirectory() as t:
            dummy_real(Path(t))
            amd = pd.read_csv(Path(t) / "hatuden_amd.csv")
            amd.loc[0, "weather_class"] = "雹"
            amd.to_csv(Path(t) / "hatuden_amd.csv", index=False)
            with self.assertRaises(ValueError):
                load_real(Path(t))

    def test_実データの参照先と出力先はリポジトリのgit管理外(self):
        self.assertNotEqual(REAL_DIR.resolve(), ROOT.resolve())
        self.assertFalse(str(REAL_DIR.resolve()).startswith(str(ROOT.resolve())))  # 実データはこのリポジトリの外
        self.assertEqual(OUT.resolve(), (ROOT / "outputs" / "real").resolve())
        ignore = (ROOT / ".gitignore").read_text()
        self.assertIn("outputs/*", ignore)  # outputs/real/ も除外される
        self.assertIn("data/real/", ignore)


class TestResidualsOnDummy(unittest.TestCase):
    def test_madは外れ値に強い(self):
        x = np.r_[np.random.default_rng(0).normal(0, 1, 500), [1000.0] * 5]
        self.assertAlmostEqual(mad(x), 1.0, delta=0.2)

    def test_残差の列と停電の検出(self):
        val = make_split("val", 1)[DETECTOR_COLUMNS + []]
        r = real_residuals(val)
        for c in ["pred", "short", "rel", "z_rel", "z_abs"]:
            self.assertIn(c, r.columns)
        out = make_split("val", 1)["anomaly_type"] == "outage"
        self.assertGreater(r.loc[out, "z_rel"].min(), 3)


if __name__ == "__main__":
    unittest.main()
