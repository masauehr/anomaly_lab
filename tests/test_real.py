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
from run_inject import Scorer  # noqa: E402
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


class TestInject(unittest.TestCase):
    """注入実験の部品。実データの代わりに合成データ（異常なし）で動作を確かめる"""

    @classmethod
    def setUpClass(cls):
        cls.df = make_split("train", 1, gsr_noise=0.2)[DETECTOR_COLUMNS]  # 実データ並みにノイズを大きくした正常系列
        cls.sc = Scorer(cls.df)

    def test_注入前の警報率は目標付近(self):
        for kind in ("rel", "z"):
            self.assertLess(self.sc.alarm(self.sc.y, kind).mean(), 0.08, kind)

    def test_水準別の標準化は水準ごとに中央値が0付近(self):
        z = self.sc.score(self.sc.y, "z")
        for k in range(5):
            m = self.sc.level == k
            if m.sum() >= 20:
                self.assertLess(abs(np.median(z[m])), 0.3, k)

    def test_強い影は検知できる(self):
        y = self.sc.y.copy(); y[100:105] *= 0.2
        for kind in ("rel", "z"):
            self.assertTrue(self.sc.alarm(y, kind)[100:105].any(), kind)

    def test_注入は元の系列を書き換えない(self):
        y0 = self.sc.y.copy()
        yy = self.sc.y.copy(); yy[10:15] *= 0.5
        self.assertTrue(np.array_equal(self.sc.y, y0))

    def test_しきい値は注入前のデータだけで決まる(self):
        thr = dict(self.sc.thr)
        y = self.sc.y.copy(); y[50:60] = 0
        self.sc.alarm(y, "rel")
        self.assertEqual(thr, self.sc.thr)


if __name__ == "__main__":
    unittest.main()
