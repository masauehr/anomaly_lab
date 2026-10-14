import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from anomaly_synth import make_split  # noqa: E402
from baselines import DETECTOR_COLUMNS  # noqa: E402
from unsup import FEATURE_SETS, UnsupDetector, make_features, out_of_fold_score  # noqa: E402


class TestUnsup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = make_split("train")[DETECTOR_COLUMNS]

    def test_特徴量セットは増えていく(self):
        a, b, c = (make_features(self.train, f) for f in "ABC")
        self.assertLess(a.shape[1], b.shape[1])
        self.assertLess(b.shape[1], c.shape[1])
        self.assertTrue(set(a.columns) <= set(b.columns) <= set(c.columns))
        self.assertEqual(set(FEATURE_SETS), {"A", "B", "C"})

    def test_特徴量は欠損なし(self):
        for f in "ABC":
            self.assertFalse(make_features(self.train, f).isna().any().any(), f)

    def test_固着の日は前日との差が0(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=4), "gen_kwh": [5.0, 7.0, 7.0, 7.0],
            "gsr": [8.0] * 4, "sunshine_ratio": [0.5] * 4, "weather_class": [1] * 4})
        self.assertEqual(make_features(df, "C")["diff_prev"].tolist()[1:], [2.0, 0.0, 0.0])

    def test_スコアは決定的で長さが合う(self):
        val = make_split("val", 1)[DETECTOR_COLUMNS]
        for kind in ("iforest", "lof"):
            s1 = UnsupDetector(kind, "B").fit(self.train).score(val)
            s2 = UnsupDetector(kind, "B").fit(self.train).score(val)
            self.assertEqual(len(s1), len(val))
            self.assertTrue(np.allclose(s1, s2))

    def test_停電の日は正常日よりスコアが高い(self):
        val = make_split("val", 1)
        s = UnsupDetector("iforest", "B").fit(self.train).score(val[DETECTOR_COLUMNS])
        self.assertGreater(s[val["anomaly_type"] == "outage"].median(), s[val["is_anomaly"] == 0].median())

    def test_out_of_foldは全日を返す(self):
        s = out_of_fold_score(self.train, "iforest", "A")
        self.assertEqual(len(s), len(self.train))
        self.assertTrue((s.index == self.train.index).all())


if __name__ == "__main__":
    unittest.main()
