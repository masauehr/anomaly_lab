import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from anomaly_synth import make_split  # noqa: E402
from anomaly_synth.generator import ORACLE_COLUMNS  # noqa: E402
from baselines import DETECTOR_COLUMNS  # noqa: E402
from residual import FEATURES, fit_expected, make_features, out_of_fold_residual, relative_residual, smooth  # noqa: E402


class TestResidual(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = make_split("train")[DETECTOR_COLUMNS]

    def test_特徴量に答え列が含まれない(self):
        for kind, cols in FEATURES.items():
            self.assertFalse(set(cols) & set(ORACLE_COLUMNS), kind)
            self.assertTrue(set(cols) <= set(make_features(self.train, kind).columns), kind)

    def test_日射量が多いほど期待発電量が多い(self):
        m = fit_expected(self.train, "linear_gsr")
        lo = pd.DataFrame({"date": pd.to_datetime(["2024-06-01"]), "gsr": [5.0]})
        hi = pd.DataFrame({"date": pd.to_datetime(["2024-06-01"]), "gsr": [20.0]})
        self.assertGreater(m.predict(make_features(hi, "linear_gsr"))[0], m.predict(make_features(lo, "linear_gsr"))[0])

    def test_正常データの相対残差は0付近(self):
        r = out_of_fold_residual(self.train, "linear_gsr")
        self.assertLess(abs(r.mean()), 0.02)
        self.assertEqual(len(r), len(self.train))

    def test_停電は残差が大きい(self):
        val = make_split("val", 1)
        m = fit_expected(self.train, "linear_gsr")
        r = relative_residual(val[DETECTOR_COLUMNS], m, "linear_gsr")
        o = val["anomaly_type"] == "outage"
        self.assertGreater(r[o].min(), 0.5)
        self.assertLess(r[val["is_anomaly"] == 0].quantile(0.99), 0.4)

    def test_平滑化は今日と過去だけを使う(self):
        s = pd.Series([0.0] * 5 + [7.0] + [0.0] * 3)
        out = smooth(s, 7)
        self.assertAlmostEqual(out.iloc[4], 0.0)      # 未来の7は影響しない
        self.assertAlmostEqual(out.iloc[5], 7.0 / 6)  # 今日を含む（先頭は6日分しかない）
        self.assertTrue(smooth(s, 1).equals(s))


if __name__ == "__main__":
    unittest.main()
