import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ae import DAY_FEATURES, AEDetector, day_features, make_windows, out_of_fold_score  # noqa: E402
from anomaly_synth import make_split  # noqa: E402
from baselines import DETECTOR_COLUMNS  # noqa: E402


class TestAE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = make_split("train")[DETECTOR_COLUMNS]
        cls.val = make_split("val", 1)

    def test_日の特徴量の形(self):
        f = day_features(self.train)
        self.assertEqual(f.shape, (len(self.train), len(DAY_FEATURES)))
        self.assertFalse(np.isnan(f).any())
        self.assertTrue((f[:, 3:6].sum(1) == 1).all())  # 天気のone-hot

    def test_窓は直近の日を最後に並べ_先頭は最初の日で埋める(self):
        f = np.arange(5, dtype=np.float32).reshape(-1, 1)
        w = make_windows(f, 3)
        self.assertEqual(w.shape, (5, 3))
        self.assertEqual(w[0].tolist(), [0, 0, 0])
        self.assertEqual(w[3].tolist(), [1, 2, 3])
        self.assertTrue(np.array_equal(make_windows(f, 1), f))

    def test_同じシードなら同じスコア(self):
        s1 = AEDetector(1, 2, epochs=50, seed=0).fit(self.train).score(self.val[DETECTOR_COLUMNS])
        s2 = AEDetector(1, 2, epochs=50, seed=0).fit(self.train).score(self.val[DETECTOR_COLUMNS])
        s3 = AEDetector(1, 2, epochs=50, seed=1).fit(self.train).score(self.val[DETECTOR_COLUMNS])
        self.assertTrue(np.allclose(s1, s2))
        self.assertFalse(np.allclose(s1, s3))

    def test_学習で復元誤差が下がる(self):
        d = AEDetector(1, 4, epochs=200).fit(self.train)
        self.assertEqual(len(d.history), 200)
        self.assertLess(d.history[-1], d.history[0] * 0.5)

    def test_停電は正常日より復元誤差が大きい(self):
        d = AEDetector(1, 4, epochs=300).fit(self.train)
        s = d.score(self.val[DETECTOR_COLUMNS])
        self.assertGreater(s[self.val["anomaly_type"] == "outage"].median(),
                           s[self.val["is_anomaly"] == 0].median() * 3)

    def test_out_of_foldは全日を返す(self):
        s = out_of_fold_score(self.train, window=1, code=2, epochs=30)
        self.assertEqual(len(s), len(self.train))
        self.assertTrue((s.index == self.train.index).all())


if __name__ == "__main__":
    unittest.main()
