import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import final_eval  # noqa: E402
from anomaly_synth import make_split  # noqa: E402
from combined import CombinedDetector  # noqa: E402


class TestCombined(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = make_split("train")
        cls.det = CombinedDetector(0.03, 0.0, True).fit(cls.train)

    def test_固着ルールが固着の日に警報を出す(self):
        val = make_split("val", 1)
        comp = self.det.components(val)
        s = val["anomaly_type"] == "stuck"
        self.assertTrue(comp.loc[s, "stuck"].all())
        self.assertLess(comp.loc[val["is_anomaly"] == 0, "stuck"].mean(), 0.01)

    def test_使わない部品は警報を出さない(self):
        comp = self.det.components(make_split("val", 1))
        self.assertFalse(comp["avg7"].any())  # rate_7d=0
        comp2 = CombinedDetector(0.03, 0.0, False).fit(self.train).components(make_split("val", 1))
        self.assertFalse(comp2["stuck"].any())

    def test_警報はどれか1つの部品があれば出る(self):
        val = make_split("val", 1)
        comp = self.det.components(val)
        self.assertTrue(np.array_equal(self.det.alarm(val), comp.any(axis=1).to_numpy()))

    def test_正常日の警報率は目標に近い(self):
        df = make_split("val", 1, events={})
        self.assertLess(self.det.alarm(df).mean(), 0.06)


class TestSensitivityParams(unittest.TestCase):
    def test_既定値は従来と同じ(self):
        a = make_split("val", 3)
        b = make_split("val", 3, degrade_depth=(0.05, 0.15), shade_factor=(0.6, 0.8))
        self.assertTrue(a.equals(b))

    def test_日射の計測誤差の既定値は従来と同じ(self):
        a = make_split("val", 3)
        self.assertTrue(a.equals(make_split("val", 3, gsr_noise=0.04)))
        noisy = make_split("val", 3, gsr_noise=0.2)
        self.assertTrue(a["gen_kwh"].equals(noisy["gen_kwh"]))  # 発電量は変わらない
        self.assertFalse(a["gsr"].equals(noisy["gsr"]))          # 日射量の計測だけが荒くなる

    def test_影の強さを指定できる(self):
        df = make_split("val", 3, events={"shade": 2}, shade_factor=(0.9, 0.9))
        s = df[df["anomaly_type"] == "shade"]
        self.assertTrue(np.allclose(s["gen_kwh"] / s["gen_clean_kwh"], 0.9, atol=0.01))

    def test_劣化の深さを指定できる(self):
        df = make_split("val", 3, events={"degrade": 1}, degrade_depth=(0.2, 0.2))
        g = df[df["anomaly_type"] == "degrade"]
        self.assertAlmostEqual(1 - (g["gen_kwh"] / g["gen_clean_kwh"]).min(), 0.2, delta=0.02)


class TestFinalEvalLock(unittest.TestCase):
    def test_使用済みならtestを拒否する(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "TEST_USED.json").write_text(json.dumps({"used_at": "x"}))
            with mock.patch.object(final_eval, "OUT", Path(d)), \
                    mock.patch.object(sys, "argv", ["final_eval.py", "--split", "test"]):
                with self.assertRaises(SystemExit) as cm:
                    final_eval.main()
            self.assertIn("使用済み", str(cm.exception))

    def test_valとtestの系列が重ならない(self):
        v, t = set(final_eval.INDEXES["val"]), set(final_eval.INDEXES["test"])
        self.assertTrue(min(v) > 10)  # 開発に使った val の index=1〜10 とは別
        self.assertTrue(max(t) < 1000)  # index>=1000 は追加testの予約


if __name__ == "__main__":
    unittest.main()
