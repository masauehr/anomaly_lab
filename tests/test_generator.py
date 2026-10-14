import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from anomaly_synth import DEFAULT_EVENTS, SPLIT_BASE, make_split, generate_normal, inject_anomalies  # noqa: E402


class TestGenerator(unittest.TestCase):
    def test_再現性(self):
        a = make_split("val", 0)
        b = make_split("val", 0)
        self.assertTrue(a.equals(b))

    def test_シードで変わる(self):
        a = make_split("val", 0)
        b = make_split("val", 1)
        self.assertFalse(a["gen_kwh"].equals(b["gen_kwh"]))

    def test_trainは正常のみ(self):
        tr = make_split("train")
        self.assertEqual(tr["is_anomaly"].sum(), 0)
        self.assertTrue((tr["gen_kwh"] == tr["gen_clean_kwh"]).all())

    def test_分割は時間順で重ならない(self):
        tr, va, te = make_split("train"), make_split("val"), make_split("test")
        self.assertLess(tr["date"].max(), va["date"].min())
        self.assertLess(va["date"].max(), te["date"].min())

    def test_シード範囲の分離(self):
        bases = sorted(SPLIT_BASE.values())
        self.assertTrue(all(b - a >= 1_000_000 for a, b in zip(bases, bases[1:])))

    def test_正常日は値が変わらない(self):
        df = make_split("val")
        ok = df["is_anomaly"] == 0
        self.assertTrue((df.loc[ok, "gen_kwh"] == df.loc[ok, "gen_clean_kwh"]).all())

    def test_異常イベント数(self):
        df = make_split("test")
        for kind, n in DEFAULT_EVENTS.items():
            ids = df.loc[df["anomaly_type"] == kind, "anomaly_id"].nunique()
            self.assertEqual(ids, n, kind)

    def test_停電は発電が極端に小さい(self):
        df = make_split("val")
        o = df[df["anomaly_type"] == "outage"]
        self.assertTrue((o["gen_kwh"] <= o["gen_clean_kwh"] * 0.1 + 0.01).all())

    def test_影は2割から4割低下(self):
        df = make_split("val")
        s = df[df["anomaly_type"] == "shade"]
        r = s["gen_kwh"] / s["gen_clean_kwh"]
        self.assertTrue(((r > 0.58) & (r < 0.82)).all())

    def test_固着は同じ値が続く(self):
        df = make_split("val")
        for _, g in df[df["anomaly_type"] == "stuck"].groupby("anomaly_id"):
            self.assertEqual(g["gen_kwh"].nunique(), 1)
            prev = df.loc[g.index[0] - 1, "gen_kwh"]
            self.assertEqual(g["gen_kwh"].iloc[0], prev)

    def test_劣化は徐々に悪化し2パーセント以上を異常日とする(self):
        df = make_split("val")
        for _, g in df[df["anomaly_type"] == "degrade"].groupby("anomaly_id"):
            r = (g["gen_kwh"] / g["gen_clean_kwh"]).to_numpy()
            self.assertTrue((r <= 0.98 + 0.01).all())
            self.assertGreater(len(g), 12)

    def test_異常同士は重ならず余白がある(self):
        df = make_split("test")
        pos = np.where(df["is_anomaly"].to_numpy() == 1)[0]
        ids = df["anomaly_id"].to_numpy()
        for i, j in zip(pos, pos[1:]):
            if ids[i] != ids[j]:
                self.assertGreaterEqual(j - i, 5)

    def test_注入は元のdfを変えない(self):
        base = generate_normal("2024-01-01", 200, 1)
        copy = base.copy()
        inject_anomalies(base, 2, {"outage": 2})
        self.assertTrue(base.equals(copy))

    def test_値の範囲(self):
        df = make_split("train")
        self.assertTrue((df["gen_kwh"] >= 0).all())
        self.assertTrue(df["gen_kwh"].max() < 30)
        self.assertTrue(df["sunshine_ratio"].between(0, 1).all())

    def test_天気で発電量が変わる(self):
        df = make_split("train")
        m = df.groupby("weather_class")["gen_kwh"].mean()
        self.assertGreater(m[0], m[1])
        self.assertGreater(m[1], m[2])


if __name__ == "__main__":
    unittest.main()
