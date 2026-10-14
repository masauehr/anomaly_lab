import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from anomaly_eval import alarm_by_normal_quantile, auprc, evaluate_events, false_alarms_per_month  # noqa: E402
from baselines import DETECTOR_COLUMNS, score_moving_average_drop, score_stuck_run  # noqa: E402
from anomaly_synth import make_split  # noqa: E402
from anomaly_synth.generator import ORACLE_COLUMNS  # noqa: E402


def toy():
    # 10日。異常は 2〜3日目（id=1）と 7日目（id=2）
    return pd.DataFrame({
        "is_anomaly": [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
        "anomaly_id": [0, 0, 1, 1, 0, 0, 0, 2, 0, 0],
        "anomaly_type": ["none", "none", "outage", "outage", "none", "none", "none", "shade", "none", "none"],
    })


class TestMetrics(unittest.TestCase):
    def test_イベントの検知と遅れ(self):
        alarm = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        r = evaluate_events(toy(), alarm).set_index("anomaly_id")
        self.assertTrue(r.loc[1, "detected"])
        self.assertEqual(r.loc[1, "delay"], 1)  # 異常の2日目で初めて警報
        self.assertFalse(r.loc[2, "detected"])
        self.assertTrue(np.isnan(r.loc[2, "delay"]))

    def test_誤警報は正常日のみ数える(self):
        alarm = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)  # 正常日1回、異常日1回
        self.assertAlmostEqual(false_alarms_per_month(toy(), alarm), 1 / 7 * 30)

    def test_余韻の猶予日数(self):
        # 異常は3,4日目(index2,3)と8日目(index7)。直後2日以内の正常日(index4,5,8,9)は数えない
        alarm = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 0], dtype=bool)
        self.assertAlmostEqual(false_alarms_per_month(toy(), alarm, grace=0), 3 / 7 * 30)
        self.assertAlmostEqual(false_alarms_per_month(toy(), alarm, grace=2), 0.0)
        # 猶予の外（index6の次の日など）の警報は数える
        alarm2 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
        self.assertAlmostEqual(false_alarms_per_month(toy(), alarm2, grace=2), 1 / 3 * 30)

    def test_分位点しきい値(self):
        scores = np.arange(100)
        thr = alarm_by_normal_quantile(scores, 0.1)
        self.assertAlmostEqual((scores > thr).mean(), 0.1, delta=0.02)

    def test_auprcは完全分離で1(self):
        df = toy()
        self.assertAlmostEqual(auprc(df, df["is_anomaly"].to_numpy().astype(float)), 1.0)

    def test_検知器の入力に答え列が含まれない(self):
        self.assertFalse(set(DETECTOR_COLUMNS) & set(ORACLE_COLUMNS))
        self.assertTrue(set(DETECTOR_COLUMNS) <= set(make_split("val").columns))

    def test_移動平均は今日を含まない(self):
        df = pd.DataFrame({"gen_kwh": [10.0] * 8 + [0.0]})
        s = score_moving_average_drop(df, window=7)
        self.assertAlmostEqual(s.iloc[-1], 1.0)  # 直近平均10に対し今日0 → 落ち込み率1.0
        self.assertEqual(s.iloc[0], 0.0)

    def test_固着の連続日数(self):
        df = pd.DataFrame({"gen_kwh": [1.0, 2.0, 2.0, 2.0, 3.0]})
        self.assertEqual(score_stuck_run(df).tolist(), [1, 1, 2, 3, 1])


if __name__ == "__main__":
    unittest.main()
