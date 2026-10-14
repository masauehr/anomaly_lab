"""Phase5: 検知器の組み合わせ（残差の日次 ＋ 残差の7日平均 ＋ 固着ルール）。

どれか1つでも警報を出せば警報（OR）。各部品のしきい値は、正常データ（train）で決める。
  - 日次の残差: 停電・影・突発的な落ち込みに強い
  - 7日平均の残差: 徐々に進む劣化に強い（ただし異常直後に余韻が残る）
  - 固着ルール: 同じ値が続く（残差では見つけにくい固着）
"""
import numpy as np
import pandas as pd

from anomaly_eval import alarm_by_normal_quantile
from baselines import DETECTOR_COLUMNS, score_stuck_run
from residual import fit_expected, out_of_fold_residual, relative_residual, smooth


class CombinedDetector:
    """rate_daily / rate_7d は、正常日にその部品が警報を出す割合の目標。0なら使わない"""

    def __init__(self, rate_daily: float = 0.03, rate_7d: float = 0.0, use_stuck: bool = True, kind: str = "linear_gsr"):
        self.rate_daily, self.rate_7d, self.use_stuck, self.kind = rate_daily, rate_7d, use_stuck, kind

    def fit(self, train: pd.DataFrame):
        tr = train[DETECTOR_COLUMNS]
        self.model = fit_expected(tr, self.kind)
        oof = out_of_fold_residual(tr, self.kind)
        self.thr_daily = alarm_by_normal_quantile(oof, self.rate_daily) if self.rate_daily > 0 else np.inf
        self.thr_7d = alarm_by_normal_quantile(smooth(oof, 7), self.rate_7d) if self.rate_7d > 0 else np.inf
        return self

    def components(self, df: pd.DataFrame) -> pd.DataFrame:
        """部品ごとの警報（bool）。検知器に渡してよい列だけを使う"""
        x = df[DETECTOR_COLUMNS]
        resid = relative_residual(x, self.model, self.kind)
        out = pd.DataFrame(index=df.index)
        out["daily"] = (resid > self.thr_daily).to_numpy()
        out["avg7"] = (smooth(resid, 7) > self.thr_7d).to_numpy()
        out["stuck"] = (score_stuck_run(x) > 1).to_numpy() if self.use_stuck else False
        return out

    def alarm(self, df: pd.DataFrame) -> np.ndarray:
        return self.components(df).any(axis=1).to_numpy()
