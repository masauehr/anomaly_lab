"""Phase1: AIを使わないベースライン検知器。

どの検知器も「異常スコア（大きいほど異常）」を返す。入力は検知器が見てよい列だけ（DETECTOR_COLUMNS）。
答え合わせ列（gen_clean_kwh・is_anomaly など）は渡さない。
"""
import numpy as np
import pandas as pd

# 検知器が見てよい列
DETECTOR_COLUMNS = ["date", "weather_class", "sunshine_ratio", "gsr", "gen_kwh"]


def score_low_value(df: pd.DataFrame) -> pd.Series:
    """発電量が小さいほど異常（固定しきい値型）。天気を考えない。"""
    return -df["gen_kwh"]


def score_moving_average_drop(df: pd.DataFrame, window: int = 7) -> pd.Series:
    """直近 window 日の平均（今日を含まない）からの落ち込み率。天気を考えない。

    スコア = (直近平均 − 今日) / 直近平均。先頭 window 日は計算できないので0にする。
    """
    past = df["gen_kwh"].shift(1).rolling(window).mean()
    return ((past - df["gen_kwh"]) / past.clip(lower=0.5)).fillna(0.0)


def score_seasonal_ratio(df: pd.DataFrame, ref: pd.DataFrame, half_window: int = 15) -> pd.Series:
    """季節平均との比の小ささ。ref（正常データ）の同じ時期（前後 half_window 日）の平均発電量と比べる。

    スコア = 1 − 今日 / 季節平均。天気を考えないため、曇り・雨の日が大きく外れる。
    """
    ref_doy = ref["date"].dt.dayofyear.to_numpy()
    ref_gen = ref["gen_kwh"].to_numpy()
    doy = df["date"].dt.dayofyear.to_numpy()
    mean = np.empty(len(df))
    for i, d in enumerate(doy):
        diff = np.abs(ref_doy - d)
        diff = np.minimum(diff, 365 - diff)  # 年末と年始を近いとみなす
        mean[i] = ref_gen[diff <= half_window].mean()
    return pd.Series(1.0 - df["gen_kwh"].to_numpy() / mean, index=df.index)


def score_stuck_run(df: pd.DataFrame) -> pd.Series:
    """同じ値が何日連続しているか（固着検知のルール）。1=連続なし。"""
    same = df["gen_kwh"].eq(df["gen_kwh"].shift(1))
    run = np.ones(len(df))
    for i in range(1, len(df)):
        run[i] = run[i - 1] + 1 if same.iloc[i] else 1
    return pd.Series(run, index=df.index)
