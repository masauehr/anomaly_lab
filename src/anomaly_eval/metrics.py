"""評価指標。accuracy は使わない（異常が少ないと「全部正常」で高くなるため）。

警報は「異常スコアがしきい値を超えた日」。しきい値は正常データ（train）のスコア分位点で決める。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def alarm_by_normal_quantile(normal_scores, rate: float) -> float:
    """正常データのスコアの (1-rate) 分位点をしきい値にする。

    正常な日に警報が出る割合が、おおよそ rate になる（例: rate=0.03 なら約1か月に1回）。
    警報は score > しきい値 の日。
    """
    return float(np.quantile(np.asarray(normal_scores, dtype=float), 1.0 - rate))


def false_alarms_per_month(df: pd.DataFrame, alarm: np.ndarray) -> float:
    """正常日に出た警報の件数を、30日あたりに換算する。"""
    normal = (df["is_anomaly"] == 0).to_numpy()
    n_false = int((np.asarray(alarm, dtype=bool) & normal).sum())
    return n_false / normal.sum() * 30.0


def auprc(df: pd.DataFrame, score) -> float:
    """適合率—再現率曲線の下の面積。しきい値に依存しない異常度そのものの良さ（日単位）。"""
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=-np.inf)
    score = np.where(np.isneginf(score), np.nanmin(score[np.isfinite(score)]), score)
    return float(average_precision_score(df["is_anomaly"].to_numpy(), score))


def evaluate_events(df: pd.DataFrame, alarm: np.ndarray) -> pd.DataFrame:
    """異常イベント（anomaly_id）ごとに、検知できたか・検知までの日数を返す。

    検知 = イベントの異常日のどれかで警報が出たこと。
    検知までの日数 = 最初に警報が出た日 − イベントの最初の異常日（検知できたときのみ）。
    """
    alarm = np.asarray(alarm, dtype=bool)
    rows = []
    ev = df[df["anomaly_id"] > 0]
    for aid, g in ev.groupby("anomaly_id"):
        pos = g.index.to_numpy()  # dfは0からの連番を前提とする
        hit = np.where(alarm[pos])[0]
        rows.append(dict(
            anomaly_id=int(aid),
            type=g["anomaly_type"].iloc[0],
            days=len(g),
            detected=bool(len(hit)),
            delay=int(hit[0]) if len(hit) else np.nan,
            alarm_days=int(len(hit)),
        ))
    return pd.DataFrame(rows, columns=["anomaly_id", "type", "days", "detected", "delay", "alarm_days"])
