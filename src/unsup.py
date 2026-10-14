"""Phase3: 教師なし手法（Isolation Forest・LOF）による検知。

期待値モデルは作らず、「正常データ（train）の分布から外れた日」を異常とする。
特徴量の作り方（どこまで人が教えるか）で結果が大きく変わることを確かめる。
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# 特徴量セット。A→B→C と、人が「異常の見つけ方」を教える度合いが増える
FEATURE_SETS = {
    "A": "素の値（発電量・日射量・日照率・天気・季節）",
    "B": "A＋発電量/日射量の比",
    "C": "B＋比の7日平均＋前日との差の絶対値",
}


def make_features(df: pd.DataFrame, fset: str) -> pd.DataFrame:
    """検知器に渡してよい列だけから特徴量を作る"""
    x = pd.DataFrame(index=df.index)
    x["gen"] = df["gen_kwh"]
    x["gsr"] = df["gsr"]
    x["sun"] = df["sunshine_ratio"]
    x["weather"] = df["weather_class"]
    x["season"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365.25)  # 年末と年始がつながるように
    if fset in ("B", "C"):
        x["ratio"] = df["gen_kwh"] / df["gsr"].clip(lower=1.0)
    if fset == "C":
        x["ratio_7d"] = x["ratio"].rolling(7, min_periods=1).mean()
        x["diff_prev"] = df["gen_kwh"].diff().abs().fillna(1.0)  # 前日との差。固着では0になる
    return x


def make_model(kind: str):
    if kind == "iforest":
        return IsolationForest(n_estimators=200, random_state=0, n_jobs=1)
    return LocalOutlierFactor(n_neighbors=20, novelty=True)


class UnsupDetector:
    """正常データで学習し、異常スコア（大きいほど異常）を返す"""

    def __init__(self, kind: str, fset: str):
        self.kind, self.fset = kind, fset

    def fit(self, train: pd.DataFrame):
        x = make_features(train, self.fset)
        self.scaler = StandardScaler().fit(x)
        self.model = make_model(self.kind).fit(self.scaler.transform(x))
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        x = self.scaler.transform(make_features(df, self.fset))
        return pd.Series(-self.model.score_samples(x), index=df.index)


def out_of_fold_score(train: pd.DataFrame, kind: str, fset: str) -> pd.Series:
    """trainのスコアを「その年を学習に使わないモデル」で計算する（年ごとのhold-out）。

    特徴量（前日との差・7日平均）は系列全体で作ってから、年ごとに切り分ける。
    """
    parts = []
    for year in sorted(train["date"].dt.year.unique()):
        te = (train["date"].dt.year == year).to_numpy()
        det = UnsupDetector(kind, fset).fit(train[~te].reset_index(drop=True))
        parts.append(det.score(train[te]))
    return pd.concat(parts).sort_index()
