"""Phase2: 天気から期待発電量を予測し、実測との残差で異常を検知する。

「少ない」ではなく「天気から期待される量より少ない」を異常とみなす。
期待値のモデルは正常データ（train）だけで学習する。
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# 期待値モデルの種類 → 使う特徴量（検知器に渡してよい列だけ）
FEATURES = {
    "linear_gsr": ["gsr"],
    "linear_all": ["gsr", "sunshine_ratio", "weather_class"],
    "rf": ["gsr", "sunshine_ratio", "weather_class", "doy"],
}


def make_features(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    x = df.copy()
    x["doy"] = x["date"].dt.dayofyear
    return x[FEATURES[kind]]


def make_model(kind: str):
    if kind.startswith("linear"):
        return LinearRegression()
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0, n_jobs=1)


def fit_expected(train: pd.DataFrame, kind: str):
    """正常データから「期待発電量」のモデルを学習する"""
    m = make_model(kind)
    m.fit(make_features(train, kind), train["gen_kwh"])
    return m


def relative_residual(df: pd.DataFrame, model, kind: str) -> pd.Series:
    """相対残差 = (期待値 − 実測) / 期待値。大きいほど「期待より少ない」。

    発電のばらつきは大きさに比例する（約3%）ので、差そのものでなく割合で見る。
    """
    exp = np.clip(model.predict(make_features(df, kind)), 0.5, None)
    return pd.Series((exp - df["gen_kwh"].to_numpy()) / exp, index=df.index)


def smooth(s: pd.Series, window: int) -> pd.Series:
    """直近 window 日の平均（今日を含む）。徐々に進む異常（劣化）を拾いやすくする"""
    return s if window <= 1 else s.rolling(window, min_periods=1).mean()


def out_of_fold_residual(train: pd.DataFrame, kind: str) -> pd.Series:
    """trainの残差を「その年を学習に使わないモデル」で計算する（年ごとのhold-out）。

    学習に使った日の残差は小さく出る（特にRandomForest）ため、しきい値は学習に使っていない年の残差で決める。
    """
    parts = []
    for year in sorted(train["date"].dt.year.unique()):
        te = train["date"].dt.year == year
        model = fit_expected(train[~te].reset_index(drop=True), kind)
        parts.append(relative_residual(train[te], model, kind))
    return pd.concat(parts).sort_index()
