"""Phase4: オートエンコーダ（正常データを圧縮して復元し、復元できない日を異常とする）。

入力は直近 window 日分の特徴量。小さな全結合ネットワークで、CPUで数秒で学習できる。
異常スコア = 復元誤差（入力と出力の二乗誤差の平均）。大きいほど「正常データのパターンから外れている」。
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

DAY_FEATURES = ["gen", "gsr", "sun", "w0", "w1", "w2", "season", "ratio"]


def day_features(df: pd.DataFrame) -> np.ndarray:
    """1日ぶんの特徴量（検知器に渡してよい列だけから作る）。天気は one-hot にする"""
    w = df["weather_class"].to_numpy()
    out = np.column_stack([
        df["gen_kwh"], df["gsr"], df["sunshine_ratio"],
        (w == 0), (w == 1), (w == 2),
        np.sin(2 * np.pi * df["date"].dt.dayofyear.to_numpy() / 365.25),
        df["gen_kwh"] / df["gsr"].clip(lower=1.0),
    ])
    return out.astype(np.float32)


def make_windows(f: np.ndarray, window: int) -> np.ndarray:
    """各日について、その日を最後とする直近 window 日分を1本のベクトルにする（先頭は最初の日で埋める）"""
    n = len(f)
    pad = np.vstack([np.repeat(f[:1], window - 1, axis=0), f]) if window > 1 else f
    return np.stack([pad[i:i + window].reshape(-1) for i in range(n)])


class AutoEncoder(nn.Module):
    def __init__(self, dim: int, code: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, code), nn.ReLU(),
            nn.Linear(code, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class AEDetector:
    """正常データで学習し、復元誤差を異常スコア（大きいほど異常）として返す"""

    def __init__(self, window: int = 1, code: int = 4, epochs: int = 300, seed: int = 0):
        self.window, self.code, self.epochs, self.seed = window, code, epochs, seed

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        return make_windows(day_features(df), self.window)

    def fit(self, train: pd.DataFrame):
        x = self._x(train)
        self.mean, self.std = x.mean(0), x.std(0) + 1e-6
        xt = torch.tensor((x - self.mean) / self.std, dtype=torch.float32)
        torch.manual_seed(self.seed)
        self.model = AutoEncoder(xt.shape[1], self.code)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.history = []  # エポックごとの学習時の復元誤差（学習曲線）
        for _ in range(self.epochs):  # 全データを1バッチで学習（データが小さいため）
            opt.zero_grad()
            loss = ((self.model(xt) - xt) ** 2).mean()
            loss.backward()
            opt.step()
            self.history.append(loss.item())
        self.final_loss = self.history[-1]
        return self

    def score(self, df: pd.DataFrame) -> pd.Series:
        xt = torch.tensor((self._x(df) - self.mean) / self.std, dtype=torch.float32)
        with torch.no_grad():
            err = ((self.model(xt) - xt) ** 2).mean(1).numpy()
        return pd.Series(err, index=df.index)


def out_of_fold_score(train: pd.DataFrame, **cfg) -> pd.Series:
    """trainのスコアを「その年を学習に使わないモデル」で計算する（年ごとのhold-out）"""
    parts = []
    for year in sorted(train["date"].dt.year.unique()):
        te = (train["date"].dt.year == year).to_numpy()
        det = AEDetector(**cfg).fit(train[~te].reset_index(drop=True))
        parts.append(det.score(train)[te])
    return pd.concat(parts).sort_index()
