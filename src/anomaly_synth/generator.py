"""合成データ生成器。

正常な日次発電量を作り、そこへ異常（停電・劣化・影・固着）を注入する。
異常の種類と位置は正解（is_anomaly / anomaly_type）として列に持つ。検知器に渡してはいけない列は
README・DATA_SPEC に明記してある（ORACLE_COLUMNS）。
"""
import numpy as np
import pandas as pd

# 異常の種類（"none" は正常）
ANOMALY_TYPES = ["none", "outage", "degrade", "shade", "stuck"]

# split_seed の範囲（plate_reader_lab と同じ流儀。分割ごとに桁を分ける）
SPLIT_BASE = {"train": 1_000_000, "val": 2_000_000, "test": 3_000_000}
# 分割は時間順（未来を学習に使わない）
SPLIT_START = {"train": "2022-01-01", "val": "2024-01-01", "test": "2025-01-01"}
SPLIT_DAYS = {"train": 730, "val": 366, "test": 365}

# 1分割あたりの異常イベント数（trainは常に0）。異常日は全体の1〜2割になる（実運用より多いが、評価の統計を取るため）
DEFAULT_EVENTS = {"outage": 4, "degrade": 1, "shade": 2, "stuck": 2}

# 検知器に渡してはいけない列（答えそのもの、または答えが分かる列）
ORACLE_COLUMNS = ["gen_clean_kwh", "is_anomaly", "anomaly_type", "anomaly_id"]

KWH_PER_MJ = 0.85  # 日射量(MJ/m2)から発電量(kWh)への換算係数（約4.5kWpのシステムを想定）


def generate_normal(start: str, n_days: int, seed: int, gsr_noise: float = 0.04) -> pd.DataFrame:
    """正常な日次系列を作る。天気は晴/曇/雨のマルコフ連鎖、季節は正弦波。

    gsr_noise は日射量の計測誤差（割合）。実データ並みに大きくして、検知限界の変化を調べるときに使う。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy()
    season = np.sin(2 * np.pi * (doy - 80) / 365.25)  # 夏至付近で最大

    # 晴天時の日射量（MJ/m2）。冬約10、夏約26
    clear_gsr = 18.0 + 8.0 * season

    # 天気: 0=晴, 1=曇, 2=雨。梅雨（5〜7月）は雨が増える
    month = dates.month.to_numpy()
    rain_boost = np.where((month >= 5) & (month <= 7), 0.15, 0.0)
    weather = np.zeros(n_days, dtype=int)
    for i in range(1, n_days):
        prev = weather[i - 1]
        p_rain = np.clip({0: 0.10, 1: 0.25, 2: 0.45}[prev] + rain_boost[i], 0, 0.9)
        p_cloud = {0: 0.25, 1: 0.45, 2: 0.30}[prev]
        u = rng.random()
        weather[i] = 2 if u < p_rain else (1 if u < p_rain + p_cloud else 0)

    # 天気ごとの日射の割合（晴天比）
    base = np.array([0.92, 0.55, 0.20])[weather]
    spread = np.array([0.04, 0.12, 0.08])[weather]
    ratio = np.clip(base + spread * rng.standard_normal(n_days), 0.03, 1.0)

    gsr_true = clear_gsr * ratio
    gen = KWH_PER_MJ * gsr_true * (1 + 0.03 * rng.standard_normal(n_days))
    # 検知器が見る共変量は計測誤差を含む
    gsr_obs = gsr_true * (1 + gsr_noise * rng.standard_normal(n_days))
    sunshine = np.clip(ratio + 0.05 * rng.standard_normal(n_days), 0, 1)

    return pd.DataFrame({
        "date": dates,
        "weather_class": weather,
        "sunshine_ratio": sunshine.round(3),
        "gsr": gsr_obs.round(2),
        "gen_kwh": gen.round(2),
        "gen_clean_kwh": gen.round(2),
        "is_anomaly": 0,
        "anomaly_type": "none",
        "anomaly_id": 0,
    })


def _free_start(rng, occupied: np.ndarray, length: int, margin: int = 5, tries: int = 1000) -> int:
    """他の異常と重ならない開始位置を選ぶ（前後にmarginの余白）。系列の先頭30日は避ける。"""
    n = len(occupied)
    for _ in range(tries):
        s = int(rng.integers(30, n - length - margin))
        lo, hi = max(0, s - margin), min(n, s + length + margin)
        if not occupied[lo:hi].any():
            return s
    raise RuntimeError("異常を置く空きが見つからない（イベント数が多すぎる）")


def inject_anomalies(df: pd.DataFrame, seed: int, events: dict | None = None,
                     degrade_depth: tuple = (0.05, 0.15), shade_factor: tuple = (0.6, 0.8)) -> pd.DataFrame:
    """正常系列に異常を注入して返す（元のdfは変更しない）。

    注入の乱数は正常系列とは別のrng（seed）で動かすため、同じ正常系列に別の異常を重ねられる。
    degrade_depth=劣化の最終的な低下率の範囲、shade_factor=影の日の発電量の割合の範囲（感度分析で変える）。
    """
    events = DEFAULT_EVENTS if events is None else events
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    occupied = np.zeros(n, dtype=bool)
    gen = out["gen_kwh"].to_numpy(dtype=float).copy()
    clean = out["gen_clean_kwh"].to_numpy(dtype=float)
    label = out["anomaly_type"].to_numpy(dtype=object).copy()
    ident = out["anomaly_id"].to_numpy(dtype=int).copy()
    next_id = 1

    for kind in ["outage", "degrade", "shade", "stuck"]:
        for _ in range(int(events.get(kind, 0))):
            if kind == "outage":
                length = int(rng.integers(1, 4))
            elif kind == "degrade":
                length = int(rng.integers(30, 51))
            elif kind == "shade":
                length = int(rng.integers(3, 9))
            else:
                length = int(rng.integers(3, 7))
            s = _free_start(rng, occupied, length)
            sl = slice(s, s + length)
            occupied[sl] = True

            if kind == "outage":
                gen[sl] = clean[sl] * rng.uniform(0.0, 0.1, length)
                mark = np.ones(length, dtype=bool)
            elif kind == "degrade":
                depth = rng.uniform(*degrade_depth)  # 最終的な低下率
                loss = depth * np.arange(1, length + 1) / length  # 直線的に悪化
                mark = loss >= 0.02  # 低下が2%に満たない序盤は「異常日」に数えない（値も変えない）
                gen[sl] = np.where(mark, clean[sl] * (1 - loss), gen[sl])
            elif kind == "shade":
                gen[sl] = clean[sl] * rng.uniform(*shade_factor)
                mark = np.ones(length, dtype=bool)
            else:  # stuck: 開始前日の値で固着する（共変量は動き続ける）
                gen[sl] = gen[s - 1]
                mark = np.ones(length, dtype=bool)

            idx = np.arange(s, s + length)[mark]
            label[idx] = kind
            ident[idx] = next_id
            next_id += 1

    out["gen_kwh"] = np.round(gen, 2)
    out["anomaly_type"] = label
    out["anomaly_id"] = ident
    out["is_anomaly"] = (out["anomaly_type"] != "none").astype(int)
    return out


def make_split(split: str, index: int = 0, events: dict | None = None, gsr_noise: float = 0.04,
               **inject_kwargs) -> pd.DataFrame:
    """分割（train/val/test）のデータを作る。index は同じ分割内の別系列（split_seed = 基準 + index）。

    train は正常のみ（異常を注入しない）。index が 1000 以上のtest系列は「testの追加系列」として使える。
    """
    seed = SPLIT_BASE[split] + index
    df = generate_normal(SPLIT_START[split], SPLIT_DAYS[split], seed, gsr_noise)
    if split == "train":
        return df
    # 異常注入用のシードは、正常系列とは別の流れにする（+500_000）
    return inject_anomalies(df, seed + 500_000, events, **inject_kwargs)
