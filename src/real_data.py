"""Phase6: 自宅の実データ（日次発電量＋気象）を、検知器が使う形に変換する。

🔒 個人データ（自宅の発電量）の扱い:
  - ml_forecast/data/ を**読み取り専用**で参照する。このリポジトリへコピーしない。
  - 結果（検知した日付・図・数値の詳細）は outputs/real/（git管理外）にだけ書く。コミットしない。
"""
from pathlib import Path

import pandas as pd

# ml_forecast のデータ置き場（このリポジトリの隣）
REAL_DIR = Path(__file__).resolve().parent.parent.parent / "ml_forecast" / "data"
WEATHER_MAP = {"快晴": 0, "晴れ": 0, "薄曇り〜曇り": 1, "曇り・雨": 2}  # 合成データの 0=晴 1=曇 2=雨 に合わせる


def load_real(base: Path = REAL_DIR) -> pd.DataFrame:
    """検知器に渡してよい列（date, weather_class, sunshine_ratio, gsr, gen_kwh）だけを返す"""
    daily = pd.read_csv(base / "hatuden_daily.csv", parse_dates=["date"])
    amd = pd.read_csv(base / "hatuden_amd.csv", parse_dates=["date"])
    m = daily.merge(amd, on="date", how="inner").sort_values("date").reset_index(drop=True)
    unknown = set(m["weather_class"].dropna()) - set(WEATHER_MAP)
    if unknown:
        raise ValueError(f"未知の天気分類: {unknown}")
    return pd.DataFrame({
        "date": m["date"],
        "weather_class": m["weather_class"].map(WEATHER_MAP).fillna(1).astype(int),
        "sunshine_ratio": (m["sunshine_ratio"] / 100.0).clip(0, 1),  # 実データは%表記
        "gsr": m["GSR"],
        "gen_kwh": m["generation_kwh"],
        "tmax": m["TMP_max"],  # 検知器の標準入力ではない（気温の影響を調べる用）
    })
