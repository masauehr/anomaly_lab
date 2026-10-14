"""合成の日次太陽光発電量（正常＋注入した異常）を作るパッケージ。"""
from .generator import (  # noqa: F401
    ANOMALY_TYPES,
    DEFAULT_EVENTS,
    SPLIT_BASE,
    SPLIT_START,
    SPLIT_DAYS,
    make_split,
    generate_normal,
    inject_anomalies,
)
