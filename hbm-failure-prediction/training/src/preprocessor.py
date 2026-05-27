"""
preprocessor.py
对原始 HBM 遥测数据进行清洗、时间轴对齐和滑动窗口切片。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL = "10min"   # 采样粒度


class HBMPreprocessor:
    """清洗并切分 HBM 遥测时间序列。"""

    def __init__(
        self,
        window_days: int = 30,
        step_days: int = 1,
        outlier_sigma: float = 3.0,
    ):
        self.window_days = window_days
        self.step_days = step_days
        self.outlier_sigma = outlier_sigma

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #

    def process(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """主入口：按 (server_id, hbm_id) 分组，返回清洗后的时间序列字典。

        Returns:
            dict: key = "server_id::hbm_id"，value = 对齐后的 DataFrame
        """
        results: Dict[str, pd.DataFrame] = {}
        groups = df.groupby(["server_id", "hbm_id"])
        for (srv, hbm), group_df in groups:
            key = f"{srv}::{hbm}"
            try:
                cleaned = self._process_single(group_df.copy(), hbm_id=hbm)
                results[key] = cleaned
            except Exception as exc:  # noqa: BLE001
                logger.warning("跳过 %s: %s", key, exc)
        return results

    def sliding_windows(
        self, ts: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.Timestamp]]:
        """在单条时间序列上生成滑动窗口样本。

        Returns:
            List of (window_df, window_end_timestamp)
        """
        freq = pd.tseries.frequencies.to_offset(SAMPLE_INTERVAL)
        window_size = self.window_days * 24 * 6   # 每天 144 个 10 分钟
        step_size = self.step_days * 24 * 6

        windows = []
        n = len(ts)
        start = 0
        while start + window_size <= n:
            end = start + window_size
            window_df = ts.iloc[start:end].copy()
            window_end_ts = ts.index[end - 1]
            windows.append((window_df, window_end_ts))
            start += step_size
        return windows

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #

    def _process_single(self, df: pd.DataFrame, hbm_id: int) -> pd.DataFrame:
        df = df.sort_values("timestamp").set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")]

        # 1. 时间轴重采样（前向填充缺失，ECC 计数用 0 填充）
        full_range = pd.date_range(df.index.min(), df.index.max(), freq=SAMPLE_INTERVAL, tz="UTC")
        df = df.reindex(full_range)
        df["ce_count"] = df["ce_count"].fillna(0).astype(int)
        df["ue_count"] = df["ue_count"].fillna(0).astype(int)
        df["fail_label"] = df["fail_label"].fillna(0).astype(int)
        df["temp_c"] = df["temp_c"].interpolate(method="linear").bfill().ffill()
        df["power_w"] = df["power_w"].interpolate(method="linear").bfill().ffill()
        df["ce_col_bitmap"] = df["ce_col_bitmap"].ffill()
        df["hbm_id"] = hbm_id

        # 2. 异常值检测（仅标记，不删除）
        df = self._detect_outliers(df)

        return df

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_outlier"] = False
        for col in ["ce_count", "ue_count", "temp_c", "power_w"]:
            if col not in df.columns:
                continue
            mu = df[col].mean()
            sigma = df[col].std()
            if sigma == 0:
                continue
            mask = (df[col] - mu).abs() > self.outlier_sigma * sigma
            df.loc[mask, "is_outlier"] = True
        return df

    def extract_failure_events(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """返回所有 fail_label=1 的时间戳列表（升序）。"""
        if "fail_label" not in df.columns:
            return []
        mask = df["fail_label"] == 1
        return sorted(df.index[mask].tolist())
