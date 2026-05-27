"""
data_loader.py
读取和校验原始 HBM ECC 日志 CSV 文件。

CSV 格式（列定义）：
  timestamp    : ISO8601 字符串 或 Unix 秒整数
  server_id    : 服务器唯一标识 (str)
  hbm_id       : HBM die 编号 0-7 (int)
  ce_count     : 该 10 分钟内 CE 错误计数 (int ≥ 0)
  ue_count     : 该 10 分钟内 UE 错误计数 (int ≥ 0)
  ce_col_bitmap: 列级 CE 分布，16进制字符串（32字节=64个hex字符），可为空
  temp_c       : HBM 温度（摄氏度，float）
  power_w      : HBM 功耗（瓦，float）
  fail_label   : 故障标签，1=该时刻发生故障，0=正常 (int)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "timestamp", "server_id", "hbm_id",
    "ce_count", "ue_count",
    "temp_c", "power_w", "fail_label",
]

OPTIONAL_COLUMNS = ["ce_col_bitmap"]


class HBMDataLoader:
    """加载并校验原始 HBM ECC 日志。"""

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: 若为 True，缺失必要列时抛出异常；否则仅打印警告。
        """
        self.strict = strict

    def load(self, csv_path: str | Path) -> pd.DataFrame:
        """读取单个 CSV 文件并进行基本校验。

        Returns:
            DataFrame，index 未设置，timestamp 列为 datetime64。
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

        logger.info("加载文件: %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)

        df = self._validate_columns(df)
        df = self._parse_timestamp(df)
        df = self._coerce_dtypes(df)
        df = self._basic_sanity_check(df)

        logger.info("加载完成: %d 行, %d 列", len(df), len(df.columns))
        return df

    def load_multiple(self, csv_paths: List[str | Path]) -> pd.DataFrame:
        """读取多个 CSV 并合并。"""
        frames = [self.load(p) for p in csv_paths]
        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values(["server_id", "hbm_id", "timestamp"], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        return combined

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #

    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            msg = f"缺少必要列: {missing}"
            if self.strict:
                raise ValueError(msg)
            logger.warning(msg)

        # 补充可选列（以 NaN 填充）
        for col in OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df

    def _parse_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in df.columns:
            return df
        # 尝试 Unix 整数
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def _coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        int_cols = ["ce_count", "ue_count", "fail_label", "hbm_id"]
        float_cols = ["temp_c", "power_w"]
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _basic_sanity_check(self, df: pd.DataFrame) -> pd.DataFrame:
        n_total = len(df)

        # CE/UE 不能为负
        for col in ["ce_count", "ue_count"]:
            if col in df.columns:
                n_neg = (df[col] < 0).sum()
                if n_neg > 0:
                    logger.warning("%s 有 %d 个负值，置为 0", col, n_neg)
                    df.loc[df[col] < 0, col] = 0

        # fail_label 只能为 0/1
        if "fail_label" in df.columns:
            invalid = ~df["fail_label"].isin([0, 1])
            if invalid.any():
                logger.warning("fail_label 中有 %d 个无效值，置为 0", invalid.sum())
                df.loc[invalid, "fail_label"] = 0

        # 统计故障比例
        if "fail_label" in df.columns:
            n_fail = df["fail_label"].sum()
            logger.info("故障样本: %d / %d (%.2f%%)", n_fail, n_total, 100 * n_fail / max(n_total, 1))

        return df


def load_raw_logs(csv_path: str | Path, strict: bool = True) -> pd.DataFrame:
    """便捷函数：加载单个 CSV。"""
    return HBMDataLoader(strict=strict).load(csv_path)
