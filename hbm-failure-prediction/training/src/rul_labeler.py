"""
rul_labeler.py
为每个滑动窗口样本生成 RUL（剩余使用寿命）标签。

RUL 定义：
  RUL(window_end = d_k) = min(d_next_fail - d_k, RUL_CAP)  [单位：天]

  - 若窗口结束后无故障事件，RUL = RUL_CAP（健康）
  - 故障当天 RUL = 0
  - 高危样本（RUL ≤ high_risk_rul）赋予更高采样权重
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RULLabeler:
    """根据故障事件时间戳生成 RUL 标签。"""

    def __init__(self, rul_cap: int = 30, high_risk_rul: int = 7, high_risk_weight: float = 3.0):
        self.rul_cap = rul_cap
        self.high_risk_rul = high_risk_rul
        self.high_risk_weight = high_risk_weight

    def label_windows(
        self,
        windows: List[Tuple[pd.DataFrame, pd.Timestamp]],
        failure_events: List[pd.Timestamp],
    ) -> List[Tuple[pd.DataFrame, pd.Timestamp, float, float]]:
        """为每个 (window_df, window_end_ts) 分配 RUL 标签和样本权重。

        Returns:
            List of (window_df, window_end_ts, rul_days, sample_weight)
        """
        labeled = []
        for window_df, end_ts in windows:
            rul = self._compute_rul(end_ts, failure_events)
            weight = self.high_risk_weight if rul <= self.high_risk_rul else 1.0
            labeled.append((window_df, end_ts, rul, weight))
        return labeled

    def _compute_rul(
        self, window_end: pd.Timestamp, failure_events: List[pd.Timestamp]
    ) -> float:
        """计算单个窗口的 RUL（天）。"""
        future_failures = [f for f in failure_events if f > window_end]
        if not future_failures:
            return float(self.rul_cap)
        next_fail = min(future_failures)
        delta_days = (next_fail - window_end).total_seconds() / 86400.0
        return float(min(delta_days, self.rul_cap))

    def validate_distribution(self, rul_values: List[float]) -> None:
        """打印 RUL 分布统计，检查样本不平衡情况。"""
        arr = np.array(rul_values)
        n = len(arr)
        n_critical = (arr <= 3).sum()
        n_warning = ((arr > 3) & (arr <= 7)).sum()
        n_watch = ((arr > 7) & (arr <= 14)).sum()
        n_normal = (arr > 14).sum()
        logger.info("RUL 分布 (共 %d 样本):", n)
        logger.info("  CRITICAL (≤3天):  %d (%.1f%%)", n_critical, 100 * n_critical / max(n, 1))
        logger.info("  WARNING  (3-7天): %d (%.1f%%)", n_warning, 100 * n_warning / max(n, 1))
        logger.info("  WATCH   (7-14天): %d (%.1f%%)", n_watch, 100 * n_watch / max(n, 1))
        logger.info("  NORMAL  (>14天):  %d (%.1f%%)", n_normal, 100 * n_normal / max(n, 1))

        high_risk_ratio = (n_critical + n_warning) / max(n, 1)
        if high_risk_ratio < 0.05:
            logger.warning(
                "高危样本比例仅 %.2f%%，可能导致模型对高危区预测不准，建议增加历史故障数据",
                100 * high_risk_ratio,
            )
