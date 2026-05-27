"""
model_evaluator.py
评估 HBM RUL 模型，生成分段指标报告。
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# 告警区间定义（与 model_config.yaml 保持一致）
BINS = [
    (0, 3,   "CRITICAL"),
    (3, 7,   "WARNING"),
    (7, 14,  "WATCH"),
    (14, 31, "NORMAL"),
]


class HBMModelEvaluator:
    """计算并打印 RUL 预测评估指标。"""

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """计算全局和分段指标。

        Returns:
            字典包含：rmse, mae, r2, 以及各区间的 precision/recall/f1
        """
        metrics: Dict[str, float] = {}

        # 全局回归指标
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-10))
        metrics.update({"rmse": rmse, "mae": mae, "r2": r2})

        if verbose:
            logger.info("=" * 50)
            logger.info("全局指标: RMSE=%.3f  MAE=%.3f  R²=%.4f", rmse, mae, r2)

        # 分段分类指标（将 RUL 映射为告警等级后计算）
        y_true_cls = _rul_to_class(y_true)
        y_pred_cls = _rul_to_class(y_pred)

        for level_id, level_name in enumerate(["CRITICAL", "WARNING", "WATCH", "NORMAL"]):
            tp = int(((y_true_cls == level_id) & (y_pred_cls == level_id)).sum())
            fp = int(((y_true_cls != level_id) & (y_pred_cls == level_id)).sum())
            fn = int(((y_true_cls == level_id) & (y_pred_cls != level_id)).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            metrics[f"{level_name}_precision"] = float(precision)
            metrics[f"{level_name}_recall"] = float(recall)
            metrics[f"{level_name}_f1"] = float(f1)
            if verbose:
                logger.info(
                    "  %-8s: Precision=%.3f  Recall=%.3f  F1=%.3f  (TP=%d FP=%d FN=%d)",
                    level_name, precision, recall, f1, tp, fp, fn,
                )

        if verbose:
            logger.info("=" * 50)

        return metrics

    def print_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: list[str],
        top_k: int = 15,
    ) -> None:
        """打印 Top-K 特征重要性。"""
        idx = np.argsort(importances)[::-1]
        logger.info("Top-%d 特征重要性:", top_k)
        for rank, i in enumerate(idx[:top_k]):
            logger.info("  %2d. %-35s %.4f", rank + 1, feature_names[i], importances[i])


def _rul_to_class(rul: np.ndarray) -> np.ndarray:
    """将连续 RUL（天）映射为 0=CRITICAL/1=WARNING/2=WATCH/3=NORMAL。"""
    cls = np.full(len(rul), 3, dtype=int)   # 默认 NORMAL
    cls[rul <= 14] = 2   # WATCH
    cls[rul <= 7] = 1    # WARNING
    cls[rul <= 3] = 0    # CRITICAL
    return cls
