"""
model_trainer.py
随机森林 RUL 模型训练，支持时间序列交叉验证。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class HBMModelTrainer:
    """训练 HBM RUL 随机森林模型。"""

    def __init__(self, config_path: str | Path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.rf_cfg = cfg["random_forest"]
        self.train_cfg = cfg["training"]
        self.threshold_cfg = cfg["thresholds"]
        self.model: Optional[RandomForestRegressor] = None

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> RandomForestRegressor:
        """使用全量数据训练最终模型。

        Args:
            X: 特征矩阵，shape (n_samples, 43)
            y: RUL 标签，shape (n_samples,)
            sample_weights: 样本权重，shape (n_samples,)，可为 None

        Returns:
            训练好的 RandomForestRegressor
        """
        logger.info("开始训练 RandomForest: n_samples=%d, n_features=%d", len(y), X.shape[1])
        self.model = RandomForestRegressor(
            n_estimators=self.rf_cfg["n_estimators"],
            max_depth=self.rf_cfg["max_depth"],
            min_samples_split=self.rf_cfg["min_samples_split"],
            min_samples_leaf=self.rf_cfg["min_samples_leaf"],
            max_features=self.rf_cfg["max_features"],
            n_jobs=self.rf_cfg["n_jobs"],
            random_state=self.rf_cfg["random_state"],
        )
        self.model.fit(X, y, sample_weight=sample_weights)
        logger.info("训练完成")
        return self.model

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """时间序列 K 折交叉验证（按顺序切分，不打乱，防止时间泄露）。

        Returns:
            包含 rmse_mean, mae_mean, recall_critical_mean 等指标的字典
        """
        n_splits = self.train_cfg["cv_folds"]
        kf = KFold(n_splits=n_splits, shuffle=False)

        rmse_list, mae_list, recall_list = [], [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            w_tr = sample_weights[train_idx] if sample_weights is not None else None

            rf = RandomForestRegressor(
                n_estimators=self.rf_cfg["n_estimators"],
                max_depth=self.rf_cfg["max_depth"],
                min_samples_split=self.rf_cfg["min_samples_split"],
                min_samples_leaf=self.rf_cfg["min_samples_leaf"],
                max_features=self.rf_cfg["max_features"],
                n_jobs=self.rf_cfg["n_jobs"],
                random_state=self.rf_cfg["random_state"],
            )
            rf.fit(X_tr, y_tr, sample_weight=w_tr)
            y_pred = rf.predict(X_val)

            rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
            mae = float(np.mean(np.abs(y_pred - y_val)))
            # 高危区召回：RUL 真实 ≤ 3 天时，预测也 ≤ 3 天的比例
            critical_mask = y_val <= self.threshold_cfg["rul_critical"]
            if critical_mask.sum() > 0:
                recall_crit = float(
                    (y_pred[critical_mask] <= self.threshold_cfg["rul_critical"]).mean()
                )
            else:
                recall_crit = float("nan")

            logger.info(
                "Fold %d/%d: RMSE=%.3f  MAE=%.3f  Recall_Critical=%.3f",
                fold + 1, n_splits, rmse, mae, recall_crit if not np.isnan(recall_crit) else -1,
            )
            rmse_list.append(rmse)
            mae_list.append(mae)
            if not np.isnan(recall_crit):
                recall_list.append(recall_crit)

        result = {
            "rmse_mean": float(np.mean(rmse_list)),
            "rmse_std": float(np.std(rmse_list)),
            "mae_mean": float(np.mean(mae_list)),
            "mae_std": float(np.std(mae_list)),
            "recall_critical_mean": float(np.mean(recall_list)) if recall_list else float("nan"),
        }

        logger.info("交叉验证结果: %s", result)

        # 检查召回率目标
        target = self.train_cfg["target_recall_critical"]
        if not np.isnan(result["recall_critical_mean"]) and result["recall_critical_mean"] < target:
            logger.warning(
                "高危区召回率 %.3f 未达到目标 %.3f，建议调整样本权重或超参数",
                result["recall_critical_mean"], target,
            )
        return result

    def save(self, output_dir: str | Path) -> Path:
        """保存模型到 joblib 文件。"""
        if self.model is None:
            raise RuntimeError("模型尚未训练")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "rf_model.joblib"
        joblib.dump(self.model, model_path)
        logger.info("模型已保存: %s", model_path)
        return model_path

    @classmethod
    def load(cls, model_path: str | Path) -> RandomForestRegressor:
        """加载已保存的模型。"""
        return joblib.load(model_path)
