"""
model_exporter.py
将 sklearn RandomForestRegressor 导出为 BMC C 头文件。

导出格式：
  rf_model.h       - 每棵树的 feature/threshold/left/right/value 数组
  feature_scaler.h - StandardScaler 的 mean 和 std 常量数组
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_HEADER_BANNER = """\
/*
 * {filename}
 * AUTO-GENERATED FILE — DO NOT EDIT MANUALLY
 * Model Hash : {model_hash}
 * Trained    : {trained_at}
 * Trees      : {n_trees}
 * Features   : {n_features}
 * Max Depth  : {max_depth}
 *
 * Deploy to  : BMC firmware (AST2600 / ARM Cortex-A7)
 * Include in : firmware/src/rf_predict.c
 */
"""


class ModelExporter:
    """将训练好的 RF 模型序列化为 C 头文件。"""

    def __init__(self, config_path: Optional[str | Path] = None):
        self.max_depth = 12
        if config_path is not None:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            self.max_depth = cfg["random_forest"].get("max_depth", 12)

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #

    def export_rf_header(
        self,
        model: RandomForestRegressor,
        output_path: str | Path,
    ) -> Path:
        """将 RF 模型导出为 rf_model.h。

        Args:
            model      : 已训练的 RandomForestRegressor
            output_path: 输出路径（含文件名）

        Returns:
            实际写入的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_trees = len(model.estimators_)
        n_features = model.n_features_in_

        model_hash = self._compute_model_hash(model)
        trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = []
        lines.append(
            _HEADER_BANNER.format(
                filename=output_path.name,
                model_hash=model_hash,
                trained_at=trained_at,
                n_trees=n_trees,
                n_features=n_features,
                max_depth=self.max_depth,
            )
        )
        lines.append("#ifndef RF_MODEL_H\n#define RF_MODEL_H\n")
        lines.append("#include <stdint.h>\n")
        lines.append(f"#define RF_N_TREES     {n_trees}\n")
        lines.append(f"#define RF_N_FEATURES  {n_features}\n")
        lines.append(f"#define RF_MAX_DEPTH   {self.max_depth}\n\n")

        # 每棵树的节点数组
        total_nodes = 0
        for tree_idx, estimator in enumerate(model.estimators_):
            tree = estimator.tree_
            n_nodes = tree.node_count
            total_nodes += n_nodes

            feature   = tree.feature          # int array
            threshold = tree.threshold         # float array
            left      = tree.children_left    # int array
            right     = tree.children_right   # int array
            # value shape: (n_nodes, n_outputs, max_n_classes)
            # 回归树: value[:, 0, 0]
            value = tree.value[:, 0, 0]

            lines.append(f"/* Tree {tree_idx} ({n_nodes} nodes) */\n")
            lines.append(f"static const int16_t  TREE_{tree_idx}_FEATURE[{n_nodes}]    = {{{_arr_to_c(feature, '%d')}}};\n")
            lines.append(f"static const float    TREE_{tree_idx}_THRESHOLD[{n_nodes}]  = {{{_arr_to_c(threshold, '%.6ff')}}};\n")
            lines.append(f"static const int16_t  TREE_{tree_idx}_LEFT[{n_nodes}]       = {{{_arr_to_c(left, '%d')}}};\n")
            lines.append(f"static const int16_t  TREE_{tree_idx}_RIGHT[{n_nodes}]      = {{{_arr_to_c(right, '%d')}}};\n")
            lines.append(f"static const float    TREE_{tree_idx}_VALUE[{n_nodes}]      = {{{_arr_to_c(value, '%.4ff')}}};\n\n")

        # 树大小表
        lines.append("/* Tree metadata table */\n")
        lines.append(f"static const int RF_TREE_SIZES[{n_trees}] = {{{', '.join(str(e.tree_.node_count) for e in model.estimators_)}}};\n\n")

        # RF_TREES 描述符数组（供 rf_predict.c 遍历所有树）
        lines.append("typedef struct {\n")
        lines.append("    const int16_t *feature;\n")
        lines.append("    const float   *threshold;\n")
        lines.append("    const int16_t *left;\n")
        lines.append("    const int16_t *right;\n")
        lines.append("    const float   *value;\n")
        lines.append("    int            n_nodes;\n")
        lines.append("} rf_tree_desc_t;\n\n")
        tree_entries = []
        for t in range(n_trees):
            n_nodes = model.estimators_[t].tree_.node_count
            tree_entries.append(
                f"    {{ TREE_{t}_FEATURE, TREE_{t}_THRESHOLD, TREE_{t}_LEFT, "
                f"TREE_{t}_RIGHT, TREE_{t}_VALUE, {n_nodes} }}"
            )
        lines.append(f"static const rf_tree_desc_t RF_TREES[{n_trees}] = {{\n")
        lines.append(",\n".join(tree_entries))
        lines.append("\n};\n\n")
        lines.append("#endif /* RF_MODEL_H */\n")

        output_path.write_text("".join(lines), encoding="utf-8")
        file_size_kb = output_path.stat().st_size / 1024
        logger.info(
            "rf_model.h 导出完成: %s  (%.1f KB, %d 树, %d 总节点)",
            output_path, file_size_kb, n_trees, total_nodes,
        )
        if file_size_kb > 1024:
            logger.warning("文件大小 %.1f KB 超过 1MB，可能超出 BMC Flash 预算", file_size_kb)
        return output_path

    def export_scaler_header(
        self,
        scaler: StandardScaler,
        output_path: str | Path,
        model_hash: str = "unknown",
    ) -> Path:
        """将 StandardScaler 导出为 feature_scaler.h。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mean = scaler.mean_.astype(np.float32)
        std = scaler.scale_.astype(np.float32)
        n = len(mean)
        trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        content = _HEADER_BANNER.format(
            filename=output_path.name,
            model_hash=model_hash,
            trained_at=trained_at,
            n_trees="N/A",
            n_features=n,
            max_depth="N/A",
        )
        content += "#ifndef FEATURE_SCALER_H\n#define FEATURE_SCALER_H\n\n"
        content += f"#define SCALER_N_FEATURES {n}\n\n"
        content += f"static const float FEATURE_MEAN[{n}] = {{{_arr_to_c(mean, '%.8ff')}}};\n\n"
        content += f"static const float FEATURE_STD[{n}]  = {{{_arr_to_c(std,  '%.8ff')}}};\n\n"
        content += "#endif /* FEATURE_SCALER_H */\n"

        output_path.write_text(content, encoding="utf-8")
        logger.info("feature_scaler.h 导出完成: %s", output_path)
        return output_path

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_model_hash(model: RandomForestRegressor) -> str:
        """基于第一棵树的阈值计算模型指纹（用于版本校验）。"""
        h = hashlib.sha256()
        for est in model.estimators_[:10]:  # 取前10棵树做hash
            h.update(est.tree_.threshold.tobytes())
        return h.hexdigest()[:16]


def _arr_to_c(arr: np.ndarray, fmt: str) -> str:
    """将 numpy 数组格式化为 C 初始化列表字符串。"""
    formatted = [fmt % v for v in arr]
    # 每行最多 8 个元素，提升可读性
    lines = []
    for i in range(0, len(formatted), 8):
        lines.append(", ".join(formatted[i:i + 8]))
    return ",\n    ".join(lines)
