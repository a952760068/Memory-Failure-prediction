"""训练入口脚本：加载合成数据，执行完整训练流水线并导出 C 头文件。"""
from __future__ import annotations
import logging
import sys
from pathlib import Path

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import yaml

from data_loader import HBMDataLoader
from preprocessor import HBMPreprocessor
from feature_engineering import HBMFeatureExtractor
from rul_labeler import RULLabeler
from model_trainer import HBMModelTrainer
from model_evaluator import HBMModelEvaluator
from model_exporter import ModelExporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"


def build_dataset_from_raw(csv_paths):
    """从原始 CSV 构建特征矩阵和 RUL 标签。"""
    loader = HBMDataLoader()
    preprocessor = HBMPreprocessor(window_days=30, step_days=1)
    extractor = HBMFeatureExtractor()
    labeler = RULLabeler(rul_cap=30, high_risk_rul=7, high_risk_weight=3.0)

    all_X, all_y, all_w = [], [], []

    for csv_path in csv_paths:
        df_raw = loader.load(csv_path)
        ts_dict = preprocessor.process(df_raw)

        for key, ts in ts_dict.items():
            srv_id, hbm_id = key.split("::")
            failure_events = preprocessor.extract_failure_events(ts)
            windows = preprocessor.sliding_windows(ts)
            labeled = labeler.label_windows(windows, failure_events)

            first_ts = ts.index.min().timestamp()
            for window_df, end_ts, rul, weight in labeled:
                age_days = (end_ts.timestamp() - first_ts) / 86400.0
                feat = extractor.build_feature_vector(window_df.reset_index(), age_days)
                all_X.append(feat)
                all_y.append(rul)
                all_w.append(weight)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    w = np.array(all_w, dtype=np.float32)
    logger.info("数据集构建完成: X=%s, y=%s", X.shape, y.shape)
    return X, y, w, extractor


def main():
    raw_dir = DATA_DIR / "raw"
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("data/raw/ 目录中无 CSV 文件，请先放入真实数据或运行 tools/log_simulator.py 生成数据")
        logger.info("提示: cd tools && python log_simulator.py --out ../training/data/raw/simulated.csv")
        return

    # 1. 构建数据集
    X, y, w, extractor = build_dataset_from_raw(csv_files)

    # 2. 划分训练/测试（时间顺序，不打乱）
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    w_train = w[:split]

    # 3. 特征归一化
    extractor.fit_scaler(X_train)
    X_train_norm = extractor.transform(X_train)
    X_test_norm = extractor.transform(X_test)

    # 4. 交叉验证
    trainer = HBMModelTrainer(CONFIG_DIR / "model_config.yaml")
    cv_results = trainer.cross_validate(X_train_norm, y_train, w_train)

    # 5. 训练最终模型
    trainer.train(X_train_norm, y_train, w_train)

    # 6. 评估测试集
    evaluator = HBMModelEvaluator()
    y_pred = trainer.model.predict(X_test_norm)
    metrics = evaluator.evaluate(y_test, y_pred)
    with open(CONFIG_DIR / "feature_config.yaml") as f:
        feat_names = yaml.safe_load(f)["feature"]["feature_names"]
    evaluator.print_feature_importance(trainer.model.feature_importances_, feat_names)

    # 7. 保存 joblib 模型
    model_dir = DATA_DIR / "processed"
    trainer.save(model_dir)

    # 8. 导出 C 头文件
    exporter = ModelExporter(CONFIG_DIR / "model_config.yaml")
    firmware_include = BASE_DIR.parent / "firmware" / "include"
    model_hash = exporter._compute_model_hash(trainer.model)
    exporter.export_rf_header(trainer.model, firmware_include / "rf_model.h")
    exporter.export_scaler_header(extractor._scaler, firmware_include / "feature_scaler.h", model_hash)

    logger.info("全流程完成！C 头文件已写入 firmware/include/")


if __name__ == "__main__":
    main()
