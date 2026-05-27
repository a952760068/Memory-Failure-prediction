"""
rul_validator.py
验证 Python 模型预测结果与 C 侧实现的数值一致性。

使用方式：
  1. Python 侧：对一批测试窗口计算特征向量 → 模型预测 RUL
  2. C 侧（需提供可执行文件 firmware/build/rul_inference_cli）：
     接受相同归一化特征向量，输出 RUL
  3. 比较两侧输出，断言误差 < tolerance

当 C 可执行文件不存在时，仅执行 Python 侧自校验（特征维数/范围检查）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "training" / "src"))

from feature_engineering import HBMFeatureExtractor, N_FEATURES


def validate_feature_dimensions(extractor: HBMFeatureExtractor) -> bool:
    """验证特征提取器输出维数正确。"""
    import pandas as pd
    # 构造一个全零的 30 天窗口
    n = 4320
    df = pd.DataFrame({
        "ce_count": [0] * n,
        "ue_count": [0] * n,
        "temp_c": [70.0] * n,
        "power_w": [30.0] * n,
        "ce_col_bitmap": [None] * n,
    })
    feat = extractor.build_feature_vector(df, hbm_age_days=100.0)
    assert feat.shape == (N_FEATURES,), f"特征维数错误: {feat.shape}"
    assert not np.isnan(feat).any(), "特征包含 NaN"
    assert not np.isinf(feat).any(), "特征包含 Inf"
    print(f"[PASS] 特征维数验证: shape={feat.shape}")
    return True


def validate_against_c_binary(
    c_binary: Path,
    feature_vecs: np.ndarray,
    py_preds: np.ndarray,
    tolerance: float = 0.01,
) -> bool:
    """将归一化特征向量写入 JSON，调用 C 二进制，比较输出。"""
    if not c_binary.exists():
        print(f"[SKIP] C 二进制不存在: {c_binary}，跳过 C 一致性验证")
        return True

    all_passed = True
    for i, (feat_vec, py_rul) in enumerate(zip(feature_vecs, py_preds)):
        # 将特征向量写入临时 JSON
        input_data = {"features": feat_vec.tolist()}
        result = subprocess.run(
            [str(c_binary)],
            input=json.dumps(input_data),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            print(f"[FAIL] 样本 {i}: C 二进制返回错误: {result.stderr}")
            all_passed = False
            continue
        try:
            output = json.loads(result.stdout)
            c_rul = float(output["rul_days"])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[FAIL] 样本 {i}: 解析 C 输出失败: {e}")
            all_passed = False
            continue

        diff = abs(py_rul - c_rul)
        if diff > tolerance:
            print(f"[FAIL] 样本 {i}: Python={py_rul:.4f}  C={c_rul:.4f}  差异={diff:.4f} > {tolerance}")
            all_passed = False
        else:
            print(f"[PASS] 样本 {i}: RUL差异={diff:.6f}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="RUL 预测一致性验证工具")
    parser.add_argument("--model", default="../training/data/processed/rf_model.joblib",
                        help="Python RF 模型路径")
    parser.add_argument("--scaler", default=None, help="StandardScaler 路径（joblib）")
    parser.add_argument("--c-binary", default="../firmware/build/rul_inference_cli",
                        help="C 侧推理 CLI 二进制路径")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="RUL 允许误差（天）")
    args = parser.parse_args()

    # 基本特征维数验证
    extractor = HBMFeatureExtractor()
    ok = validate_feature_dimensions(extractor)
    if not ok:
        sys.exit(1)

    # 若模型存在，执行 C 一致性验证
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[INFO] 模型文件不存在: {model_path}，仅执行特征验证")
        print("所有可用验证通过。")
        return

    import joblib
    model = joblib.load(model_path)
    print(f"[INFO] 加载模型: {model_path}")

    # 生成测试特征
    import pandas as pd
    n = 4320
    test_cases = []
    for ce_level in [0, 50, 200, 500, 1000]:
        df = pd.DataFrame({
            "ce_count": [ce_level] * n,
            "ue_count": [0] * n,
            "temp_c": [70.0] * n,
            "power_w": [30.0] * n,
            "ce_col_bitmap": [None] * n,
        })
        feat = extractor.build_feature_vector(df, hbm_age_days=100.0)
        test_cases.append(feat)

    X = np.array(test_cases)
    scaler_path = Path(args.scaler) if args.scaler else None
    if scaler_path and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_norm = scaler.transform(X)
    else:
        X_norm = X  # 未归一化，C 侧使用相同输入

    py_preds = model.predict(X_norm)
    print(f"[INFO] Python 预测结果: {py_preds}")

    c_binary = Path(args.c_binary)
    all_ok = validate_against_c_binary(c_binary, X_norm, py_preds, args.tolerance)

    if all_ok:
        print("\n[SUCCESS] 所有验证通过！")
    else:
        print("\n[FAIL] 部分验证失败，请检查 C 侧特征提取实现。")
        sys.exit(1)


if __name__ == "__main__":
    main()
