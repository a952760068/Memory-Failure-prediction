"""
test_model_export.py
单元测试：模型导出功能验证。
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model_exporter import ModelExporter, _arr_to_c


class TestModelExporter:
    def _make_tiny_rf(self) -> RandomForestRegressor:
        rng = np.random.default_rng(0)
        X = rng.random((200, 43)).astype(np.float32)
        y = rng.uniform(0, 30, 200).astype(np.float32)
        rf = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=42)
        rf.fit(X, y)
        return rf

    def test_export_rf_header_creates_file(self, tmp_path):
        rf = self._make_tiny_rf()
        exporter = ModelExporter()
        out = exporter.export_rf_header(rf, tmp_path / "rf_model.h")
        assert out.exists()
        content = out.read_text()
        assert "RF_N_TREES" in content
        assert "TREE_0_FEATURE" in content
        assert "RF_TREES" in content

    def test_export_scaler_header(self, tmp_path):
        scaler = StandardScaler()
        X = np.random.default_rng(1).random((50, 43))
        scaler.fit(X)
        exporter = ModelExporter()
        out = exporter.export_scaler_header(scaler, tmp_path / "feature_scaler.h")
        assert out.exists()
        content = out.read_text()
        assert "FEATURE_MEAN" in content
        assert "FEATURE_STD" in content
        assert "SCALER_N_FEATURES 43" in content

    def test_model_hash_stable(self):
        rf = self._make_tiny_rf()
        h1 = ModelExporter._compute_model_hash(rf)
        h2 = ModelExporter._compute_model_hash(rf)
        assert h1 == h2
        assert len(h1) == 16

    def test_arr_to_c_format(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _arr_to_c(arr, "%.2ff")
        assert "1.00f" in result
        assert "2.00f" in result
