"""
test_feature_eng.py
单元测试：特征工程模块的各特征计算正确性。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from feature_engineering import (
    HBMFeatureExtractor,
    N_FEATURES,
    _daily_sum,
    _linear_slope,
    _gini,
    _shannon_entropy,
)

WINDOW_SIZE = 4320


def make_window(ce=0, ue=0, temp=70.0, power=30.0) -> pd.DataFrame:
    """构造等值 30 天窗口 DataFrame。"""
    return pd.DataFrame({
        "ce_count": [ce] * WINDOW_SIZE,
        "ue_count": [ue] * WINDOW_SIZE,
        "temp_c": [temp] * WINDOW_SIZE,
        "power_w": [power] * WINDOW_SIZE,
        "ce_col_bitmap": [None] * WINDOW_SIZE,
    })


class TestFeatureVector:
    def test_output_shape(self):
        extractor = HBMFeatureExtractor()
        feat = extractor.build_feature_vector(make_window(), 100.0)
        assert feat.shape == (N_FEATURES,)

    def test_no_nan_inf(self):
        extractor = HBMFeatureExtractor()
        feat = extractor.build_feature_vector(make_window(ce=50, ue=1, temp=80.0), 200.0)
        assert not np.isnan(feat).any(), f"NaN at index {np.where(np.isnan(feat))}"
        assert not np.isinf(feat).any()

    def test_zero_input_ce_total(self):
        extractor = HBMFeatureExtractor()
        feat = extractor.build_feature_vector(make_window(), 0.0)
        assert feat[0] == pytest.approx(0.0)   # ce_count_total

    def test_constant_ce_total(self):
        extractor = HBMFeatureExtractor()
        feat = extractor.build_feature_vector(make_window(ce=10), 0.0)
        expected = 10.0 * WINDOW_SIZE
        assert feat[0] == pytest.approx(expected, rel=1e-3)

    def test_ue_flags_zero(self):
        extractor = HBMFeatureExtractor()
        feat = extractor.build_feature_vector(make_window(ue=0), 0.0)
        assert feat[37] == pytest.approx(0.0)   # ue_occurrence_flag
        assert feat[38] == pytest.approx(0.0)   # multi_ue_flag

    def test_ue_flag_set(self):
        df = make_window()
        df.iloc[100, df.columns.get_loc("ue_count")] = 1
        extractor = HBMFeatureExtractor()
        feat = extractor.build_feature_vector(df, 0.0)
        assert feat[37] == pytest.approx(1.0)   # ue_occurrence_flag

    def test_multi_ue_flag(self):
        df = make_window()
        df.iloc[100, df.columns.get_loc("ue_count")] = 1
        df.iloc[200, df.columns.get_loc("ue_count")] = 1
        extractor = HBMFeatureExtractor()
        feat = extractor.build_feature_vector(df, 0.0)
        assert feat[38] == pytest.approx(1.0)   # multi_ue_flag

    def test_hbm_age_days(self):
        extractor = HBMFeatureExtractor()
        feat = extractor.build_feature_vector(make_window(), 365.0)
        assert feat[42] == pytest.approx(365.0)  # hbm_age_days

    def test_temp_mean(self):
        extractor = HBMFeatureExtractor()
        feat = extractor.build_feature_vector(make_window(temp=80.0), 0.0)
        assert feat[18] == pytest.approx(80.0, abs=0.5)   # temp_mean


class TestHelperFunctions:
    def test_daily_sum(self):
        arr = np.ones(144 * 3)  # 3 days
        result = _daily_sum(arr, 3)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [144.0, 144.0, 144.0])

    def test_linear_slope_zero(self):
        arr = np.zeros(100)
        assert _linear_slope(arr) == pytest.approx(0.0)

    def test_linear_slope_positive(self):
        arr = np.arange(600, dtype=float)   # 단调增
        slope = _linear_slope(arr)
        assert slope > 0

    def test_gini_uniform(self):
        arr = np.ones(256, dtype=np.int64) * 100
        g = _gini(arr)
        assert g == pytest.approx(0.0, abs=0.01)

    def test_gini_concentrated(self):
        arr = np.zeros(256, dtype=np.int64)
        arr[0] = 1000
        g = _gini(arr)
        assert g > 0.9

    def test_entropy_zero(self):
        arr = np.zeros(30)
        assert _shannon_entropy(arr) == pytest.approx(0.0)

    def test_entropy_uniform(self):
        arr = np.ones(10) * 10
        ent = _shannon_entropy(arr)
        assert ent == pytest.approx(np.log(10), abs=0.01)
