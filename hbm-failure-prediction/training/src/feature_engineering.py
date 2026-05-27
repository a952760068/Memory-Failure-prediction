"""
feature_engineering.py
从 30 天观测窗口中提取 43 维特征向量。

特征顺序与 C 侧 firmware/include/rf_model.h 严格对齐，
顺序改动必须同步修改 feature_config.yaml 和 C 侧代码。
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

N_FEATURES = 43
CE_BURST_THRESHOLD = 100
TEMP_HIGH_1 = 85.0
TEMP_HIGH_2 = 90.0
TEMP_CYCLE_THRESHOLD = 75.0
SAMPLES_PER_HOUR = 6   # 10分钟间隔，每小时6个采样点
SAMPLES_PER_DAY = 144  # 每天 144 个采样点


class HBMFeatureExtractor:
    """从单个 30 天窗口 DataFrame 提取 43 维特征。"""

    def __init__(self, ce_burst_threshold: int = CE_BURST_THRESHOLD):
        self.ce_burst_threshold = ce_burst_threshold
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #

    def build_feature_vector(
        self, window_df: pd.DataFrame, hbm_age_days: float = 0.0
    ) -> np.ndarray:
        """主入口：返回长度 43 的原始特征 numpy 数组。

        Args:
            window_df : 30 天观测窗口 DataFrame（按时间升序排列）
            hbm_age_days: 该 HBM die 已在线的天数
        Returns:
            shape (43,) float32 数组
        """
        ce = window_df["ce_count"].values.astype(float)
        ue = window_df["ue_count"].values.astype(float)
        temp = window_df["temp_c"].values.astype(float) if "temp_c" in window_df.columns else np.zeros(len(window_df))
        power = window_df["power_w"].values.astype(float) if "power_w" in window_df.columns else np.zeros(len(window_df))
        col_bitmap = window_df["ce_col_bitmap"].values if "ce_col_bitmap" in window_df.columns else None

        feat = np.empty(N_FEATURES, dtype=np.float32)
        offset = 0

        # 特征 0-6: ECC CE 系列
        ce_feats = self._extract_ecc_ce_features(ce)
        feat[offset:offset + len(ce_feats)] = ce_feats
        offset += len(ce_feats)

        # 特征 7-12: ECC UE 系列
        ue_feats = self._extract_ecc_ue_features(ue, ce)
        feat[offset:offset + len(ue_feats)] = ue_feats
        offset += len(ue_feats)

        # 特征 13-14: 突发事件
        burst_feats = self._extract_burst_features(ce)
        feat[offset:offset + len(burst_feats)] = burst_feats
        offset += len(burst_feats)

        # 特征 15-17: 列级错误
        col_feats = self._extract_column_features(col_bitmap, len(window_df))
        feat[offset:offset + len(col_feats)] = col_feats
        offset += len(col_feats)

        # 特征 18-26: 温度系列
        temp_feats = self._extract_thermal_features(temp, ce)
        feat[offset:offset + len(temp_feats)] = temp_feats
        offset += len(temp_feats)

        # 特征 27-31: 功耗系列
        power_feats = self._extract_power_features(power, ce)
        feat[offset:offset + len(power_feats)] = power_feats
        offset += len(power_feats)

        # 特征 32-42: 时序特征
        temporal_feats = self._extract_temporal_features(ce, ue, temp, hbm_age_days)
        feat[offset:offset + len(temporal_feats)] = temporal_feats
        offset += len(temporal_feats)

        assert offset == N_FEATURES, f"特征维数错误: {offset} != {N_FEATURES}"
        return feat

    def fit_scaler(self, X: np.ndarray) -> StandardScaler:
        """在训练集上拟合 StandardScaler，保存备用。"""
        self._scaler = StandardScaler()
        self._scaler.fit(X)
        return self._scaler

    def transform(self, X: np.ndarray) -> np.ndarray:
        """使用已拟合的 scaler 做 z-score 归一化。"""
        if self._scaler is None:
            raise RuntimeError("请先调用 fit_scaler()")
        return self._scaler.transform(X).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 特征子组提取（私有）
    # ------------------------------------------------------------------ #

    def _extract_ecc_ce_features(self, ce: np.ndarray) -> np.ndarray:
        """7 维 CE 特征 (索引 0-6)。"""
        n_days = max(len(ce) // SAMPLES_PER_DAY, 1)
        daily_ce = _daily_sum(ce, n_days)

        total = float(ce.sum())
        mean_d = float(daily_ce.mean())
        max_d = float(daily_ce.max())
        std_d = float(daily_ce.std())
        rate_h = total / max(len(ce) / SAMPLES_PER_HOUR, 1)
        slope = _linear_slope(ce)
        # 加速度：CE 趋势的二阶差分均值
        if len(ce) >= 3:
            accel = float(np.diff(np.diff(ce)).mean())
        else:
            accel = 0.0
        return np.array([total, mean_d, max_d, std_d, rate_h, slope, accel], dtype=np.float32)

    def _extract_ecc_ue_features(self, ue: np.ndarray, ce: np.ndarray) -> np.ndarray:
        """6 维 UE 特征 (索引 7-12)。"""
        n_days = max(len(ue) // SAMPLES_PER_DAY, 1)
        daily_ue = _daily_sum(ue, n_days)

        total = float(ue.sum())
        mean_d = float(daily_ue.mean())
        max_d = float(daily_ue.max())
        rate_h = total / max(len(ue) / SAMPLES_PER_HOUR, 1)
        slope = _linear_slope(ue)
        ce_ue_ratio = float(ce.sum()) / max(total, 1.0)
        return np.array([total, mean_d, max_d, rate_h, slope, ce_ue_ratio], dtype=np.float32)

    def _extract_burst_features(self, ce: np.ndarray) -> np.ndarray:
        """2 维突发事件特征 (索引 13-14)。"""
        in_burst = False
        burst_count = 0
        burst_lens: list[int] = []
        cur_len = 0
        for val in ce:
            if val >= self.ce_burst_threshold:
                if not in_burst:
                    in_burst = True
                    burst_count += 1
                    cur_len = 1
                else:
                    cur_len += 1
            else:
                if in_burst:
                    burst_lens.append(cur_len)
                    cur_len = 0
                in_burst = False
        if in_burst:
            burst_lens.append(cur_len)

        mean_dur = float(np.mean(burst_lens)) if burst_lens else 0.0
        # 转换为分钟（每个采样点 = 10 分钟）
        mean_dur_min = mean_dur * 10.0
        return np.array([float(burst_count), mean_dur_min], dtype=np.float32)

    def _extract_column_features(
        self, col_bitmap: Optional[np.ndarray], n_samples: int
    ) -> np.ndarray:
        """3 维列级错误特征 (索引 15-17)。"""
        if col_bitmap is None or not any(isinstance(v, str) for v in col_bitmap if v == v):
            return np.zeros(3, dtype=np.float32)

        col_counts = np.zeros(256, dtype=np.int64)
        valid = 0
        for bm in col_bitmap:
            if not isinstance(bm, str) or len(bm) < 2:
                continue
            try:
                raw = bytes.fromhex(bm.strip())
            except ValueError:
                continue
            valid += 1
            for byte_idx, byte_val in enumerate(raw[:32]):
                for bit in range(8):
                    if byte_val & (1 << bit):
                        col_counts[byte_idx * 8 + bit] += 1

        if valid == 0:
            return np.zeros(3, dtype=np.float32)

        col_rate = col_counts / max(valid, 1)
        max_rate = float(col_rate.max())
        mean_rate = float(col_rate[col_rate > 0].mean()) if col_rate.any() else 0.0
        # 近似 Gini 系数
        concentration = float(_gini(col_counts))
        return np.array([max_rate, mean_rate, concentration], dtype=np.float32)

    def _extract_thermal_features(self, temp: np.ndarray, ce: np.ndarray) -> np.ndarray:
        """9 维温度特征 (索引 18-26)。"""
        if len(temp) == 0:
            return np.zeros(9, dtype=np.float32)

        mean_t = float(temp.mean())
        max_t = float(temp.max())
        min_t = float(temp.min())
        std_t = float(temp.std())
        slope_t = _linear_slope(temp)
        above_85 = float((temp > TEMP_HIGH_1).sum() / SAMPLES_PER_HOUR)
        above_90 = float((temp > TEMP_HIGH_2).sum() / SAMPLES_PER_HOUR)

        # CE 与温度的 Pearson 相关系数
        if temp.std() > 0 and ce.std() > 0:
            corr_tc = float(np.corrcoef(temp, ce)[0, 1])
        else:
            corr_tc = 0.0
        if np.isnan(corr_tc):
            corr_tc = 0.0

        # 热循环次数：跨越 TEMP_CYCLE_THRESHOLD 的次数
        cycles = _count_threshold_crossings(temp, TEMP_CYCLE_THRESHOLD)

        return np.array(
            [mean_t, max_t, min_t, std_t, slope_t, above_85, above_90, corr_tc, float(cycles)],
            dtype=np.float32,
        )

    def _extract_power_features(self, power: np.ndarray, ce: np.ndarray) -> np.ndarray:
        """5 维功耗特征 (索引 27-31)。"""
        if len(power) == 0:
            return np.zeros(5, dtype=np.float32)

        mean_p = float(power.mean())
        max_p = float(power.max())
        std_p = float(power.std())
        slope_p = _linear_slope(power)

        if power.std() > 0 and ce.std() > 0:
            corr_pc = float(np.corrcoef(power, ce)[0, 1])
        else:
            corr_pc = 0.0
        if np.isnan(corr_pc):
            corr_pc = 0.0

        return np.array([mean_p, max_p, std_p, slope_p, corr_pc], dtype=np.float32)

    def _extract_temporal_features(
        self, ce: np.ndarray, ue: np.ndarray, temp: np.ndarray, hbm_age_days: float
    ) -> np.ndarray:
        """11 维时序特征 (索引 32-42)。"""
        n_days = max(len(ce) // SAMPLES_PER_DAY, 1)
        daily_ce = _daily_sum(ce, n_days)
        daily_ue = _daily_sum(ue, n_days)

        # 特征 32: 连续无错误天数（从窗口尾部计算）
        streak = 0
        for d in reversed(daily_ce):
            if d == 0:
                streak += 1
            else:
                break
        error_free_streak = float(streak)

        # 特征 33: 距最后一次 UE 的天数（在窗口内）
        last_ue_day = -1
        for d_idx in range(n_days - 1, -1, -1):
            start = d_idx * SAMPLES_PER_DAY
            end = min(start + SAMPLES_PER_DAY, len(ue))
            if ue[start:end].sum() > 0:
                last_ue_day = d_idx
                break
        days_since_ue = float(n_days - 1 - last_ue_day) if last_ue_day >= 0 else float(n_days)

        # 特征 34: 距最后一次 CE 突发的天数
        last_burst_day = -1
        for d_idx in range(n_days - 1, -1, -1):
            start = d_idx * SAMPLES_PER_DAY
            end = min(start + SAMPLES_PER_DAY, len(ce))
            if ce[start:end].max() >= self.ce_burst_threshold:
                last_burst_day = d_idx
                break
        days_since_burst = float(n_days - 1 - last_burst_day) if last_burst_day >= 0 else float(n_days)

        # 特征 35: 后 15 天 CE 均值 / 前 15 天 CE 均值（趋势比值）
        half = n_days // 2
        first_half_mean = float(daily_ce[:half].mean()) if half > 0 else 0.0
        second_half_mean = float(daily_ce[half:].mean()) if half < n_days else 0.0
        weekly_ce_trend = second_half_mean / max(first_half_mean, 1.0)

        # 特征 36: CE 熵（Shannon entropy，按天分布）
        ce_entropy = float(_shannon_entropy(daily_ce))

        # 特征 37: 窗口内是否出现 UE (0/1)
        ue_flag = float(1 if ue.sum() > 0 else 0)

        # 特征 38: 窗口内 UE ≥ 2 (0/1)
        multi_ue_flag = float(1 if ue.sum() >= 2 else 0)

        # 特征 39: 高温高 CE 共现小时数
        high_temp_mask = temp > TEMP_HIGH_1
        high_ce_mask = ce >= self.ce_burst_threshold
        cooccur_hours = float((high_temp_mask & high_ce_mask).sum() / SAMPLES_PER_HOUR)

        # 特征 40: 最近 7 天 CE 计数
        last7_start = max(0, len(ce) - 7 * SAMPLES_PER_DAY)
        ce_last7 = float(ce[last7_start:].sum())

        # 特征 41: 最近 7 天 CE 均值 / 全窗口 CE 均值
        total_mean = float(ce.mean())
        last7_mean = float(ce[last7_start:].mean()) if last7_start < len(ce) else 0.0
        growth_rate = last7_mean / max(total_mean, 1.0)

        # 特征 42: HBM 在线天数
        age = float(hbm_age_days)

        return np.array(
            [
                error_free_streak, days_since_ue, days_since_burst,
                weekly_ce_trend, ce_entropy, ue_flag, multi_ue_flag,
                cooccur_hours, ce_last7, growth_rate, age,
            ],
            dtype=np.float32,
        )


# ------------------------------------------------------------------ #
# 工具函数
# ------------------------------------------------------------------ #

def _daily_sum(arr: np.ndarray, n_days: int) -> np.ndarray:
    """将采样点数组按天聚合为日总量。"""
    out = np.zeros(n_days, dtype=np.float64)
    for d in range(n_days):
        start = d * SAMPLES_PER_DAY
        end = min(start + SAMPLES_PER_DAY, len(arr))
        out[d] = arr[start:end].sum()
    return out


def _linear_slope(arr: np.ndarray) -> float:
    """用最小二乘法计算时间序列的线性趋势斜率（归一化到每小时变化量）。"""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    # 归一化 x 到小时单位
    x_hours = x / SAMPLES_PER_HOUR
    mean_x = x_hours.mean()
    mean_y = arr.mean()
    denom = ((x_hours - mean_x) ** 2).sum()
    if denom < 1e-10:
        return 0.0
    slope = float(((x_hours - mean_x) * (arr - mean_y)).sum() / denom)
    if np.isnan(slope):
        return 0.0
    return slope


def _count_threshold_crossings(arr: np.ndarray, threshold: float) -> int:
    """计算信号跨越阈值的次数（上升沿 + 下降沿各算一次）。"""
    if len(arr) < 2:
        return 0
    above = arr > threshold
    crossings = int(np.diff(above.astype(int)).sum() != 0)
    return int(np.abs(np.diff(above.astype(int))).sum())


def _gini(arr: np.ndarray) -> float:
    """计算 Gini 系数，衡量分布集中程度（0=均匀，1=极端集中）。

    使用标准 Lorenz 曲线面积法：G = 1 - 2 * (Lorenz 曲线下面积)
    """
    arr = arr.astype(np.float64)
    total = arr.sum()
    if total == 0:
        return 0.0
    n = len(arr)
    arr_sorted = np.sort(arr)
    # Lorenz 曲线面积 = sum((2i - n - 1) * x_i) / (n * total)，i 从 1 到 n
    index = np.arange(1, n + 1, dtype=np.float64)
    lorenz_area = (np.sum((2.0 * index - n - 1.0) * arr_sorted)) / (n * total)
    gini = float(lorenz_area)
    return max(0.0, min(1.0, gini))


def _shannon_entropy(arr: np.ndarray) -> float:
    """计算 Shannon 熵（以 nats 为单位）。"""
    arr = arr.astype(np.float64)
    total = arr.sum()
    if total == 0:
        return 0.0
    p = arr / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))
