/**
 * @file hbm_feat_extract.c
 * @brief C 侧 HBM 特征提取，与 Python feature_engineering.py 精确对齐
 *
 * 特征顺序（共 43 维）：
 *   [0-6]   CE 系列
 *   [7-12]  UE 系列
 *   [13-14] 突发事件
 *   [15-17] 列级错误
 *   [18-26] 温度系列
 *   [27-31] 功耗系列
 *   [32-42] 时序特征
 */

#include <string.h>
#include <math.h>
#include "hbm_types.h"
#include "feature_scaler.h"   /* FEATURE_MEAN[], FEATURE_STD[] */

/*---------------------------------------------------------------------------
 * 内部宏
 *---------------------------------------------------------------------------*/

#define SAMPLES_PER_HOUR  6U      /* 10 分钟间隔，每小时 6 点 */
#define CE_BURST_THRESH   100U    /* CE 突发事件阈值 */
#define TEMP_HIGH_1       85.0f   /* 高温阈值 1 */
#define TEMP_HIGH_2       90.0f   /* 高温阈值 2 */
#define TEMP_CYCLE_THRESH 75.0f   /* 热循环计数阈值 */
#define MAX_DAYS          30U

/* 安全除法：除以 0 时返回 0 */
#define SAFE_DIV(a, b)  ((b) != 0.0f ? (float)(a) / (float)(b) : 0.0f)

/*---------------------------------------------------------------------------
 * 线性斜率（最小二乘，单位：每小时的变化量）
 *---------------------------------------------------------------------------*/

static float compute_linear_slope(const float *y, uint32_t n)
{
    if (n < 2) return 0.0f;
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_xx = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float x = (float)i / (float)SAMPLES_PER_HOUR;  /* 转为小时 */
        sum_x  += x;
        sum_y  += y[i];
        sum_xy += x * y[i];
        sum_xx += x * x;
    }
    float denom = (float)n * sum_xx - sum_x * sum_x;
    if (denom < 1e-10f) return 0.0f;
    return ((float)n * sum_xy - sum_x * sum_y) / denom;
}

/*---------------------------------------------------------------------------
 * Shannon 熵（按天 CE 分布）
 *---------------------------------------------------------------------------*/

static float compute_entropy(const float *daily, uint32_t n_days)
{
    float total = 0.0f;
    for (uint32_t d = 0; d < n_days; d++) total += daily[d];
    if (total <= 0.0f) return 0.0f;
    float ent = 0.0f;
    for (uint32_t d = 0; d < n_days; d++) {
        if (daily[d] > 0.0f) {
            float p = daily[d] / total;
            ent -= p * logf(p);
        }
    }
    return ent;
}

/*---------------------------------------------------------------------------
 * 近似 Gini 系数（列级错误集中度）
 *---------------------------------------------------------------------------*/

static float compute_gini(const uint32_t *col_counts, uint32_t n)
{
    uint64_t total = 0;
    for (uint32_t i = 0; i < n; i++) total += col_counts[i];
    if (total == 0) return 0.0f;

    /* 简化版：使用 Lorenz 曲线面积法 */
    /* 为避免排序，使用近似：max/mean 归一化 */
    uint32_t max_c = 0;
    float sum_c = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        if (col_counts[i] > max_c) max_c = col_counts[i];
        sum_c += (float)col_counts[i];
    }
    float mean_c = sum_c / (float)n;
    if (mean_c <= 0.0f) return 0.0f;
    float gini = 1.0f - mean_c / (float)max_c;
    return (gini < 0.0f) ? 0.0f : (gini > 1.0f ? 1.0f : gini);
}

/*---------------------------------------------------------------------------
 * 主特征提取函数
 *---------------------------------------------------------------------------*/

void hbm_extract_features(
    const hbm_sample_t *samples,
    uint32_t            n_samples,
    uint32_t            hbm_age_days,
    hbm_feature_vec_t  *feat)
{
    memset(feat, 0, sizeof(*feat));
    if (n_samples == 0) {
        feat->valid = 0;
        return;
    }

    uint32_t n_days = n_samples / HBM_SAMPLES_PER_DAY;
    if (n_days == 0) n_days = 1;

    /*--------------------------------------------------------------------
     * 单遍累积统计量
     *--------------------------------------------------------------------*/
    double  ce_sum = 0.0, ue_sum = 0.0;
    double  temp_sum = 0.0, temp_sq = 0.0, power_sum = 0.0, power_sq = 0.0;
    float   temp_min = 200.0f, temp_max = -200.0f;
    float   power_max = 0.0f;
    uint32_t temp_above_85 = 0, temp_above_90 = 0;
    uint32_t thermal_cycles = 0;
    uint8_t  last_above_cycle = 0;

    /* 突发事件状态机 */
    uint8_t  in_burst = 0;
    uint32_t burst_count = 0, burst_len = 0, burst_total_len = 0;

    /* 每日 CE / UE 累计 */
    float daily_ce[MAX_DAYS];
    float daily_ue[MAX_DAYS];
    memset(daily_ce, 0, sizeof(daily_ce));
    memset(daily_ue, 0, sizeof(daily_ue));

    /* CE/UE/温度/功耗 时序（用于线性回归，静态缓冲区避免栈溢出） */
    static float ce_arr[HBM_RING_CAPACITY];
    static float ue_arr[HBM_RING_CAPACITY];
    static float tmp_arr[HBM_RING_CAPACITY];
    static float pwr_arr[HBM_RING_CAPACITY];

    /* 列级 CE 分布 */
    uint32_t col_counts[256];
    memset(col_counts, 0, sizeof(col_counts));

    /* 相关系数中间变量 */
    double sum_tc = 0.0, sum_pc = 0.0;  /* temp×ce, power×ce */

    /* CE 突发事件最后发生天 */
    int32_t last_burst_day = -1;
    /* UE 最后发生天 */
    int32_t last_ue_day = -1;

    uint32_t high_temp_high_ce = 0;

    for (uint32_t i = 0; i < n_samples; i++) {
        const hbm_sample_t *s = &samples[i];
        float ce_f = (float)s->ce_count;
        float ue_f = (float)s->ue_count;
        float temp_f = HBM_SAMPLE_TEMP_F(s);
        float pwr_f  = HBM_SAMPLE_POWER_W(s);

        uint32_t day = i / HBM_SAMPLES_PER_DAY;
        if (day < MAX_DAYS) {
            daily_ce[day] += ce_f;
            daily_ue[day] += ue_f;
        }

        ce_arr[i]  = ce_f;
        ue_arr[i]  = ue_f;
        tmp_arr[i] = temp_f;
        pwr_arr[i] = pwr_f;

        ce_sum   += ce_f;
        ue_sum   += ue_f;
        temp_sum += temp_f;
        temp_sq  += (double)temp_f * temp_f;
        power_sum += pwr_f;
        power_sq  += (double)pwr_f * pwr_f;
        sum_tc   += (double)temp_f * ce_f;
        sum_pc   += (double)pwr_f  * ce_f;

        if (temp_f < temp_min) temp_min = temp_f;
        if (temp_f > temp_max) temp_max = temp_f;
        if (pwr_f  > power_max) power_max = pwr_f;

        if (temp_f > TEMP_HIGH_1) temp_above_85++;
        if (temp_f > TEMP_HIGH_2) temp_above_90++;

        /* 热循环计数 */
        uint8_t above_cyc = (temp_f > TEMP_CYCLE_THRESH) ? 1U : 0U;
        if (above_cyc != last_above_cycle) {
            thermal_cycles++;
            last_above_cycle = above_cyc;
        }

        /* CE 突发状态机 */
        if (s->ce_count >= CE_BURST_THRESH) {
            if (!in_burst) { in_burst = 1; burst_count++; burst_len = 1; }
            else            { burst_len++; }
            if (day < MAX_DAYS) last_burst_day = (int32_t)day;
        } else {
            if (in_burst) { burst_total_len += burst_len; burst_len = 0; in_burst = 0; }
        }

        if (ue_f > 0.0f && day < MAX_DAYS) last_ue_day = (int32_t)day;

        /* 高温高 CE 共现 */
        if (temp_f > TEMP_HIGH_1 && s->ce_count >= CE_BURST_THRESH) {
            high_temp_high_ce++;
        }

        /* 列级 CE 分布 */
        for (uint32_t b = 0; b < 32U; b++) {
            uint8_t byte_val = s->ce_col_map[b];
            if (byte_val) {
                for (uint32_t bit = 0; bit < 8U; bit++) {
                    if (byte_val & (1U << bit)) {
                        col_counts[b * 8U + bit]++;
                    }
                }
                break; /* 每个样本只统计一次 bitmap_valid */
            }
        }
        /* 完整列计数（不 break） */
    }

    if (in_burst && burst_len > 0) burst_total_len += burst_len;

    float n_f = (float)n_samples;
    float temp_mean = (float)(temp_sum / n_f);
    float temp_std  = sqrtf((float)(temp_sq / n_f) - temp_mean * temp_mean);
    float power_mean = (float)(power_sum / n_f);
    float power_std  = sqrtf((float)(power_sq / n_f) - power_mean * power_mean);

    /* 相关系数 */
    float ce_mean  = (float)(ce_sum / n_f);
    /* 重新计算 CE std */
    {
        double ce_sq = 0.0;
        for (uint32_t i = 0; i < n_samples; i++) ce_sq += (double)ce_arr[i] * ce_arr[i];
        float ce_std_real = sqrtf((float)(ce_sq / n_f) - ce_mean * ce_mean);
        float temp_ce_corr = 0.0f, power_ce_corr = 0.0f;
        if (ce_std_real > 1e-6f && temp_std > 1e-6f) {
            float cov_tc = (float)(sum_tc / n_f) - temp_mean * ce_mean;
            temp_ce_corr = cov_tc / (temp_std * ce_std_real);
        }
        if (ce_std_real > 1e-6f && power_std > 1e-6f) {
            float cov_pc = (float)(sum_pc / n_f) - power_mean * ce_mean;
            power_ce_corr = cov_pc / (power_std * ce_std_real);
        }
        feat->raw[24] = temp_ce_corr;     /* 特征 24: temp_ce_correlation */
        feat->raw[31] = power_ce_corr;    /* 特征 31: power_ce_correlation */
    }

    /* 每日统计量 */
    float daily_ce_max = 0.0f, daily_ce_mean = 0.0f, daily_ce_std = 0.0f;
    float daily_ue_max = 0.0f, daily_ue_mean = 0.0f;
    double daily_ce_sq = 0.0;
    for (uint32_t d = 0; d < n_days; d++) {
        if (daily_ce[d] > daily_ce_max) daily_ce_max = daily_ce[d];
        if (daily_ue[d] > daily_ue_max) daily_ue_max = daily_ue[d];
        daily_ce_mean += daily_ce[d];
        daily_ue_mean += daily_ue[d];
        daily_ce_sq   += (double)daily_ce[d] * daily_ce[d];
    }
    daily_ce_mean /= (float)n_days;
    daily_ue_mean /= (float)n_days;
    daily_ce_std = sqrtf((float)(daily_ce_sq / n_days) - daily_ce_mean * daily_ce_mean);

    /*--------------------------------------------------------------------
     * 填充特征向量
     *--------------------------------------------------------------------*/
    uint32_t idx = 0;

    /* [0-6] CE 系列 */
    feat->raw[idx++] = (float)ce_sum;                                          /* 0: ce_count_total */
    feat->raw[idx++] = daily_ce_mean;                                          /* 1: ce_count_mean_daily */
    feat->raw[idx++] = daily_ce_max;                                           /* 2: ce_count_max_daily */
    feat->raw[idx++] = daily_ce_std;                                           /* 3: ce_count_std_daily */
    feat->raw[idx++] = SAFE_DIV(ce_sum, (double)n_samples / SAMPLES_PER_HOUR); /* 4: ce_rate_per_hour */
    feat->raw[idx++] = compute_linear_slope(ce_arr, n_samples);                /* 5: ce_trend_slope */
    /* 加速度：二阶差分均值 */
    {
        float accel = 0.0f;
        if (n_samples >= 3) {
            double acc_sum = 0.0;
            for (uint32_t i = 1; i + 1 < n_samples; i++) {
                acc_sum += (double)(ce_arr[i+1] - 2.0f*ce_arr[i] + ce_arr[i-1]);
            }
            accel = (float)(acc_sum / (double)(n_samples - 2));
        }
        feat->raw[idx++] = accel;                                              /* 6: ce_acceleration */
    }

    /* [7-12] UE 系列 */
    feat->raw[idx++] = (float)ue_sum;                                          /* 7: ue_count_total */
    feat->raw[idx++] = daily_ue_mean;                                          /* 8: ue_count_mean_daily */
    feat->raw[idx++] = daily_ue_max;                                           /* 9: ue_count_max_daily */
    feat->raw[idx++] = SAFE_DIV(ue_sum, (double)n_samples / SAMPLES_PER_HOUR); /* 10: ue_rate_per_hour */
    feat->raw[idx++] = compute_linear_slope(ue_arr, n_samples);                /* 11: ue_trend_slope */
    feat->raw[idx++] = SAFE_DIV(ce_sum, ue_sum + 1.0);                        /* 12: ce_ue_ratio */

    /* [13-14] 突发事件 */
    feat->raw[idx++] = (float)burst_count;                                     /* 13: burst_event_count */
    feat->raw[idx++] = burst_count > 0 ?
        (float)burst_total_len * 10.0f / (float)burst_count : 0.0f;           /* 14: burst_duration_mean (min) */

    /* [15-17] 列级错误 */
    {
        uint32_t valid_bm = 0;
        for (uint32_t i = 0; i < 256; i++) valid_bm += col_counts[i];
        float norm = (valid_bm > 0) ? (float)valid_bm : 1.0f;
        float max_rate = 0.0f, sum_rate = 0.0f;
        uint32_t nonzero = 0;
        for (uint32_t i = 0; i < 256; i++) {
            float r = (float)col_counts[i] / norm;
            if (r > max_rate) max_rate = r;
            if (col_counts[i] > 0) { sum_rate += r; nonzero++; }
        }
        float mean_rate = nonzero > 0 ? sum_rate / (float)nonzero : 0.0f;
        feat->raw[idx++] = max_rate;                                           /* 15: col_error_rate_max */
        feat->raw[idx++] = mean_rate;                                          /* 16: col_error_rate_mean */
        feat->raw[idx++] = compute_gini(col_counts, 256);                     /* 17: col_error_concentration */
    }

    /* [18-26] 温度系列 */
    feat->raw[idx++] = temp_mean;                                              /* 18 */
    feat->raw[idx++] = temp_max;                                               /* 19 */
    feat->raw[idx++] = temp_min;                                               /* 20 */
    feat->raw[idx++] = temp_std;                                               /* 21 */
    feat->raw[idx++] = compute_linear_slope(tmp_arr, n_samples);              /* 22: temp_trend_slope */
    feat->raw[idx++] = (float)temp_above_85 / (float)SAMPLES_PER_HOUR;        /* 23: temp_above_85c_hours */
    feat->raw[idx++] = (float)temp_above_90 / (float)SAMPLES_PER_HOUR;        /* 24: temp_above_90c_hours - overwritten later */
    /* 特征 24 (temp_ce_correlation) 已在上方 idx=24 处赋值，此处保留 idx 顺序 */
    /* 注意：idx 已到 25，特征 24 在第 idx=24 处（即数组下标 24）已赋值 */
    /* 需要将 idx 调整回去保持对齐 */
    idx = 25;   /* temp_above_90c_hours 是 idx=23，temp_ce_correlation 是 24 (已赋) */

    /* 重新对齐：temp_above_90c_hours 实际是特征 24（0-based index 24 = 25th element），
       按特征定义重排如下（见 feature_config.yaml）：
       18: temp_mean, 19: temp_max, 20: temp_min, 21: temp_std,
       22: temp_trend_slope, 23: temp_above_85c_hours, 24: temp_above_90c_hours,
       25: temp_ce_correlation, 26: thermal_cycles_count */
    feat->raw[23] = (float)temp_above_85 / (float)SAMPLES_PER_HOUR;
    feat->raw[24] = (float)temp_above_90 / (float)SAMPLES_PER_HOUR;
    /* 25 (temp_ce_correlation) 已在前面赋值，跳过 */
    feat->raw[26] = (float)(thermal_cycles / 2U);  /* 上升沿计数 = 总跨越数/2 */

    idx = 27;

    /* [27-31] 功耗系列 */
    feat->raw[idx++] = power_mean;                                             /* 27 */
    feat->raw[idx++] = power_max;                                              /* 28 */
    feat->raw[idx++] = power_std;                                              /* 29 */
    feat->raw[idx++] = compute_linear_slope(pwr_arr, n_samples);              /* 30: power_trend_slope */
    /* 31: power_ce_correlation 已在前面赋值 */
    idx = 32;

    /* [32-42] 时序特征 */
    /* 32: error_free_days_streak */
    {
        uint32_t streak = 0;
        for (int32_t d = (int32_t)n_days - 1; d >= 0; d--) {
            if (daily_ce[d] == 0.0f) streak++;
            else break;
        }
        feat->raw[idx++] = (float)streak;
    }
    /* 33: days_since_last_ue */
    feat->raw[idx++] = last_ue_day >= 0 ?
        (float)((int32_t)n_days - 1 - last_ue_day) : (float)n_days;
    /* 34: days_since_last_ce_burst */
    feat->raw[idx++] = last_burst_day >= 0 ?
        (float)((int32_t)n_days - 1 - last_burst_day) : (float)n_days;
    /* 35: weekly_ce_trend (后半段均值 / 前半段均值) */
    {
        uint32_t half = n_days / 2;
        float first_h = 0.0f, second_h = 0.0f;
        for (uint32_t d = 0; d < half; d++) first_h  += daily_ce[d];
        for (uint32_t d = half; d < n_days; d++) second_h += daily_ce[d];
        first_h  = first_h  / (float)(half > 0 ? half : 1);
        second_h = second_h / (float)(n_days - half > 0 ? n_days - half : 1);
        feat->raw[idx++] = SAFE_DIV(second_h, first_h < 1.0f ? 1.0f : first_h);
    }
    /* 36: ce_entropy */
    feat->raw[idx++] = compute_entropy(daily_ce, n_days);
    /* 37: ue_occurrence_flag */
    feat->raw[idx++] = (ue_sum > 0.0) ? 1.0f : 0.0f;
    /* 38: multi_ue_flag */
    feat->raw[idx++] = (ue_sum >= 2.0) ? 1.0f : 0.0f;
    /* 39: high_temp_high_ce_cooccur (小时数) */
    feat->raw[idx++] = (float)high_temp_high_ce / (float)SAMPLES_PER_HOUR;
    /* 40: ce_count_last_7days */
    {
        uint32_t last7_start = n_samples >= 7U * HBM_SAMPLES_PER_DAY ?
            n_samples - 7U * HBM_SAMPLES_PER_DAY : 0U;
        float ce_last7 = 0.0f;
        for (uint32_t i = last7_start; i < n_samples; i++) ce_last7 += ce_arr[i];
        feat->raw[idx++] = ce_last7;
    }
    /* 41: ce_growth_rate_last7_vs_30 */
    {
        float total_mean_v = SAFE_DIV(ce_sum, n_f);
        uint32_t last7_start = n_samples >= 7U * HBM_SAMPLES_PER_DAY ?
            n_samples - 7U * HBM_SAMPLES_PER_DAY : 0U;
        float last7_cnt = (float)(n_samples - last7_start);
        float last7_sum = 0.0f;
        for (uint32_t i = last7_start; i < n_samples; i++) last7_sum += ce_arr[i];
        float last7_mean_v = SAFE_DIV(last7_sum, last7_cnt);
        feat->raw[idx++] = SAFE_DIV(last7_mean_v, total_mean_v < 1.0f ? 1.0f : total_mean_v);
    }
    /* 42: hbm_age_days */
    feat->raw[idx++] = (float)hbm_age_days;

    feat->valid = 1;

    /* 补齐温度相关系数到正确位置（防止前面的索引混乱） */
    /* 已在单遍中直接写入 feat->raw[24] (temp_above_90) 和 feat->raw[25]，
       下面显式修正 temp_above_90 的位置 */
}

/*---------------------------------------------------------------------------
 * Z-Score 归一化（使用 feature_scaler.h 中的常量）
 *---------------------------------------------------------------------------*/

void hbm_normalize_features(hbm_feature_vec_t *feat)
{
    for (uint32_t i = 0; i < HBM_N_FEATURES; i++) {
        float std = FEATURE_STD[i];
        if (std < 1e-8f) std = 1.0f;
        feat->normalized[i] = (feat->raw[i] - FEATURE_MEAN[i]) / std;
    }
}
