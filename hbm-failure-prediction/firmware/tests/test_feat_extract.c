/**
 * @file test_feat_extract.c
 * @brief 特征提取单元测试：验证边界条件和已知输入的特征值
 */
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "hbm_types.h"

extern void hbm_extract_features(const hbm_sample_t *samples, uint32_t n_samples,
                                   uint32_t hbm_age_days, hbm_feature_vec_t *feat);
extern void hbm_normalize_features(hbm_feature_vec_t *feat);

static hbm_sample_t g_samples[HBM_RING_CAPACITY];

static void fill_samples_constant(hbm_sample_t *s, uint32_t n,
                                   uint16_t ce, uint8_t ue,
                                   int8_t temp_c, uint16_t power_mw)
{
    for (uint32_t i = 0; i < n; i++) {
        memset(&s[i], 0, sizeof(s[i]));
        s[i].ce_count  = ce;
        s[i].ue_count  = ue;
        s[i].temp_c    = temp_c;
        s[i].power_mw  = power_mw;
        s[i].flags     = HBM_SAMPLE_FLAG_VALID;
    }
}

static void test_zero_input(void)
{
    memset(g_samples, 0, sizeof(g_samples));
    hbm_feature_vec_t feat;
    hbm_extract_features(g_samples, HBM_RING_CAPACITY, 100, &feat);
    assert(feat.valid == 1);
    /* ce_count_total 应为 0 */
    assert(fabsf(feat.raw[0]) < 1e-4f);
    /* ue_occurrence_flag 应为 0 */
    assert(fabsf(feat.raw[37]) < 1e-4f);
    printf("[PASS] test_zero_input\n");
}

static void test_constant_ce(void)
{
    uint32_t n = HBM_RING_CAPACITY;
    fill_samples_constant(g_samples, n, 10, 0, 70, 30000);

    hbm_feature_vec_t feat;
    hbm_extract_features(g_samples, n, 365, &feat);
    assert(feat.valid == 1);

    /* ce_count_total = 10 * 4320 = 43200 */
    float expected_total = 10.0f * (float)n;
    assert(fabsf(feat.raw[0] - expected_total) < 1.0f);

    /* temp_mean 应约等于 70 */
    assert(fabsf(feat.raw[18] - 70.0f) < 0.1f);

    /* ue_occurrence_flag = 0 */
    assert(fabsf(feat.raw[37]) < 1e-4f);

    /* hbm_age_days = 365 */
    assert(fabsf(feat.raw[42] - 365.0f) < 1e-4f);
    printf("[PASS] test_constant_ce (total=%.0f, temp=%.1f)\n",
           feat.raw[0], feat.raw[18]);
}

static void test_ue_flags(void)
{
    uint32_t n = HBM_RING_CAPACITY;
    fill_samples_constant(g_samples, n, 5, 0, 65, 25000);
    /* 注入 3 个 UE */
    g_samples[100].ue_count = 1;
    g_samples[200].ue_count = 1;
    g_samples[300].ue_count = 1;

    hbm_feature_vec_t feat;
    hbm_extract_features(g_samples, n, 50, &feat);

    /* ue_occurrence_flag = 1 */
    assert(fabsf(feat.raw[37] - 1.0f) < 1e-4f);
    /* multi_ue_flag = 1 */
    assert(fabsf(feat.raw[38] - 1.0f) < 1e-4f);
    printf("[PASS] test_ue_flags\n");
}

static void test_normalize_no_crash(void)
{
    uint32_t n = HBM_RING_CAPACITY;
    fill_samples_constant(g_samples, n, 50, 0, 80, 40000);
    hbm_feature_vec_t feat;
    hbm_extract_features(g_samples, n, 200, &feat);
    hbm_normalize_features(&feat);
    /* 归一化后不应出现 NaN */
    for (uint32_t i = 0; i < HBM_N_FEATURES; i++) {
        assert(feat.normalized[i] == feat.normalized[i]);  /* NaN != NaN */
    }
    printf("[PASS] test_normalize_no_crash\n");
}

int main(void)
{
    printf("=== Feature Extract Unit Tests ===\n");
    test_zero_input();
    test_constant_ce();
    test_ue_flags();
    test_normalize_no_crash();
    printf("All feature extract tests PASSED.\n");
    return 0;
}
