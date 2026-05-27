/**
 * @file test_rf_predict.c
 * @brief RF 推理单元测试（使用桩模型）
 *
 * 编译时需加 -DRF_MODEL_STUB
 */
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "hbm_types.h"

extern void rf_predict(const hbm_feature_vec_t *feat, float *rul_out, float *confidence);

static void test_low_ce_gives_normal_rul(void)
{
    hbm_feature_vec_t feat;
    /* 桩树：ce_count_total（特征0）> 500 → RIGHT→节点2 → RUL=20（正常）
       sklearn 语义：feat <= threshold → LEFT，feat > threshold → RIGHT */
    for (uint32_t i = 0; i < HBM_N_FEATURES; i++) {
        feat.normalized[i] = 0.0f;
    }
    feat.normalized[0] = 600.0f;  /* 600 > 500 → RIGHT → RUL=20 */
    feat.valid = 1;

    float rul, conf;
    rf_predict(&feat, &rul, &conf);
    assert(fabsf(rul - 20.0f) < 0.01f);
    printf("[PASS] test_low_ce_gives_normal_rul (RUL=%.2f, conf=%.2f)\n", rul, conf);
}

static void test_high_ce_gives_critical_rul(void)
{
    hbm_feature_vec_t feat;
    /* 桩树：ce_count_total <= 500 → LEFT→节点1 → RUL=5（高危） */
    for (uint32_t i = 0; i < HBM_N_FEATURES; i++) {
        feat.normalized[i] = 0.0f;
    }
    feat.normalized[0] = 100.0f;  /* 100 <= 500 → LEFT → RUL=5 */
    feat.valid = 1;

    float rul, conf;
    rf_predict(&feat, &rul, &conf);
    assert(fabsf(rul - 5.0f) < 0.01f);
    printf("[PASS] test_high_ce_gives_critical_rul (RUL=%.2f)\n", rul);
}

static void test_rul_clamped_to_range(void)
{
    hbm_feature_vec_t feat;
    for (uint32_t i = 0; i < HBM_N_FEATURES; i++) feat.normalized[i] = 0.0f;
    feat.valid = 1;

    float rul, conf;
    rf_predict(&feat, &rul, &conf);
    assert(rul >= 0.0f && rul <= 30.0f);
    assert(conf >= 0.0f && conf <= 1.0f);
    printf("[PASS] test_rul_clamped_to_range\n");
}

int main(void)
{
    printf("=== RF Predict Unit Tests (stub model) ===\n");
    test_low_ce_gives_normal_rul();
    test_high_ce_gives_critical_rul();
    test_rul_clamped_to_range();
    printf("All RF predict tests PASSED.\n");
    return 0;
}
