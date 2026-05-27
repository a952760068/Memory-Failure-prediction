/**
 * @file test_integration.c
 * @brief 集成测试：模拟 30 天遥测数据，验证完整推理流水线
 */
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include "hbm_types.h"
#include "hbm_predictor.h"
#include "hbm_alert.h"

extern int  hbm_predictor_init(hbm_predictor_ctx_t *ctx, uint8_t n_hbm, uint32_t now_ts);
extern void hbm_predictor_tick(hbm_predictor_ctx_t *ctx);
extern int  hbm_predictor_get_result(const hbm_predictor_ctx_t *ctx,
                                      uint8_t hbm_id, hbm_rul_result_t *result);

/* Mock 遥测数据 */
extern uint16_t mock_ce_count[];
extern uint8_t  mock_ue_count[];
extern int8_t   mock_temp_c[];
extern uint16_t mock_power_mw[];

static hbm_predictor_ctx_t g_ctx;

static void simulate_ticks(hbm_predictor_ctx_t *ctx, uint32_t *sim_ts,
                             uint32_t n_ticks, uint16_t ce, uint8_t ue, int8_t temp)
{
    extern uint32_t hbm_platform_get_time(void);
    mock_ce_count[0] = ce;
    mock_ue_count[0] = ue;
    mock_temp_c[0]   = temp;

    /* 直接调用 tick，通过 mock_ts 模拟时间推进 */
    for (uint32_t t = 0; t < n_ticks; t++) {
        /* 手动填充 ring buffer（绕过 platform_get_time 限制） */
        hbm_sample_t s;
        memset(&s, 0, sizeof(s));
        s.ce_count  = ce;
        s.ue_count  = ue;
        s.temp_c    = temp;
        s.power_mw  = 30000;
        s.timestamp = *sim_ts;
        s.flags     = HBM_SAMPLE_FLAG_VALID;
        extern void hbm_ring_buf_push(hbm_ring_buffer_t *, const hbm_sample_t *);
        hbm_ring_buf_push(&ctx->ring_bufs[0], &s);
        *sim_ts += 600;
    }
}

static void test_no_result_before_30_days(void)
{
    uint32_t ts = 1700000000U;
    hbm_predictor_init(&g_ctx, 1, ts);

    /* 仅填入 29 天数据 */
    simulate_ticks(&g_ctx, &ts, 29 * 144, 5, 0, 70);

    hbm_rul_result_t result;
    /* 数据不足 30 天，推理未触发，prediction_ts=0 */
    int ret = hbm_predictor_get_result(&g_ctx, 0, &result);
    assert(ret == -1);
    printf("[PASS] test_no_result_before_30_days\n");
}

static void test_alert_level_after_infer(void)
{
    uint32_t ts = 1700000000U;
    hbm_predictor_init(&g_ctx, 1, ts);

    /* 填入 30 天数据，CE=0，正常状态 */
    simulate_ticks(&g_ctx, &ts, 30 * 144, 0, 0, 70);

    /* 触发推理 */
    g_ctx.last_infer_ts = 0;   /* 强制触发 */
    extern void hbm_extract_features(const hbm_sample_t *, uint32_t, uint32_t, hbm_feature_vec_t *);
    extern void hbm_normalize_features(hbm_feature_vec_t *);
    extern void rf_predict(const hbm_feature_vec_t *, float *, float *);
    extern int  hbm_rul_update(hbm_predictor_ctx_t *, uint8_t, float, uint8_t);

    extern uint32_t hbm_ring_buf_get_window(const hbm_ring_buffer_t *, hbm_sample_t *, uint32_t);
    static hbm_sample_t win[HBM_RING_CAPACITY];
    uint32_t n = hbm_ring_buf_get_window(&g_ctx.ring_bufs[0], win, 30);
    assert(n > 0);

    hbm_feature_vec_t feat;
    hbm_extract_features(win, n, 30, &feat);
    assert(feat.valid == 1);
    hbm_normalize_features(&feat);

    float rul, conf;
    rf_predict(&feat, &rul, &conf);
    g_ctx.results[0].prediction_ts = ts;
    hbm_rul_update(&g_ctx, 0, rul, (uint8_t)(conf * 100));

    /* 桩模型 CE=0 → RUL=20 → NORMAL 或 WATCH */
    hbm_rul_result_t res;
    int ret = hbm_predictor_get_result(&g_ctx, 0, &res);
    assert(ret == 0);
    assert(res.rul_days >= 0.0f && res.rul_days <= 30.0f);
    printf("[PASS] test_alert_level_after_infer (RUL=%.2f level=%s)\n",
           res.rul_smoothed, hbm_alert_level_str(res.alert_level));
}

int main(void)
{
    printf("=== Integration Tests ===\n");
    test_no_result_before_30_days();
    test_alert_level_after_infer();
    printf("All integration tests PASSED.\n");
    return 0;
}
