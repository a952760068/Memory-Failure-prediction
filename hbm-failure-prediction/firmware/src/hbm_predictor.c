/**
 * @file hbm_predictor.c
 * @brief HBM 预测系统主调度模块
 *
 * 每 10 分钟由 BMC 定时器调用 hbm_predictor_tick()，
 * 内部执行：采集 → 缓冲 → 推理 → 告警
 */

#include <string.h>
#include "hbm_types.h"
#include "hbm_predictor.h"
#include "hbm_alert.h"

/* 前向声明（来自各实现文件） */
extern void hbm_ring_buf_init(hbm_ring_buffer_t *buf, uint8_t hbm_id, uint32_t first_ts);
extern void hbm_ring_buf_push(hbm_ring_buffer_t *buf, const hbm_sample_t *sample);
extern uint32_t hbm_ring_buf_days_available(const hbm_ring_buffer_t *buf);
extern uint32_t hbm_ring_buf_get_window(const hbm_ring_buffer_t *buf,
                                         hbm_sample_t *out, uint32_t n_days);
extern void hbm_extract_features(const hbm_sample_t *samples, uint32_t n_samples,
                                   uint32_t hbm_age_days, hbm_feature_vec_t *feat);
extern void hbm_normalize_features(hbm_feature_vec_t *feat);
extern void rf_predict(const hbm_feature_vec_t *feat, float *rul_out, float *confidence);
extern int  hbm_rul_update(hbm_predictor_ctx_t *ctx, uint8_t hbm_id,
                            float raw_rul, uint8_t confidence);
extern void hbm_alert_manager_check_all(hbm_predictor_ctx_t *ctx);

/* 平台层：读取遥测数据（由平台适配层实现） */
extern int hbm_platform_read_telemetry(uint8_t hbm_id, hbm_sample_t *sample_out);
extern uint32_t hbm_platform_get_time(void);

/* 推理用临时窗口缓冲（避免栈溢出，使用静态全局） */
static hbm_sample_t s_window_buf[HBM_RING_CAPACITY];

/*---------------------------------------------------------------------------
 * 初始化
 *---------------------------------------------------------------------------*/

int hbm_predictor_init(hbm_predictor_ctx_t *ctx, uint8_t n_hbm, uint32_t now_ts)
{
    if (ctx == NULL || n_hbm == 0 || n_hbm > HBM_MAX_DIES) return -1;

    memset(ctx, 0, sizeof(*ctx));
    ctx->n_hbm           = n_hbm;
    ctx->last_infer_ts   = 0;
    ctx->infer_interval_s = HBM_SAMPLE_INTERVAL_S;

    for (uint8_t i = 0; i < n_hbm; i++) {
        hbm_ring_buf_init(&ctx->ring_bufs[i], i, now_ts);
        ctx->results[i].hbm_id = i;
        ctx->results[i].rul_smoothed = 30.0f;   /* 初始 RUL = 30天（健康） */
        ctx->results[i].alert_level  = HBM_ALERT_NORMAL;
    }
    return 0;
}

/*---------------------------------------------------------------------------
 * 定时任务主入口
 *---------------------------------------------------------------------------*/

void hbm_predictor_tick(hbm_predictor_ctx_t *ctx)
{
    uint32_t now = hbm_platform_get_time();

    /*----------------------------------------------------------------------
     * 阶段 1：数据采集（每次 tick 必须执行）
     *----------------------------------------------------------------------*/
    for (uint8_t i = 0; i < ctx->n_hbm; i++) {
        hbm_sample_t sample;
        int ret = hbm_platform_read_telemetry(i, &sample);
        if (ret == 0) {
            sample.timestamp = now;
            sample.flags |= HBM_SAMPLE_FLAG_VALID;
            hbm_ring_buf_push(&ctx->ring_bufs[i], &sample);
        }
    }

    /*----------------------------------------------------------------------
     * 阶段 2：检查推理间隔
     *----------------------------------------------------------------------*/
    if (ctx->last_infer_ts != 0 &&
        (now - ctx->last_infer_ts) < ctx->infer_interval_s) {
        return;
    }
    ctx->last_infer_ts = now;

    /*----------------------------------------------------------------------
     * 阶段 3：逐 die 特征提取 + RF 推理
     *----------------------------------------------------------------------*/
    for (uint8_t i = 0; i < ctx->n_hbm; i++) {
        hbm_ring_buffer_t *rbuf = &ctx->ring_bufs[i];

        /* 数据不足 30 天，跳过推理 */
        if (hbm_ring_buf_days_available(rbuf) < HBM_WINDOW_DAYS) continue;

        /* 提取 30 天窗口 */
        uint32_t n_got = hbm_ring_buf_get_window(rbuf, s_window_buf, HBM_WINDOW_DAYS);
        if (n_got == 0) continue;

        /* HBM 在线天数 */
        uint32_t age_days = (rbuf->first_seen_ts > 0 && now > rbuf->first_seen_ts) ?
            (now - rbuf->first_seen_ts) / 86400U : 0U;

        /* 特征提取 */
        hbm_feature_vec_t feat;
        hbm_extract_features(s_window_buf, n_got, age_days, &feat);
        if (!feat.valid) continue;

        /* 归一化 */
        hbm_normalize_features(&feat);

        /* RF 推理 */
        float raw_rul = 30.0f, confidence = 0.5f;
        rf_predict(&feat, &raw_rul, &confidence);

        /* 更新结果与告警等级 */
        ctx->results[i].prediction_ts = now;
        hbm_rul_update(ctx, i, raw_rul, (uint8_t)(confidence * 100.0f));
    }

    /*----------------------------------------------------------------------
     * 阶段 4：告警检查与上报
     *----------------------------------------------------------------------*/
    hbm_alert_manager_check_all(ctx);
}

/*---------------------------------------------------------------------------
 * 查询接口
 *---------------------------------------------------------------------------*/

int hbm_predictor_get_result(
    const hbm_predictor_ctx_t *ctx,
    uint8_t hbm_id,
    hbm_rul_result_t *result)
{
    if (ctx == NULL || result == NULL || hbm_id >= ctx->n_hbm) return -1;
    if (ctx->results[hbm_id].prediction_ts == 0) return -1;
    *result = ctx->results[hbm_id];
    return 0;
}
