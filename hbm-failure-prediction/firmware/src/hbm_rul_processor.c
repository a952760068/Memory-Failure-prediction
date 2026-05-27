/**
 * @file hbm_rul_processor.c
 * @brief RUL 后处理：EMA 平滑 + 告警等级判断
 */

#include <math.h>
#include "hbm_types.h"
#include "hbm_alert.h"

#define EMA_ALPHA       0.3f    /* EMA 平滑系数 */
#define JUMP_THRESHOLD  5.0f   /* RUL 单次跳变超过此值需要两次确认 */

/* 未确认的跳变暂存 */
static float s_pending_rul[HBM_MAX_DIES];
static uint8_t s_pending_confirm[HBM_MAX_DIES];  /* 待确认次数 */

/**
 * @brief 更新 RUL 预测结果并判断告警等级
 *
 * @param ctx        全局上下文
 * @param hbm_id     HBM die ID
 * @param raw_rul    本次原始推理 RUL（天）
 * @param confidence 推理置信度 0-100
 * @return 1 若告警等级发生升级，0 否则
 */
int hbm_rul_update(hbm_predictor_ctx_t *ctx, uint8_t hbm_id, float raw_rul, uint8_t confidence)
{
    if (hbm_id >= ctx->n_hbm) return 0;

    hbm_rul_result_t *res = &ctx->results[hbm_id];

    /* 首次更新：直接设置（无历史值） */
    if (res->prediction_ts == 0) {
        res->rul_smoothed = raw_rul;
        s_pending_rul[hbm_id] = raw_rul;
        s_pending_confirm[hbm_id] = 0;
    } else {
        float delta = raw_rul - res->rul_smoothed;
        if (delta < 0.0f) delta = -delta;

        if (delta > JUMP_THRESHOLD) {
            /* 大跳变：需要第二次确认 */
            if (s_pending_confirm[hbm_id] > 0 &&
                fabsf(raw_rul - s_pending_rul[hbm_id]) < JUMP_THRESHOLD) {
                /* 第二次确认通过，接受新值 */
                res->rul_smoothed = EMA_ALPHA * raw_rul + (1.0f - EMA_ALPHA) * res->rul_smoothed;
                s_pending_confirm[hbm_id] = 0;
            } else {
                /* 第一次出现跳变，暂存等待确认 */
                s_pending_rul[hbm_id] = raw_rul;
                s_pending_confirm[hbm_id] = 1;
            }
        } else {
            /* 正常更新 EMA */
            res->rul_smoothed = EMA_ALPHA * raw_rul + (1.0f - EMA_ALPHA) * res->rul_smoothed;
            s_pending_confirm[hbm_id] = 0;
        }
    }

    res->rul_days   = raw_rul;
    res->confidence = confidence;

    /* 更新告警等级 */
    hbm_alert_level_t new_level = hbm_rul_to_alert_level(res->rul_smoothed);
    res->prev_alert_level = res->alert_level;
    res->alert_level      = new_level;

    return (new_level > res->prev_alert_level) ? 1 : 0;
}

/**
 * @brief 获取 RUL 平滑值对应的告警等级（不修改状态）
 */
hbm_alert_level_t hbm_rul_get_alert_level(float rul_smoothed)
{
    return hbm_rul_to_alert_level(rul_smoothed);
}
