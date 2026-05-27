/**
 * @file hbm_alert_manager.c
 * @brief HBM 告警触发：IPMI SEL / Redfish / SNMP
 *
 * 实际的 IPMI/Redfish/SNMP 调用依赖平台 SDK，此处提供标准接口。
 * 在没有真实 BMC SDK 的环境下，通过 platform 适配层的弱符号实现。
 */

#include <string.h>
#include <stdio.h>
#include "hbm_types.h"
#include "hbm_alert.h"
#include "hbm_predictor.h"

/* 告警抑制时间戳（每个 die 每个等级独立跟踪） */
static uint32_t s_last_alert_ts[HBM_MAX_DIES][4];  /* [die][level] */

/*---------------------------------------------------------------------------
 * 平台回调（弱符号，平台适配层覆盖）
 *---------------------------------------------------------------------------*/

/**
 * @brief 写入 IPMI SEL OEM 事件（平台实现）
 *
 * OEM Data 格式：
 *   Byte 0: hbm_id
 *   Byte 1: alert_level
 *   Byte 2-3: rul_days × 10（uint16 big-endian）
 *   Byte 4: confidence
 */
__attribute__((weak)) void hbm_platform_write_ipmi_sel(
    uint8_t hbm_id, uint8_t level, uint16_t rul_x10, uint8_t confidence)
{
    /* 默认实现：打印到标准输出（仅用于模拟/调试） */
    printf("[IPMI SEL] HBM%u  Level=%u  RUL=%.1fd  Confidence=%u%%\n",
           hbm_id, level, (float)rul_x10 / 10.0f, confidence);
}

/**
 * @brief 通过 Redfish 上报告警事件（平台实现）
 */
__attribute__((weak)) void hbm_platform_send_redfish_event(
    uint8_t hbm_id, uint8_t level, float rul_days, uint8_t confidence)
{
    printf("[Redfish]  HBM%u  Severity=%s  RUL=%.2fd  Confidence=%u%%\n",
           hbm_id, hbm_alert_level_str((hbm_alert_level_t)level), rul_days, confidence);
}

/**
 * @brief 发送 SNMP Trap（平台实现）
 */
__attribute__((weak)) void hbm_platform_send_snmp_trap(
    uint8_t hbm_id, uint8_t level, float rul_days)
{
    printf("[SNMP Trap] HBM%u  Level=%s  RUL=%.2fd\n",
           hbm_id, hbm_alert_level_str((hbm_alert_level_t)level), rul_days);
}

/*---------------------------------------------------------------------------
 * 内部：判断是否需要发送告警（避免重复上报）
 *---------------------------------------------------------------------------*/

static int should_send_alert(uint8_t hbm_id, uint8_t level, uint32_t now_ts)
{
    if (hbm_id >= HBM_MAX_DIES || level >= 4) return 0;
    uint32_t last = s_last_alert_ts[hbm_id][level];
    if (last == 0 || (now_ts - last) >= HBM_ALERT_SUPPRESS_S) {
        s_last_alert_ts[hbm_id][level] = now_ts;
        return 1;
    }
    return 0;
}

/*---------------------------------------------------------------------------
 * 对外接口
 *---------------------------------------------------------------------------*/

/**
 * @brief 遍历所有 die，根据 RUL 告警等级触发通知
 */
void hbm_alert_manager_check_all(hbm_predictor_ctx_t *ctx)
{
    for (uint8_t i = 0; i < ctx->n_hbm; i++) {
        const hbm_rul_result_t *res = &ctx->results[i];

        /* 仅对 WATCH 及以上等级上报，或等级发生升级时上报 */
        int level_up = (res->alert_level > res->prev_alert_level);
        int need_report = (res->alert_level >= HBM_ALERT_WATCH) || level_up;

        if (!need_report) continue;
        if (!should_send_alert(i, (uint8_t)res->alert_level, res->prediction_ts)) continue;

        uint16_t rul_x10 = (uint16_t)(res->rul_smoothed * 10.0f);

        /* 发送 IPMI SEL */
        hbm_platform_write_ipmi_sel(i, (uint8_t)res->alert_level, rul_x10, res->confidence);

        /* 发送 Redfish Event */
        hbm_platform_send_redfish_event(i, (uint8_t)res->alert_level, res->rul_smoothed, res->confidence);

        /* CRITICAL 时额外发送 SNMP Trap */
        if (res->alert_level == HBM_ALERT_CRITICAL) {
            hbm_platform_send_snmp_trap(i, (uint8_t)res->alert_level, res->rul_smoothed);
        }
    }
}

/**
 * @brief 强制上报指定 die 的告警（无抑制）
 */
void hbm_alert_force_report(const hbm_predictor_ctx_t *ctx, uint8_t hbm_id)
{
    if (hbm_id >= ctx->n_hbm) return;
    const hbm_rul_result_t *res = &ctx->results[hbm_id];
    uint16_t rul_x10 = (uint16_t)(res->rul_smoothed * 10.0f);
    hbm_platform_write_ipmi_sel(hbm_id, (uint8_t)res->alert_level, rul_x10, res->confidence);
    hbm_platform_send_redfish_event(hbm_id, (uint8_t)res->alert_level, res->rul_smoothed, res->confidence);
}
