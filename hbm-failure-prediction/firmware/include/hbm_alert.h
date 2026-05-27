/**
 * @file hbm_alert.h
 * @brief 告警阈值与接口声明
 */

#ifndef HBM_ALERT_H
#define HBM_ALERT_H

#include "hbm_types.h"

/*---------------------------------------------------------------------------
 * 告警阈值（与 model_config.yaml 保持一致）
 *---------------------------------------------------------------------------*/
#define HBM_RUL_CRITICAL_DAYS    3.0f   /**< RUL ≤ 3 天: CRITICAL */
#define HBM_RUL_WARNING_DAYS     7.0f   /**< RUL ≤ 7 天: WARNING */
#define HBM_RUL_WATCH_DAYS      14.0f   /**< RUL ≤ 14 天: WATCH */

/** 告警抑制间隔（同等级告警 1 小时内不重复上报） */
#define HBM_ALERT_SUPPRESS_S     3600U

/** IPMI SEL OEM 传感器编号基址（HBM0 = 0xD0，HBM7 = 0xD7） */
#define HBM_IPMI_SENSOR_BASE     0xD0

/** 根据 RUL（天）判断告警等级 */
static inline hbm_alert_level_t hbm_rul_to_alert_level(float rul_days)
{
    if (rul_days <= HBM_RUL_CRITICAL_DAYS) return HBM_ALERT_CRITICAL;
    if (rul_days <= HBM_RUL_WARNING_DAYS)  return HBM_ALERT_WARNING;
    if (rul_days <= HBM_RUL_WATCH_DAYS)    return HBM_ALERT_WATCH;
    return HBM_ALERT_NORMAL;
}

/** 告警等级字符串（调试用） */
static inline const char *hbm_alert_level_str(hbm_alert_level_t level)
{
    switch (level) {
    case HBM_ALERT_NORMAL:   return "NORMAL";
    case HBM_ALERT_WATCH:    return "WATCH";
    case HBM_ALERT_WARNING:  return "WARNING";
    case HBM_ALERT_CRITICAL: return "CRITICAL";
    default:                 return "UNKNOWN";
    }
}

#endif /* HBM_ALERT_H */
