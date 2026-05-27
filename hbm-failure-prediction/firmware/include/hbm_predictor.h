/**
 * @file hbm_predictor.h
 * @brief HBM RUL 预测系统对外接口
 */

#ifndef HBM_PREDICTOR_H
#define HBM_PREDICTOR_H

#include "hbm_types.h"

/**
 * @brief 初始化预测系统全局上下文
 *
 * @param ctx      全局上下文指针（调用方分配内存）
 * @param n_hbm    HBM die 数量（1 ~ HBM_MAX_DIES）
 * @param now_ts   当前 Unix 时间戳
 * @return 0 成功，-1 参数错误
 */
int hbm_predictor_init(hbm_predictor_ctx_t *ctx, uint8_t n_hbm, uint32_t now_ts);

/**
 * @brief 定时任务入口（每 10 分钟由 BMC 调用一次）
 *
 * 内部执行：数据采集 → 特征提取 → RF 推理 → 告警判断
 *
 * @param ctx 全局上下文指针
 */
void hbm_predictor_tick(hbm_predictor_ctx_t *ctx);

/**
 * @brief 获取指定 die 的最新预测结果
 *
 * @param ctx    全局上下文指针
 * @param hbm_id HBM die ID
 * @param result 输出结构体指针
 * @return 0 成功，-1 无效 ID 或数据不足
 */
int hbm_predictor_get_result(
    const hbm_predictor_ctx_t *ctx,
    uint8_t hbm_id,
    hbm_rul_result_t *result);

#endif /* HBM_PREDICTOR_H */
