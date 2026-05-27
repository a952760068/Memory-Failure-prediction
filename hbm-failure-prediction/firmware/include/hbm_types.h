/**
 * @file hbm_types.h
 * @brief HBM 预测系统核心数据结构定义
 *
 * 所有跨模块数据结构在此文件统一定义，C 侧实现的基础契约。
 * 结构体字段顺序不可随意更改，以确保内存布局一致性。
 */

#ifndef HBM_TYPES_H
#define HBM_TYPES_H

#include <stdint.h>

/*---------------------------------------------------------------------------
 * 常量定义
 *---------------------------------------------------------------------------*/

/** 每天采样点数（10 分钟间隔，每天 144 点） */
#define HBM_SAMPLES_PER_DAY     144U
/** 环形缓冲区总容量（30 天 × 144 点） */
#define HBM_RING_CAPACITY       4320U
/** 观测窗口天数 */
#define HBM_WINDOW_DAYS         30U
/** 特征向量维数（与 Python 侧严格对齐） */
#define HBM_N_FEATURES          43U
/** 支持的最大 HBM die 数量 */
#define HBM_MAX_DIES            8U
/** 采样间隔（秒） */
#define HBM_SAMPLE_INTERVAL_S   600U

/** 告警等级枚举 */
typedef enum {
    HBM_ALERT_NORMAL   = 0,   /**< RUL > 14 天，正常 */
    HBM_ALERT_WATCH    = 1,   /**< RUL ≤ 14 天，观察 */
    HBM_ALERT_WARNING  = 2,   /**< RUL ≤  7 天，警告，建议计划维护 */
    HBM_ALERT_CRITICAL = 3,   /**< RUL ≤  3 天，严重，建议立即更换 */
} hbm_alert_level_t;

/*---------------------------------------------------------------------------
 * 遥测采样点（10 分钟粒度）
 *---------------------------------------------------------------------------*/

/**
 * @brief 单次 HBM 遥测采样点
 *
 * 内存占用：42 字节（不含对齐填充，实际取决于编译器）
 */
typedef struct {
    uint32_t timestamp;         /**< Unix 时间戳（秒） */
    uint16_t ce_count;          /**< 该 10 分钟内 CE（可纠正错误）计数 */
    uint8_t  ue_count;          /**< 该 10 分钟内 UE（不可纠正错误）计数 */
    uint8_t  ce_col_map[32];    /**< 列级 CE 分布位图（256 列，1 bit/列） */
    int8_t   temp_c;            /**< HBM 温度整数部分（摄氏度，-128~127） */
    uint8_t  temp_frac;         /**< 温度小数部分 × 10（精度 0.1°C，0~9） */
    uint16_t power_mw;          /**< HBM 功耗（毫瓦，0~65535 mW ≈ 65.5 W） */
    uint8_t  flags;             /**< bit0: valid, bit1: outlier, bit2-7: 保留 */
} hbm_sample_t;

#define HBM_SAMPLE_FLAG_VALID    (1U << 0)
#define HBM_SAMPLE_FLAG_OUTLIER  (1U << 1)

/** 获取采样点温度（浮点值）的宏 */
#define HBM_SAMPLE_TEMP_F(s)  ((float)(s)->temp_c + (float)(s)->temp_frac * 0.1f)
/** 获取采样点功耗（瓦）的宏 */
#define HBM_SAMPLE_POWER_W(s) ((float)(s)->power_mw * 0.001f)

/*---------------------------------------------------------------------------
 * 环形缓冲区（每个 HBM die 独立一组）
 *---------------------------------------------------------------------------*/

/**
 * @brief 30 天遥测数据环形缓冲区
 *
 * 内存占用约：4320 × 42 bytes ≈ 181 KB/die
 * 8 个 die 合计约 1.4 MB，需确认 BMC RAM 可用空间。
 */
typedef struct {
    hbm_sample_t samples[HBM_RING_CAPACITY]; /**< 采样数据环形数组 */
    uint32_t     head;                        /**< 下一次写入位置（0 ~ HBM_RING_CAPACITY-1） */
    uint32_t     count;                       /**< 已有效数据量（最大 HBM_RING_CAPACITY） */
    uint8_t      hbm_id;                      /**< HBM die ID（0 ~ HBM_MAX_DIES-1） */
    uint32_t     first_seen_ts;               /**< 首次记录时间戳（用于计算 hbm_age_days） */
} hbm_ring_buffer_t;

/*---------------------------------------------------------------------------
 * 特征向量
 *---------------------------------------------------------------------------*/

/**
 * @brief 43 维特征向量
 *
 * raw[]:        原始特征值（未归一化）
 * normalized[]: z-score 归一化后的值，供 RF 推理使用
 * valid:        数据量充足（≥30 天）时为 1，否则为 0
 */
typedef struct {
    float   raw[HBM_N_FEATURES];
    float   normalized[HBM_N_FEATURES];
    uint8_t valid;
} hbm_feature_vec_t;

/*---------------------------------------------------------------------------
 * 预测结果与告警状态
 *---------------------------------------------------------------------------*/

/**
 * @brief 单次 RUL 预测结果
 */
typedef struct {
    uint8_t          hbm_id;          /**< HBM die ID */
    float            rul_days;         /**< RF 推理原始 RUL（天） */
    float            rul_smoothed;     /**< EMA 平滑后 RUL（α=0.3） */
    hbm_alert_level_t alert_level;    /**< 当前告警等级 */
    hbm_alert_level_t prev_alert_level; /**< 上次告警等级（用于检测升级） */
    uint32_t         prediction_ts;   /**< 本次预测时间戳 */
    uint8_t          confidence;       /**< 置信度 0~100（基于树间方差） */
} hbm_rul_result_t;

/*---------------------------------------------------------------------------
 * 全局预测上下文
 *---------------------------------------------------------------------------*/

/**
 * @brief BMC HBM 预测系统全局上下文
 *
 * 由 hbm_predictor_init() 初始化，所有模块共享此结构。
 */
typedef struct {
    hbm_ring_buffer_t ring_bufs[HBM_MAX_DIES];  /**< 每个 die 的环形缓冲区 */
    hbm_rul_result_t  results[HBM_MAX_DIES];    /**< 每个 die 的最新预测结果 */
    uint8_t           n_hbm;                    /**< 实际 HBM die 数量 */
    uint32_t          last_infer_ts;            /**< 上次推理时间戳 */
    uint32_t          infer_interval_s;         /**< 推理间隔（秒，默认 600） */
} hbm_predictor_ctx_t;

#endif /* HBM_TYPES_H */
