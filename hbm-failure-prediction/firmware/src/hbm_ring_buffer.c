/**
 * @file hbm_ring_buffer.c
 * @brief HBM 遥测数据环形缓冲区实现
 */

#include <string.h>
#include "hbm_types.h"

/*---------------------------------------------------------------------------
 * 初始化
 *---------------------------------------------------------------------------*/

void hbm_ring_buf_init(hbm_ring_buffer_t *buf, uint8_t hbm_id, uint32_t first_ts)
{
    memset(buf, 0, sizeof(*buf));
    buf->hbm_id       = hbm_id;
    buf->first_seen_ts = first_ts;
    buf->head         = 0;
    buf->count        = 0;
}

/*---------------------------------------------------------------------------
 * 写入采样点（覆盖最旧数据）
 *---------------------------------------------------------------------------*/

void hbm_ring_buf_push(hbm_ring_buffer_t *buf, const hbm_sample_t *sample)
{
    buf->samples[buf->head] = *sample;
    buf->head = (buf->head + 1U) % HBM_RING_CAPACITY;
    if (buf->count < HBM_RING_CAPACITY) {
        buf->count++;
    }
}

/*---------------------------------------------------------------------------
 * 返回缓冲区中有效数据覆盖的近似天数
 *---------------------------------------------------------------------------*/

uint32_t hbm_ring_buf_days_available(const hbm_ring_buffer_t *buf)
{
    return buf->count / HBM_SAMPLES_PER_DAY;
}

/*---------------------------------------------------------------------------
 * 提取最近 n_days 天的连续采样点（时间升序写入 out_samples）
 *
 * @param buf         环形缓冲区
 * @param out_samples 输出数组，调用方保证长度 >= n_days * HBM_SAMPLES_PER_DAY
 * @param n_days      提取天数（≤ HBM_WINDOW_DAYS）
 * @return 实际写入的采样点数
 *---------------------------------------------------------------------------*/

uint32_t hbm_ring_buf_get_window(
    const hbm_ring_buffer_t *buf,
    hbm_sample_t *out_samples,
    uint32_t n_days)
{
    uint32_t needed = n_days * HBM_SAMPLES_PER_DAY;
    if (needed > buf->count) {
        needed = buf->count;
    }
    if (needed == 0 || needed > HBM_RING_CAPACITY) {
        return 0U;
    }

    /* 计算起始读取位置（从 head 倒退 needed 步） */
    uint32_t start;
    if (buf->head >= needed) {
        start = buf->head - needed;
    } else {
        start = HBM_RING_CAPACITY - (needed - buf->head);
    }

    /* 分段拷贝（可能跨越数组边界） */
    uint32_t first_chunk = HBM_RING_CAPACITY - start;
    if (first_chunk >= needed) {
        /* 连续区间，直接拷贝 */
        memcpy(out_samples, &buf->samples[start], needed * sizeof(hbm_sample_t));
    } else {
        /* 分两段拷贝 */
        memcpy(out_samples,
               &buf->samples[start],
               first_chunk * sizeof(hbm_sample_t));
        memcpy(out_samples + first_chunk,
               &buf->samples[0],
               (needed - first_chunk) * sizeof(hbm_sample_t));
    }
    return needed;
}
