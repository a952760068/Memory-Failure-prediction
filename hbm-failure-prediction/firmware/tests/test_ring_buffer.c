/**
 * @file test_ring_buffer.c
 * @brief 环形缓冲区单元测试
 */
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "hbm_types.h"

/* 前向声明 */
extern void hbm_ring_buf_init(hbm_ring_buffer_t *buf, uint8_t hbm_id, uint32_t first_ts);
extern void hbm_ring_buf_push(hbm_ring_buffer_t *buf, const hbm_sample_t *sample);
extern uint32_t hbm_ring_buf_days_available(const hbm_ring_buffer_t *buf);
extern uint32_t hbm_ring_buf_get_window(const hbm_ring_buffer_t *buf,
                                         hbm_sample_t *out, uint32_t n_days);

static hbm_ring_buffer_t g_buf;
static hbm_sample_t g_window[HBM_RING_CAPACITY];

static void push_n_samples(hbm_ring_buffer_t *buf, uint32_t n, uint16_t ce_start)
{
    hbm_sample_t s;
    memset(&s, 0, sizeof(s));
    s.flags = HBM_SAMPLE_FLAG_VALID;
    for (uint32_t i = 0; i < n; i++) {
        s.ce_count = (uint16_t)(ce_start + i);
        s.timestamp = 1700000000U + i * 600U;
        hbm_ring_buf_push(buf, &s);
    }
}

static void test_init(void)
{
    hbm_ring_buf_init(&g_buf, 0, 1700000000U);
    assert(g_buf.count == 0);
    assert(g_buf.head == 0);
    assert(g_buf.hbm_id == 0);
    printf("[PASS] test_init\n");
}

static void test_push_and_count(void)
{
    hbm_ring_buf_init(&g_buf, 1, 1700000000U);
    push_n_samples(&g_buf, 100, 0);
    assert(g_buf.count == 100);
    assert(hbm_ring_buf_days_available(&g_buf) == 0);  /* 100 < 144 */

    push_n_samples(&g_buf, 44, 100);
    assert(g_buf.count == 144);
    assert(hbm_ring_buf_days_available(&g_buf) == 1);
    printf("[PASS] test_push_and_count\n");
}

static void test_wrap_around(void)
{
    hbm_ring_buf_init(&g_buf, 2, 1700000000U);
    /* 填满 + 溢出 100 个 */
    push_n_samples(&g_buf, HBM_RING_CAPACITY + 100, 0);
    assert(g_buf.count == HBM_RING_CAPACITY);
    assert(hbm_ring_buf_days_available(&g_buf) == HBM_WINDOW_DAYS);

    /* 验证最新数据的 ce_count（最后 100 个写入的 ce_count） */
    uint32_t got = hbm_ring_buf_get_window(&g_buf, g_window, 1);
    assert(got == HBM_SAMPLES_PER_DAY);
    printf("[PASS] test_wrap_around\n");
}

static void test_get_window_order(void)
{
    /* 填入 30 天 + 顺序 ce_count */
    hbm_ring_buf_init(&g_buf, 3, 1700000000U);
    uint32_t total = HBM_WINDOW_DAYS * HBM_SAMPLES_PER_DAY;
    push_n_samples(&g_buf, total, 10);

    uint32_t got = hbm_ring_buf_get_window(&g_buf, g_window, HBM_WINDOW_DAYS);
    assert(got == total);
    /* 验证时间顺序：前 < 后 */
    assert(g_window[0].ce_count < g_window[total - 1].ce_count);
    printf("[PASS] test_get_window_order (first=%u, last=%u)\n",
           g_window[0].ce_count, g_window[total - 1].ce_count);
}

int main(void)
{
    printf("=== Ring Buffer Unit Tests ===\n");
    test_init();
    test_push_and_count();
    test_wrap_around();
    test_get_window_order();
    printf("All ring buffer tests PASSED.\n");
    return 0;
}
