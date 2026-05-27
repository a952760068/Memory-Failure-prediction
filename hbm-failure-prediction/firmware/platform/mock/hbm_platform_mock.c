/**
 * @file hbm_platform_mock.c
 * @brief 平台适配层 Mock 实现（单元测试 / 主机仿真用）
 *
 * 在 x86 主机上编译测试时使用，提供：
 * - 伪造的遥测数据
 * - 系统时钟（使用 time()）
 * - printf 替代的告警输出
 */

#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "hbm_types.h"

/* 可在测试代码中修改这些全局变量来注入自定义数据 */
uint16_t mock_ce_count[HBM_MAX_DIES] = {0};
uint8_t  mock_ue_count[HBM_MAX_DIES] = {0};
int8_t   mock_temp_c[HBM_MAX_DIES]   = {70, 70, 70, 70, 70, 70, 70, 70};
uint8_t  mock_temp_frac[HBM_MAX_DIES]= {0};
uint16_t mock_power_mw[HBM_MAX_DIES] = {30000, 30000, 30000, 30000,
                                         30000, 30000, 30000, 30000};

int hbm_platform_read_telemetry(uint8_t hbm_id, hbm_sample_t *sample_out)
{
    if (hbm_id >= HBM_MAX_DIES || sample_out == NULL) return -1;
    memset(sample_out, 0, sizeof(*sample_out));
    sample_out->ce_count   = mock_ce_count[hbm_id];
    sample_out->ue_count   = mock_ue_count[hbm_id];
    sample_out->temp_c     = mock_temp_c[hbm_id];
    sample_out->temp_frac  = mock_temp_frac[hbm_id];
    sample_out->power_mw   = mock_power_mw[hbm_id];
    sample_out->flags      = HBM_SAMPLE_FLAG_VALID;
    return 0;
}

uint32_t hbm_platform_get_time(void)
{
    return (uint32_t)time(NULL);
}
