/**
 * @file hbm_platform.c
 * @brief AST2600 平台适配层（I2C / SMBPBI 遥测读取）
 *
 * 实际 BMC 固件集成时需根据具体 SDK 和寄存器地址完善此文件。
 * 此处提供完整的函数框架和关键注释，方便移植。
 */

#include <stdint.h>
#include <string.h>
#include <time.h>
#include "hbm_types.h"

/*---------------------------------------------------------------------------
 * 平台相关头文件（真实 BMC 环境下取消注释）
 *---------------------------------------------------------------------------*/
/* #include "ast_i2c.h"       */  /* AST2600 I2C 主机驱动 */
/* #include "ast_smbpbi.h"    */  /* SMBPBI（GPU 带外接口）驱动 */
/* #include "ast_pmbus.h"     */  /* PMBus 功耗读取 */

/*---------------------------------------------------------------------------
 * 硬件地址定义（示例，需根据实际原理图修改）
 *---------------------------------------------------------------------------*/
/** I2C 总线编号（GPU0 连接到 AST2600 I2C5） */
#define GPU_I2C_BUS        5
/** GPU HBM 温度传感器基地址（SMBPBI slave addr，每个 GPU 不同） */
#define HBM_TEMP_I2C_BASE  0x70
/** PMBus 功耗传感器 I2C 地址 */
#define POWER_PMBUS_ADDR   0x58
/** SMBPBI ECC 读取命令码 */
#define SMBPBI_CMD_ECC_CE  0x01
#define SMBPBI_CMD_ECC_UE  0x02

/*---------------------------------------------------------------------------
 * 内部：读取 HBM ECC 错误计数（通过 SMBPBI/SMBus）
 *---------------------------------------------------------------------------*/
static int read_ecc_counts(uint8_t hbm_id, uint16_t *ce_out, uint8_t *ue_out)
{
    /*
     * 实际实现步骤：
     * 1. 通过 I2C/SMBPBI 向 GPU 发送 ECC 查询命令
     * 2. GPU 返回 CE/UE 计数寄存器值
     * 3. 计算自上次读取以来的增量（delta），防止累积计数溢出
     *
     * 示例（伪代码）：
     *   uint8_t buf[4];
     *   ast_i2c_read(GPU_I2C_BUS, HBM_TEMP_I2C_BASE + hbm_id,
     *                SMBPBI_CMD_ECC_CE, buf, 4);
     *   *ce_out = (uint16_t)((buf[1] << 8) | buf[0]);
     *   *ue_out = buf[2];
     */

    /* 占位：返回 0（正常状态） */
    (void)hbm_id;
    *ce_out = 0;
    *ue_out = 0;
    return 0;  /* 0=成功，-1=失败 */
}

/*---------------------------------------------------------------------------
 * 内部：读取 HBM 温度（通过 I2C 温度传感器或 PECI）
 *---------------------------------------------------------------------------*/
static int read_hbm_temperature(uint8_t hbm_id, int8_t *temp_c, uint8_t *temp_frac)
{
    /*
     * 实际实现步骤：
     * 1. 通过 I2C 读取 GPU 板载温度传感器（LM75 或专有寄存器）
     * 2. 解析 10bit 温度值（符号位 + 9位整数 + 1位0.5°C）
     *
     * 示例（伪代码）：
     *   uint8_t reg = 0x00;  // Temperature Register
     *   uint8_t buf[2];
     *   ast_i2c_read(GPU_I2C_BUS, HBM_TEMP_I2C_BASE + hbm_id, reg, buf, 2);
     *   int16_t raw = (int16_t)((buf[0] << 8) | buf[1]) >> 5;
     *   *temp_c    = (int8_t)(raw / 8);
     *   *temp_frac = (uint8_t)(((raw % 8) * 10) / 8);
     */

    (void)hbm_id;
    *temp_c   = 70;   /* 默认 70°C */
    *temp_frac = 0;
    return 0;
}

/*---------------------------------------------------------------------------
 * 内部：读取功耗（通过 PMBus）
 *---------------------------------------------------------------------------*/
static int read_hbm_power(uint8_t hbm_id, uint16_t *power_mw)
{
    /*
     * 实际实现步骤：
     * 1. 通过 PMBus READ_POUT 命令读取 GPU 功耗
     * 2. 根据 PMBus Linear Data Format 解析实际功率值
     *
     * 示例（伪代码）：
     *   uint8_t buf[2];
     *   ast_pmbus_read(GPU_I2C_BUS, POWER_PMBUS_ADDR, 0x96, buf, 2);
     *   uint16_t linear = (buf[1] << 8) | buf[0];
     *   *power_mw = pmbus_linear_to_milliwatt(linear);
     */

    (void)hbm_id;
    *power_mw = 30000;  /* 默认 30W */
    return 0;
}

/*---------------------------------------------------------------------------
 * 对外接口实现
 *---------------------------------------------------------------------------*/

int hbm_platform_read_telemetry(uint8_t hbm_id, hbm_sample_t *sample_out)
{
    if (hbm_id >= HBM_MAX_DIES || sample_out == NULL) return -1;

    memset(sample_out, 0, sizeof(*sample_out));

    uint16_t ce = 0; uint8_t ue = 0;
    int8_t   temp_c = 0; uint8_t temp_frac = 0;
    uint16_t power_mw = 0;

    int ret = 0;
    ret |= read_ecc_counts(hbm_id, &ce, &ue);
    ret |= read_hbm_temperature(hbm_id, &temp_c, &temp_frac);
    ret |= read_hbm_power(hbm_id, &power_mw);

    sample_out->ce_count  = ce;
    sample_out->ue_count  = ue;
    sample_out->temp_c    = temp_c;
    sample_out->temp_frac = temp_frac;
    sample_out->power_mw  = power_mw;
    sample_out->flags     = (ret == 0) ? HBM_SAMPLE_FLAG_VALID : 0;

    return ret;
}

uint32_t hbm_platform_get_time(void)
{
    return (uint32_t)time(NULL);
}
