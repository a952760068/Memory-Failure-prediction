/**
 * @file feature_scaler.h (STUB)
 * @brief 单元测试用桩：均值=0，标准差=1（不做任何变换）
 *
 * 真实训练后由 model_exporter.py 生成并覆盖 firmware/include/feature_scaler.h
 */
#ifndef FEATURE_SCALER_H
#define FEATURE_SCALER_H

#define SCALER_N_FEATURES 43

static const float FEATURE_MEAN[43] = {
    0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0, 0,0,0,
    0,0,0,0,0,0,0, 0,0, 0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0
};
static const float FEATURE_STD[43] = {
    1,1,1,1,1,1,1, 1,1,1,1,1,1, 1,1, 1,1,1,
    1,1,1,1,1,1,1, 1,1, 1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,1
};

#endif /* FEATURE_SCALER_H */
