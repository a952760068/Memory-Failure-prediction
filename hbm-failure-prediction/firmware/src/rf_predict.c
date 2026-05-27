/**
 * @file rf_predict.c
 * @brief 随机森林 RUL 推理核心（纯 C，无外部依赖）
 *
 * 依赖 rf_model.h（由 model_exporter.py 自动生成）。
 * rf_model.h 不存在时用桩头文件替代（用于单元测试）。
 */

#include <string.h>
#include <math.h>
#include "hbm_types.h"

/* 包含模型头文件（自动生成），若不存在则使用 mock */
#if defined(RF_MODEL_STUB)
#  include "../platform/mock/rf_model_stub.h"
#else
#  include "rf_model.h"
#endif

/*---------------------------------------------------------------------------
 * 内部：单棵树推理（rf_tree_desc_t 由 rf_model.h / rf_model_stub.h 定义）
 *---------------------------------------------------------------------------*/

/* rf_model.h 提供 RF_TREES[]、RF_N_TREES、RF_MAX_DEPTH 以及 rf_tree_desc_t */

static float rf_predict_single_tree(const rf_tree_desc_t *tree, const float *feat)
{
    int node = 0;
    /* 最大迭代深度 = 2 * RF_MAX_DEPTH，防止无限循环 */
    int max_iter = RF_MAX_DEPTH * 2 + 2;
    for (int iter = 0; iter < max_iter; iter++) {
        int feat_idx = (int)tree->feature[node];
        if (feat_idx == -2) {   /* sklearn 叶节点标记：TREE_UNDEFINED = -2 */
            return tree->value[node];
        }
        float threshold = tree->threshold[node];
        if (feat[feat_idx] <= threshold) {
            node = (int)tree->left[node];
        } else {
            node = (int)tree->right[node];
        }
        if (node < 0) break;
    }
    return tree->value[node < 0 ? 0 : node];
}

/*---------------------------------------------------------------------------
 * 对外接口：全森林推理
 *---------------------------------------------------------------------------*/

/**
 * @brief 对归一化特征向量执行 RF 推理
 *
 * @param feat         归一化特征向量（hbm_feature_vec_t.normalized）
 * @param rul_out      输出 RUL 天数（均值）
 * @param confidence   输出置信度 0.0~1.0（基于树间预测方差）
 */
void rf_predict(const hbm_feature_vec_t *feat, float *rul_out, float *confidence)
{
    float tree_preds[RF_N_TREES];
    float sum = 0.0f;

    for (int t = 0; t < RF_N_TREES; t++) {
        tree_preds[t] = rf_predict_single_tree(&RF_TREES[t], feat->normalized);
        sum += tree_preds[t];
    }

    float mean = sum / (float)RF_N_TREES;

    /* 计算树间方差 → 置信度 */
    float var = 0.0f;
    for (int t = 0; t < RF_N_TREES; t++) {
        float diff = tree_preds[t] - mean;
        var += diff * diff;
    }
    var /= (float)RF_N_TREES;
    float std_dev = sqrtf(var);

    /* 变异系数取反：CV = std/mean，置信度 = 1 - clamp(CV, 0, 1) */
    float cv = (mean > 1e-4f) ? (std_dev / mean) : 1.0f;
    if (cv > 1.0f) cv = 1.0f;

    *rul_out    = mean;
    *confidence = 1.0f - cv;

    /* 裁剪至合理范围 [0, 30] 天 */
    if (*rul_out < 0.0f)  *rul_out = 0.0f;
    if (*rul_out > 30.0f) *rul_out = 30.0f;
}
