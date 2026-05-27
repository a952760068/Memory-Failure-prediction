/**
 * @file rf_model_stub.h
 * @brief RF 模型桩头文件（单元测试用，模拟极小 RF）
 *
 * 包含 1 棵深度为 1 的测试决策树，预测值恒为 15 天（WATCH 区间）。
 * 仅用于验证 rf_predict.c 的推理流程正确性，不代表真实模型。
 */

#ifndef RF_MODEL_STUB_H
#define RF_MODEL_STUB_H

#include <stdint.h>

#define RF_N_TREES    1
#define RF_N_FEATURES 43
#define RF_MAX_DEPTH  1

/* 桩树：根节点使用特征 0（ce_count_total），阈值 = 500
   - 若 ce <= 500 → 右子叶（RUL=20）
   - 若 ce >  500 → 左子叶（RUL=5，高危）
*/
static const int16_t TREE_0_FEATURE[3]    = { 0, -2, -2 };
static const float   TREE_0_THRESHOLD[3]  = { 500.0f, -2.0f, -2.0f };
static const int16_t TREE_0_LEFT[3]       = { 1, -1, -1 };
static const int16_t TREE_0_RIGHT[3]      = { 2, -1, -1 };
static const float   TREE_0_VALUE[3]      = { 0.0f, 5.0f, 20.0f };

typedef struct {
    const int16_t *feature;
    const float   *threshold;
    const int16_t *left;
    const int16_t *right;
    const float   *value;
    int            n_nodes;
} rf_tree_desc_t;

static const rf_tree_desc_t RF_TREES[RF_N_TREES] = {
    { TREE_0_FEATURE, TREE_0_THRESHOLD, TREE_0_LEFT, TREE_0_RIGHT, TREE_0_VALUE, 3 },
};

static const int RF_TREE_SIZES[RF_N_TREES] = { 3 };

/* Scaler 定义由 feature_scaler.h 提供（mock 或真实版本） */

#endif /* RF_MODEL_STUB_H */
