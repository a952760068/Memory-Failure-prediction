cmake_minimum_required(VERSION 3.16)
project(hbm_predictor C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-parameter")

# 包含头文件目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/platform/mock
)

# Mock 平台文件（主机测试用）
set(MOCK_PLATFORM
    platform/mock/hbm_platform_mock.c
)

# 固件核心源文件
set(FIRMWARE_SRCS
    src/hbm_ring_buffer.c
    src/hbm_feat_extract.c
    src/hbm_rul_processor.c
    src/hbm_alert_manager.c
    src/hbm_predictor.c
    # rf_predict.c 需要 rf_model.h，测试时使用 RF_MODEL_STUB 宏
    src/rf_predict.c
)

# ─────────────────────────────────────────────────────────────────
# 可执行：单元测试（使用桩模型和 Mock 平台）
# ─────────────────────────────────────────────────────────────────

add_compile_definitions(RF_MODEL_STUB)

add_executable(test_ring_buffer
    tests/test_ring_buffer.c
    src/hbm_ring_buffer.c
    ${MOCK_PLATFORM}
)
target_include_directories(test_ring_buffer PRIVATE include platform/mock)

add_executable(test_feat_extract
    tests/test_feat_extract.c
    src/hbm_feat_extract.c
    ${MOCK_PLATFORM}
)
target_include_directories(test_feat_extract PRIVATE include platform/mock)
target_link_libraries(test_feat_extract m)

add_executable(test_rf_predict
    tests/test_rf_predict.c
    src/rf_predict.c
    src/hbm_feat_extract.c
    ${MOCK_PLATFORM}
)
target_include_directories(test_rf_predict PRIVATE include platform/mock)
target_link_libraries(test_rf_predict m)

add_executable(test_integration
    tests/test_integration.c
    ${FIRMWARE_SRCS}
    ${MOCK_PLATFORM}
)
target_include_directories(test_integration PRIVATE include platform/mock)
target_link_libraries(test_integration m)

# ─────────────────────────────────────────────────────────────────
# CTest 注册
# ─────────────────────────────────────────────────────────────────
enable_testing()
add_test(NAME RingBuffer   COMMAND test_ring_buffer)
add_test(NAME FeatExtract  COMMAND test_feat_extract)
add_test(NAME RFPredict    COMMAND test_rf_predict)
add_test(NAME Integration  COMMAND test_integration)
