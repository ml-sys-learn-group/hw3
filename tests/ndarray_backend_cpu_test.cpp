//
// Created by hzx on 22-11-9.
//
#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "ndarray_backend_cpu.h"


TEST(ndarray_backend_cpu, Fill) {
    needle::cpu::AlignedArray array(10);
    needle::cpu::Fill(&array, 3.0);
    for(auto i=0; i < array.size; i++) {
        ASSERT_FLOAT_EQ(array.ptr[i], 3.0);
    }
}

TEST(ndarray_backend_cpu, GetElemSize) {
    std::vector<uint32_t> shape = {5, 4, 3};
    size_t elem_size = needle::cpu::GetElemSize(shape);
    ASSERT_EQ(elem_size, 60);
}

TEST(ndarray_backend_cpu, GetCompactStrides) {
    std::vector<uint32_t> shape = {5, 4, 3};
    size_t elem_size = needle::cpu::GetElemSize(shape);
    std::vector<uint32_t> compat_strides;
    needle::cpu::GetCompactStrides(shape, elem_size, compat_strides);
    ASSERT_EQ(compat_strides[0], 12);
    ASSERT_EQ(compat_strides[1], 3);
    ASSERT_EQ(compat_strides[2], 1);
}

TEST(ndarray_backend_cpu, GetGridIndex) {
    std::vector<uint32_t> shape = {5, 4, 3};
    std::vector<uint32_t> compat_strides = {12, 3, 1};
    std::vector<uint32_t> grid_index;
    size_t index_0 = 0;
    std::vector<uint32_t> expect_0 = {0, 0, 0};
    needle::cpu::GetGridIndex(index_0, compat_strides, grid_index);
    ASSERT_TRUE(std::equal(expect_0.begin(), expect_0.end(), grid_index.begin()));

    size_t index_1 = 59;
    std::vector<uint32_t> expect_1 = {4, 3, 2};
    grid_index.clear();
    needle::cpu::GetGridIndex(index_1, compat_strides, grid_index);
    ASSERT_TRUE(std::equal(expect_1.begin(), expect_1.end(), grid_index.begin()));

    size_t index_2 = 29;
    std::vector<uint32_t> expect_2 = {2, 1, 2};
    grid_index.clear();
    needle::cpu::GetGridIndex(index_2, compat_strides, grid_index);
    ASSERT_TRUE(std::equal(expect_2.begin(), expect_2.end(), grid_index.begin()));
}


TEST(ndarray_backend_cpu, GetStorageIndex) {
    std::vector<uint32_t> strides = {12, 3, 1};
    std::vector<uint32_t> grid_index_0 = {2 , 1, 2};
    size_t expect_0 = 29;
    auto index_0 = needle::cpu::GetStorageIndex(grid_index_0, strides);
    ASSERT_EQ(expect_0, index_0);

    std::vector<uint32_t> grid_index_1 = {0, 0, 0};
    size_t expect_1 = 0;
    auto index_1 = needle::cpu::GetStorageIndex(grid_index_1, strides);
    ASSERT_EQ(expect_1, index_1);

    std::vector<uint32_t> grid_index_2 = {0, 0, 0};
    size_t expect_2 = 0;
    auto index_2 = needle::cpu::GetStorageIndex(grid_index_2, strides);
    ASSERT_EQ(expect_2, index_2);
}

TEST(ndarray_backend_cpu, ReduceMax) {
    
}



GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}