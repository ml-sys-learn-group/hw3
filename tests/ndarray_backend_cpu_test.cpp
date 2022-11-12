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

GTEST_API_ int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}