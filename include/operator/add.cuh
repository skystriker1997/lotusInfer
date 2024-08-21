#pragma once
#include "lotus_utils.hpp"


namespace lotus {

    dim3 MakeAddGrid(uint32_t size);

    dim3 MakeAddBlock();

    __global__ void sadd(const float* x1, const float* x2, float* y, uint32_t size, ActivationFunction af);
                             
}