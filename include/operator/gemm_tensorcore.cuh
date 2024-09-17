#pragma once
#include "lotus_utils.hpp"


namespace lotus {


    dim3 MakeGemmGrid(uint32_t m, uint32_t n);

    dim3 MakeGemmBlock();

    __global__ void Gemm(float const* a, float const* b, bool use_bias, float const* bias, float* c, uint32_t m, uint32_t n, uint32_t k, ActivationFunction af);

}

