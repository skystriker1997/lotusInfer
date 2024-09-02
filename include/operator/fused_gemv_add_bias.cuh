#pragma once
#include "lotus_utils.hpp"


namespace lotus {

    dim3 MakeSFgemvaGrid(uint32_t weight_h);

    dim3 MakeSFgemvaBlock();

    __device__ __forceinline__ void ReduceAdd(float* input_tile, float* weight_tile, float& result);
    
    __global__ void Fgemva(const float *input, const float *weight, const float* bias, float *output, uint32_t weight_h, uint32_t weight_w, bool use_bias, ActivationFunction af);
        
}