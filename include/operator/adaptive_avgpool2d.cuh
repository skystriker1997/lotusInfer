#pragma once
#include "lotus_utils.hpp"

namespace lotus {

    dim3 MakeAAP2dGrid(uint32_t output_c, uint32_t output_h, uint32_t output_w);
    dim3 MakeAAP2dBlock();

    __global__ void AdaptiveAvgpool2d(const float* input, float* output, 
                                        const uint32_t kernel_h, const uint32_t kernel_w, 
                                        const uint32_t input_c, const uint32_t input_h, const uint32_t input_w,  
                                        const float stride_h, const float stride_w, 
                                        const uint32_t output_h, const uint32_t output_w,
                                        ActivationFunction af);
                             
}