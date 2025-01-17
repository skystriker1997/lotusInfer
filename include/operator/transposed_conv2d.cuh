#pragma once
#include "lotus_utils.hpp"

namespace lotus {

    dim3 MakeTransposedConv2dGrid(uint32_t output_c, uint32_t output_h, uint32_t output_w, uint32_t padding_h, uint32_t padding_w);

    dim3 MakeTransposedConv2dBlock();

    __global__ void TransConv2d(const float* input, 
                                 const float* k, 
                                 bool use_bias, const float* b, 
                                 float* output, 
                                 const uint32_t k_num, const uint32_t k_c, const uint32_t k_h, const uint32_t k_w, 
                                 const uint32_t input_c, const uint32_t input_h, const uint32_t input_w,    
                                 const uint32_t output_c, const uint32_t output_h, const uint32_t output_w,
                                 const uint32_t stride_h, const uint32_t stride_w,
                                 const uint32_t padding_h, const uint32_t padding_w,
                                 ActivationFunction af);
                                 
}

