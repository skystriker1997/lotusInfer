#pragma once
#include "lotus_utils.hpp"


namespace lotus {

    dim3 MakeMP2dGrid(uint32_t output_c, uint32_t output_h, uint32_t output_w);

    dim3 MakeMP2dBlock();

    __global__ void smaxpool2d(const float* input, float* output, 
                               const uint32_t kernel_h, const uint32_t kernel_w, 
                               const uint32_t input_c, const uint32_t padded_input_h, const uint32_t padded_input_w,  
                               const uint32_t padding_h, const uint32_t padding_w, 
                               const uint32_t stride_h, const uint32_t stride_w, 
                               const uint32_t output_h, const uint32_t output_w);
                             
}