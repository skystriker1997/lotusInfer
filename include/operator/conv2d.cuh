#pragma once
#include "lotus_utils.hpp"


namespace lotus {

    dim3 MakeConv2dGrid(uint32_t y_c, uint32_t y_h, uint32_t y_w);

    dim3 MakeConv2dBlock();

    __global__ void sconv2d(const float* x, 
                            const float* k, 
                            bool use_bias, const float* b, 
                            float* y, 
                            const uint32_t k_num, const uint32_t k_c, const uint32_t k_h, const uint32_t k_w, 
                            const uint32_t x_c, const uint32_t padded_x_h, const uint32_t padded_x_w,    
                            const uint32_t y_c, const uint32_t y_h, const uint32_t y_w,
                            const uint32_t stride_h, const uint32_t stride_w,
                            const uint32_t padding_h, const uint32_t padding_w,
                            ActivationFunction af
                            );


}