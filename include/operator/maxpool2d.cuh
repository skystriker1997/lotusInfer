#pragma once
#include "lotus_utils.hpp"


namespace lotus {

    dim3 MakeMP2dGrid(uint32_t y_c, uint32_t y_h, uint32_t y_w);

    dim3 MakeMP2dBlock();

    __global__ void smaxpool2d(const float* x, float* y, 
                               const uint32_t kernel_h, const uint32_t kernel_w, 
                               const uint32_t x_c, const uint32_t padded_x_h, const uint32_t padded_x_w, 
                               const uint32_t padding_h, const uint32_t padding_w, 
                               const uint32_t stride_h, const uint32_t stride_w, 
                               const uint32_t y_h, const uint32_t y_w);
                             
}