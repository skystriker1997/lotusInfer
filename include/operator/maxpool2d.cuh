#pragma once
#include "lotus_utils.hpp"


#define MAXPOOL2D_GRID(y_c, y_h, y_w) {(y_w+7)/8, (y_h+7)/8, (y_c+7)/8}
#define MAXPOOL2D_BLOCK() {8, 8, 8}


namespace lotus {

    __global__ void smaxpool2d(const float* x, float* y, 
                               const uint32_t kernel_h, const uint32_t kernel_w, 
                               const uint32_t x_c, const uint32_t padded_x_h, const uint32_t padded_x_w, 
                               const uint32_t padding_h, const uint32_t padding_w, 
                               const uint32_t stride_h, const uint32_t stride_w, 
                               const uint32_t y_h, const uint32_t y_w);
                             
}