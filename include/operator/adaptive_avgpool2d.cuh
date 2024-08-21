#pragma once
#include "lotus_utils.hpp"

namespace lotus {

    dim3 MakeAAP2dGrid(uint32_t y_c, uint32_t y_h, uint32_t y_w);
    dim3 MakeAAP2dBlock();

    __global__ void sadaptive_avgpool2d(const float* x, float* y, 
                                        const uint32_t kernel_h, const uint32_t kernel_w, 
                                        const uint32_t x_c, const uint32_t x_h, const uint32_t x_w,  
                                        const float stride_h, const float stride_w, 
                                        const uint32_t y_h, const uint32_t y_w);
                             
}