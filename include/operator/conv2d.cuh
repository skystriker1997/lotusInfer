#pragma once
#include "lotus_utils.hpp"


#define CONV2D_GRID(output_c, output_h, output_w) {(output_h*output_w+127)/128, (output_c+127)/128}                                                                                                                            

#define CONV2D_BLOCK() {256}

namespace lotus {


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