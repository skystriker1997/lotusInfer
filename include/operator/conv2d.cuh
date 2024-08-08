#pragma once
#include "lotus_utils.hpp"


namespace lotus {
    __global__ void sconv2d(     const float* x, 
                                const float* k, 
                                const float* b, 
                                float* y, 
                                const int k_num, const int k_h, const int k_w, const int k_c, 
                                const int x_w, const int x_h, const int x_c, 
                                const int y_w, const int y_h, const int y_c,
                                const int stride_h, const int stride_w,
                                const int padding_h, const int padding_w
                            );


}