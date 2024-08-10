#pragma once
#include "lotus_utils.hpp"


#define FGEMVA_GRID(a_h) {(a_h+7)/8}                                                                                                                               

#define FGEMVA_BLOCK() {128, 8}



namespace lotus {
    
    __global__ void sfgemva(const float *x, const float *a, const float* b, float *y, uint32_t a_h, uint32_t a_w, bool use_bias, ActivationFunction af);
        
}