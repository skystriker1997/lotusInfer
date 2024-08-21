#pragma once
#include "lotus_utils.hpp"


namespace lotus {

    dim3 MakeSFgemvaGrid(uint32_t a_h);

    dim3 MakeSFgemvaBlock();
    
    __global__ void sfgemva(const float *x, const float *a, const float* b, float *y, uint32_t a_h, uint32_t a_w, bool use_bias, ActivationFunction af);
        
}