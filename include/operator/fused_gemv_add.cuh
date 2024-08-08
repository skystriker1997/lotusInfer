#pragma once
#include <cstdint>
#include "lotus_utils.hpp"


namespace lotus {
    
    __global__ void sfgemva(const float *x, const float *a, const float* b, float *y, int x_w, int a_h);
        
}