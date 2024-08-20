#pragma once
#include "lotus_utils.hpp"


#define ADD_GRID(size) {(size+255)/256}
#define ADD_BLOCK() {256}


namespace lotus {

    __global__ void sadd(const float* x1, const float* x2, float* y, uint32_t size, ActivationFunction af);
                             
}