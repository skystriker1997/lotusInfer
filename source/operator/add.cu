#include "operator/add.cuh"


namespace lotus {

    dim3 MakeAddGrid(uint32_t size) {
        return {(size+255)/256};
    };

    dim3 MakeAddBlock() {
        return {256};
    };

    __global__ void sadd(const float* x1, const float* x2, float* y, uint32_t size, ActivationFunction af) {
        uint32_t offset = blockIdx.x*256 + threadIdx.x;
        if(offset<size) {
            float tmp = x1[offset] + x2[offset];
            if(af==ActivationFunction::RELU) {
                y[offset] = tmp>0?tmp:0; 
            } else {
                y[offset] = tmp;
            }
        }
    };
                             
}