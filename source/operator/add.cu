#include "operator/add.cuh"


namespace lotus {

    dim3 MakeAddGrid(uint32_t size) {
        return {(size+255)/256};
    };

    dim3 MakeAddBlock() {
        return {256};
    };

    __global__ void Add(const float* x1, const float* x2, float* y, uint32_t size, ActivationFunction af) {
        uint32_t offset = blockIdx.x*256 + threadIdx.x;
        if(offset<size) {
            float result = x1[offset] + x2[offset];
            if(af==ActivationFunction::RELU) {
                y[offset] = result>0?result:0; 
            } else if(af==ActivationFunction::SIGMOID) {
                y[offset] = 1.f/(1.f+exp (-result));
            } else {
                y[offset] = result;
            }
        }
    };
                             
}