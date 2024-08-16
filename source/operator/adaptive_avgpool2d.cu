#include "operator/adaptive_avgpool2d.cuh"


namespace lotus {
    __global__ void sadaptive_avgpool2d(const float* x, float* y, 
                                        const uint32_t kernel_h, const uint32_t kernel_w, 
                                        const uint32_t x_c, const uint32_t x_h, const uint32_t x_w,  
                                        const float stride_h, const float stride_w, 
                                        const uint32_t y_h, const uint32_t y_w)
    {

        uint32_t thread_offset_y_h = blockIdx.y*8 + threadIdx.y;
        uint32_t thread_offset_y_w = blockIdx.x*8 + threadIdx.x;
        uint32_t thread_offset_y_c = blockIdx.z*8 + threadIdx.z;

        if(thread_offset_y_h<y_h && thread_offset_y_w<y_w && thread_offset_y_c<x_c) {

            uint32_t thread_offset_x_h = roundf(thread_offset_y_h*stride_h);
            uint32_t thread_offset_x_w = roundf(thread_offset_y_w*stride_w);
            uint32_t thread_offset_x_c = thread_offset_y_c;

            float sum = 0;
        
            for(uint32_t i=0; i<kernel_h; ++i) {
                for(uint32_t j=0; j<kernel_w; ++j) {
                    sum += x[thread_offset_x_c*x_h*x_w + (thread_offset_x_h+i)*x_w + thread_offset_x_w+j];
                }
            }
            y[thread_offset_y_c*y_h*y_w+thread_offset_y_h*y_w+thread_offset_y_w] = sum/(float)kernel_h/(float)kernel_w;
        };
    };

}