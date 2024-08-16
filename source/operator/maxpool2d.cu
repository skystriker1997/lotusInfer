#include "operator/maxpool2d.cuh"


namespace lotus {
    __global__ void smaxpool2d(const float* x, float* y, 
                               const uint32_t kernel_h, const uint32_t kernel_w, 
                               const uint32_t x_c, const uint32_t padded_x_h, const uint32_t padded_x_w,  
                               const uint32_t padding_h, const uint32_t padding_w, 
                               const uint32_t stride_h, const uint32_t stride_w, 
                               const uint32_t y_h, const uint32_t y_w)
    {

        uint32_t thread_offset_y_h = blockIdx.y*8 + threadIdx.y;
        uint32_t thread_offset_y_w = blockIdx.x*8 + threadIdx.x;
        uint32_t thread_offset_y_c = blockIdx.z*8 + threadIdx.z;

        if(thread_offset_y_h<y_h && thread_offset_y_w<y_w && thread_offset_y_c<x_c) {

            uint32_t thread_offset_x_h = thread_offset_y_h*stride_h;
            uint32_t thread_offset_x_w = thread_offset_y_w*stride_w;
            uint32_t thread_offset_x_c = thread_offset_y_c;

            uint32_t unpadded_x_h = padded_x_h-2*padding_h;
            uint32_t unpadded_x_w = padded_x_w-2*padding_w;

            float max;
        
            for(uint32_t i=0; i<kernel_h; ++i) {
                for(uint32_t j=0; j<kernel_w; ++j) {
                    bool guard = thread_offset_x_h+i>=padding_h && thread_offset_x_h+i<padded_x_h-padding_h && thread_offset_x_w+j>=padding_w && thread_offset_x_w+j<padded_x_w-padding_w;
                    float target;
                    if(guard) {
                        uint32_t true_offset_w = thread_offset_x_w+j-padding_w;
                        uint32_t true_offset_h = thread_offset_x_h+i-padding_h;
                        target = x[thread_offset_x_c*unpadded_x_h*unpadded_x_w + true_offset_h*unpadded_x_w + true_offset_w];
                    } else {
                        target = 0;
                    }
                    if(i==0 && j==0) {
                        max = target;
                    } else {
                        max = target>max?target:max;
                    }
                }
            }

            y[thread_offset_y_c*y_h*y_w+thread_offset_y_h*y_w+thread_offset_y_w] = max;
        };
    };

}