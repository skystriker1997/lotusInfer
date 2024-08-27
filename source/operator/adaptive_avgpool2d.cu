#include "operator/adaptive_avgpool2d.cuh"


namespace lotus {

    dim3 MakeAAP2dGrid(uint32_t output_c, uint32_t output_h, uint32_t output_w) {
        return {(output_w+7)/8, (output_h+7)/8, (output_c+7)/8};
    };

    dim3 MakeAAP2dBlock() {
        return {8,8,8};
    };


    __global__ void sadaptive_avgpool2d(const float* input, float* output, 
                                        const uint32_t kernel_h, const uint32_t kernel_w, 
                                        const uint32_t input_c, const uint32_t input_h, const uint32_t input_w,  
                                        const float stride_h, const float stride_w, 
                                        const uint32_t output_h, const uint32_t output_w)
    {

        uint32_t thread_offset_output_y = blockIdx.y*8 + threadIdx.y;
        uint32_t thread_offset_output_x = blockIdx.x*8 + threadIdx.x;
        uint32_t thread_offset_output_z = blockIdx.z*8 + threadIdx.z;

        if(thread_offset_output_y<output_h && thread_offset_output_x<output_w && thread_offset_output_z<input_c) {

            uint32_t thread_offset_input_y = roundf(thread_offset_output_y*stride_h);
            uint32_t thread_offset_input_x = roundf(thread_offset_output_x*stride_w);
            uint32_t thread_offset_input_z = thread_offset_output_z;

            float sum = 0;
        
            for(uint32_t y=0; y<kernel_h; ++y) {
                for(uint32_t x=0; x<kernel_w; ++x) {
                    sum += input[thread_offset_input_z*input_h*input_w + (thread_offset_input_y+y)*input_w + thread_offset_input_x+x];
                }
            }
            output[thread_offset_output_z*output_h*output_w+thread_offset_output_y*output_w+thread_offset_output_x] = sum/(float)kernel_h/(float)kernel_w;
        };
    };

}