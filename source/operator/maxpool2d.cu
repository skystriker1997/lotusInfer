#include "operator/maxpool2d.cuh"



namespace lotus {

    dim3 MakeMP2dGrid(uint32_t output_c, uint32_t output_h, uint32_t output_w) {
        return {(output_w+7)/8, (output_h+7)/8, (output_c+7)/8};
    };

    dim3 MakeMP2dBlock() {
        return {8, 8, 8};
    };


    __global__ void Maxpool2d(const float* input, float* output, 
                               const uint32_t kernel_h, const uint32_t kernel_w, 
                               const uint32_t input_c, const uint32_t padded_input_h, const uint32_t padded_input_w,  
                               const uint32_t padding_h, const uint32_t padding_w, 
                               const uint32_t stride_h, const uint32_t stride_w, 
                               const uint32_t output_h, const uint32_t output_w,
                               ActivationFunction af)
    {

        uint32_t thread_offset_output_y = blockIdx.y*8 + threadIdx.y;
        uint32_t thread_offset_output_x = blockIdx.x*8 + threadIdx.x;
        uint32_t thread_offset_output_z = blockIdx.z*8 + threadIdx.z;

        if(thread_offset_output_y<output_h && thread_offset_output_x<output_w && thread_offset_output_z<input_c) {

            uint32_t thread_offset_input_y = thread_offset_output_y*stride_h;
            uint32_t thread_offset_input_x = thread_offset_output_x*stride_w;
            uint32_t thread_offset_input_z = thread_offset_output_z;

            uint32_t unpadded_input_h = padded_input_h-2*padding_h;
            uint32_t unpadded_input_w = padded_input_w-2*padding_w;

            float max;
        
            for(uint32_t y=0; y<kernel_h; ++y) {
                for(uint32_t x=0; x<kernel_w; ++x) {
                    bool guard = thread_offset_input_y+y>=padding_h && thread_offset_input_y+y<padded_input_h-padding_h && thread_offset_input_x+x>=padding_w && thread_offset_input_x+x<padded_input_w-padding_w;
                    float value;
                    if(guard) {
                        uint32_t true_offset_x = thread_offset_input_x+x-padding_w;
                        uint32_t true_offset_y = thread_offset_input_y+y-padding_h;
                        value = input[thread_offset_input_z*unpadded_input_h*unpadded_input_w + true_offset_y*unpadded_input_w + true_offset_x];
                        if(y==0 && x==0) {
                            max = value;
                        } else {
                            max = value>max?value:max;
                        }  
                    } 
                }
            }

            uint32_t offset = thread_offset_output_z*output_h*output_w+thread_offset_output_y*output_w+thread_offset_output_x;
            if(af == ActivationFunction::RELU) {
                output[offset] = max>0?max:0;
            } else if(af == ActivationFunction::SIGMOID) {
                output[offset] = 1.f/(1.f+exp (-max));
            } else {
                output[offset] = max;
            }
        };
    };

}