#include "operator/transposed_conv2d.cuh"



namespace lotus {


    dim3 MakeTransposedConv2dGrid(uint32_t output_c, uint32_t output_h, uint32_t output_w, uint32_t padding_h, uint32_t padding_w) {
        return {(output_w+2*padding_w+15)/16, (output_h+2*padding_h+15)/16, output_c};
    };

    dim3 MakeTransposedConv2dBlock() {
        return {16, 16};
    };


    __global__ void TransConv2d(const float* input, 
                                 const float* k, 
                                 bool use_bias, const float* b, 
                                 float* output, 
                                 const uint32_t k_num, const uint32_t k_c, const uint32_t k_h, const uint32_t k_w, 
                                 const uint32_t input_c, const uint32_t input_h, const uint32_t input_w,    
                                 const uint32_t output_c, const uint32_t output_h, const uint32_t output_w,
                                 const uint32_t stride_h, const uint32_t stride_w,
                                 const uint32_t padding_h, const uint32_t padding_w,
                                 ActivationFunction af) 
    {
        auto Max = [](const float x1, const float x2){return x1>x2?x1:x2;};

        auto Min = [](const float x1, const float x2) {return x1<x2?x1:x2;};

        uint32_t block_offset_output_x = blockIdx.x*16;
        uint32_t block_offset_output_y = blockIdx.y*16;
        uint32_t block_offset_output_z = blockIdx.z;
        uint32_t thread_offset_output_x = block_offset_output_x + threadIdx.x;
        uint32_t thread_offset_output_y = block_offset_output_y + threadIdx.y;
        uint32_t thread_offset_output_z = block_offset_output_z;
        
        uint32_t tmp = block_offset_output_x+1>k_w?block_offset_output_x+1-k_w:0;
        uint32_t block_left_border = tmp%stride_w==0?tmp/stride_w:tmp/stride_w+1;

        tmp = block_offset_output_y+1>k_h?block_offset_output_y+1-k_h:0;
        uint32_t block_top_border = tmp%stride_h==0?tmp/stride_h:tmp/stride_h+1;

        uint32_t thread_left_bound = gridDim.x*16;
        uint32_t thread_right_bound = 0;

        for(uint32_t x=block_left_border; x<block_left_border+16+k_w; ++x) {
            if(x*stride_w<=thread_offset_output_x && x*stride_w+k_w-1>=thread_offset_output_x) {
                thread_left_bound = Min(thread_left_bound, x);
                thread_right_bound = Max(thread_right_bound, x);
            }
        }

        uint32_t thread_top_bound = gridDim.y*16;
        uint32_t thread_bottom_bound = 0;

        for(uint32_t y=block_top_border; y<block_top_border+16+k_h; ++y) {
            if(y*stride_h<=thread_offset_output_y && y*stride_h+k_h-1>=thread_offset_output_y) {
                thread_top_bound = Min(y, thread_top_bound);
                thread_bottom_bound = Max(y, thread_bottom_bound);
            }
        }

        __shared__ float x_frag[2][4][32][32];
        __shared__ float k_frag[2][4][16][16];

        float result=0.f;

        uint32_t thread_ldg_input_y = block_top_border+threadIdx.y*2;
        uint32_t thread_ldg_input_x = block_left_border+threadIdx.x*2;

        uint32_t load_idx = 0;
        uint32_t store_idx = 0;

        auto LoadFromGlobal = [&](uint32_t step) {
            #pragma unroll
            for(uint32_t z=0; z<4; ++z) {
                #pragma unroll
                for(uint32_t y=0; y<2; ++y) {
                    #pragma unroll
                    for(uint32_t x=0; x<2; ++x) {
                        bool x_copy_guard = thread_ldg_input_y+y<input_h && thread_ldg_input_x+x<input_w && z+step*4<input_c;
                        if(x_copy_guard) {
                            ldgsts32(&x_frag[store_idx][z][threadIdx.y*2+y][threadIdx.x*2+x], input+input_h*input_w*(z+step*4)+(thread_ldg_input_y+y)*input_w+thread_ldg_input_x+x, 1);
                        } else {
                            x_frag[store_idx][z][threadIdx.y*2+y][threadIdx.x*2+x] = 0.f;
                        }
                    }
                }
                bool k_copy_guard = threadIdx.x<k_w && threadIdx.y<k_h && z+step*4<k_c;
                if(k_copy_guard) {
                    ldgsts32(&k_frag[store_idx][z][threadIdx.y][threadIdx.x], k+(z+step*4)*k_num*k_h*k_w+blockIdx.z*k_h*k_w+threadIdx.y*k_w+threadIdx.x, 1);
                } else {
                    k_frag[store_idx][z][threadIdx.y][threadIdx.x] = 0.f;
                }  
            }
        };

        auto Compute = [&]() {
            for(uint32_t y=thread_top_bound; y<=thread_bottom_bound; ++y) {
                for(uint32_t x=thread_left_bound; x<=thread_right_bound; ++x) {
                    #pragma unroll
                    for(uint32_t z=0; z<4; ++z) {
                        float input_value = x_frag[load_idx][z][y-block_top_border][x-block_left_border];
                        float kernel_value = k_frag[load_idx][z][thread_offset_output_y-y*stride_h][thread_offset_output_x-x*stride_w];
                        result += input_value*kernel_value;
                    }
                }
            }
        };

        LoadFromGlobal(0);
        store_idx ^= 1;
        wait();
        __syncthreads();

        for(int i=0; i<(input_c+3)/4-1; ++i) {
            LoadFromGlobal(i+1);
            store_idx ^= 1;
            Compute();
            load_idx ^= 1;
            wait();
            __syncthreads();
        }
        
        Compute();

        if(thread_offset_output_y>=padding_h && thread_offset_output_y<output_h+padding_h && thread_offset_output_x>=padding_w && thread_offset_output_x<output_w+padding_w) {
            result += use_bias?b[thread_offset_output_z]:0.f;
            uint32_t offset = thread_offset_output_z*output_h*output_w+(thread_offset_output_y-padding_h)*output_w+thread_offset_output_x-padding_w;
            if(af == ActivationFunction::RELU) {
                output[offset] = result>0?result:0;
            } else if(af == ActivationFunction::SIGMOID) {
                output[offset] = 1.f/(1.f+exp (-result));
            } else {
                output[offset] = result;
            }
        }
    }
                                 
}