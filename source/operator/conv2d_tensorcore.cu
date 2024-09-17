#include "operator/conv2d_tensorcore.cuh"



namespace lotus {


    dim3 MakeConv2dGrid(uint32_t output_c, uint32_t output_h, uint32_t output_w) {
        return {(output_h*output_w+31)/32, (output_c+31)/32};
    };


    dim3 MakeConv2dBlock() {
        return {64, 2};
    };

   
    __global__ void Conv2d( const float* input, 
                            const float* kernel, 
                            bool use_bias, const float* bias, 
                            float* output, 
                            const uint32_t kernel_num, const uint32_t kernel_c, const uint32_t kernel_h, const uint32_t kernel_w, 
                            const uint32_t input_c, const uint32_t padded_input_h, const uint32_t padded_input_w,    
                            const uint32_t output_c, const uint32_t output_h, const uint32_t output_w,
                            const uint32_t stride_h, const uint32_t stride_w,
                            const uint32_t padding_h, const uint32_t padding_w,
                            ActivationFunction af)
    {
        __shared__ __half kernel_tile[32][32];
        __shared__ float kernel_tile_buff[32][32];
        __shared__ __half input_tile[32][32];
        __shared__ float input_tile_buff[32][32];
        __shared__ float result[32][32];
        __shared__ float bias_seg[32];

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> kernel_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> input_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
        nvcuda::wmma::fill_fragment(acc_frag, static_cast<float>(0));

        uint32_t warp_offset_tile_y = threadIdx.y*16;
        uint32_t warp_offset_tile_x = threadIdx.x/32*16;

        uint32_t thread_offset_tile_y = threadIdx.x%32;
        uint32_t thread_offset_tile_x = (threadIdx.x/32+threadIdx.y*2)*8;

        uint32_t thread_offset_kernel_y = blockIdx.y*32+thread_offset_tile_y;
        uint32_t thread_offset_kernel_x = thread_offset_tile_x;
        uint32_t thread_offset_input_y = thread_offset_tile_y;
        uint32_t thread_offset_input_x = blockIdx.x*32+thread_offset_tile_x;       

        uint32_t channel_size = kernel_h * kernel_w;
        uint32_t kernel_size = kernel_c * channel_size;

        uint32_t unpadded_input_w = padded_input_w - 2*padding_w;
        uint32_t unpadded_input_h = padded_input_h - 2*padding_h;

        auto LoadFromGlobal = [&]() {
            bool bound_check;
            #pragma unroll
            for(uint32_t i=0; i<8; ++i) {
                bound_check = thread_offset_kernel_y<kernel_num && thread_offset_kernel_x+i<kernel_size;
                if(bound_check) {
                    ldgsts32(&kernel_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i], kernel+thread_offset_kernel_y*kernel_size+thread_offset_kernel_x+i, 1);
                } else {
                    kernel_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i] = 0;
                }
            }
            #pragma unroll
            for(uint32_t i=0; i<8; ++i) {
                uint32_t channel_idx = thread_offset_input_y / channel_size;
                uint32_t row_idx_in_window = (thread_offset_input_y-channel_idx*channel_size) / kernel_w;
                uint32_t col_idx_in_window = thread_offset_input_y-channel_idx*channel_size-row_idx_in_window*kernel_w;
                uint32_t row_idx = (thread_offset_input_x+i)/output_w*stride_h+row_idx_in_window;
                uint32_t col_idx = (thread_offset_input_x+i)%output_w*stride_w + col_idx_in_window;
                bound_check = thread_offset_input_x+i<output_h*output_w && thread_offset_input_y<kernel_size && (row_idx>=padding_h && row_idx<padded_input_h-padding_h) && (col_idx>=padding_w && col_idx<padded_input_w-padding_w);
                if(bound_check) {
                    ldgsts32(&input_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i], input+(row_idx-padding_h)*unpadded_input_w+(col_idx-padding_w)+channel_idx*unpadded_input_h*unpadded_input_w, 1);
                } else {
                    input_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i] = 0;
                } 
            }
        };

        auto SF2HF = [&]() {
            #pragma unroll 
            for(uint32_t i=0; i<8; ++i) {
                kernel_tile[thread_offset_tile_y][thread_offset_tile_x+i] = __float2half(kernel_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i]);
                input_tile[thread_offset_tile_x+i][thread_offset_tile_y] = __float2half(input_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i]);
            }
        };

        auto ExeMMA = [&]() {
            nvcuda::wmma::load_matrix_sync(kernel_frag, &kernel_tile[warp_offset_tile_y][0], 32);
            nvcuda::wmma::load_matrix_sync(input_frag, &input_tile[warp_offset_tile_x][0], 32);
            nvcuda::wmma::mma_sync(acc_frag, kernel_frag, input_frag, acc_frag);
            nvcuda::wmma::load_matrix_sync(kernel_frag, &kernel_tile[warp_offset_tile_y][16], 32);
            nvcuda::wmma::load_matrix_sync(input_frag, &input_tile[warp_offset_tile_x][16], 32);            
            nvcuda::wmma::mma_sync(acc_frag, kernel_frag, input_frag, acc_frag);
        };
        if(use_bias) {
            if(threadIdx.y==0 && threadIdx.x/32==0) {
                if(thread_offset_kernel_y<kernel_num) {
                    ldgsts32(bias_seg+threadIdx.x, bias+thread_offset_kernel_y, 1);
                } else {
                    bias_seg[threadIdx.x] = 0;
                }
            }
        }

        LoadFromGlobal();
        wait();
        __syncthreads();
        SF2HF();
        __syncthreads();

        for(uint32_t step=0; step<(kernel_size+31)/32-1; ++step) {
            thread_offset_kernel_x += 32;
            thread_offset_input_y += 32;

            LoadFromGlobal();
            ExeMMA();

            wait();
            __syncthreads();
            SF2HF();
            __syncthreads();
        }

        ExeMMA();

        nvcuda::wmma::store_matrix_sync(&result[warp_offset_tile_y][warp_offset_tile_x], acc_frag, 32, nvcuda::wmma::mem_row_major);

        if(thread_offset_kernel_y<kernel_num) {
            #pragma unroll 
            for(uint32_t i=0; i<8; ++i) {
                if(thread_offset_input_x+i<output_h*output_w) {
                    float tmp = result[thread_offset_tile_y][thread_offset_tile_x+i];
                    tmp += use_bias?bias_seg[thread_offset_tile_y]:0;
                    if(af == ActivationFunction::RELU) {
                        tmp = tmp>0?tmp:0;
                    } else if(af == ActivationFunction::SIGMOID) {
                        tmp = 1.f/(1.f+exp (-tmp));
                    } 
                    uint32_t entry = thread_offset_kernel_y*(output_h*output_w)+thread_offset_input_x+i;
                    output[entry] = tmp;
                }
            }
        }
        

    }


}