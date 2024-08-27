#include "operator/conv2d.cuh"

namespace lotus {


    dim3 MakeConv2dGrid(uint32_t output_c, uint32_t output_h, uint32_t output_w) {
        return {(output_h*output_w+127)/128, (output_c+127)/128};
    };


    dim3 MakeConv2dBlock() {
        return {256};
    };


    __global__ void sconv2d(const float* input, 
                            const float* k, 
                            bool use_bias, const float* b, 
                            float* output, 
                            const uint32_t k_num, const uint32_t k_c, const uint32_t k_h, const uint32_t k_w, 
                            const uint32_t input_c, const uint32_t padded_input_h, const uint32_t padded_input_w,    
                            const uint32_t output_c, const uint32_t output_h, const uint32_t output_w,
                            const uint32_t stride_h, const uint32_t stride_w,
                            const uint32_t padding_h, const uint32_t padding_w,
                            ActivationFunction af
                            )
    {
        __shared__ float k_smem[2][128][8];
        __shared__ float input_smem[2][8][128];

        float k_frag[2][8];
        float input_frag[2][8];

        float output_frag[8][8] = {0};

        uint32_t warp_idx = threadIdx.x / 32;

        uint32_t thread_idx_in_warp_x = (threadIdx.x%32) % 8;
        uint32_t thread_idx_in_warp_y = (threadIdx.x%32) / 8;

        uint32_t block_offset_output_x = blockIdx.x*128;
        uint32_t block_offset_output_y = blockIdx.y*128;

        uint32_t thread_offset_blocktile_x = (warp_idx%2)*64 + thread_idx_in_warp_x*8;
        uint32_t thread_offset_blocktile_y = (warp_idx/2)*32 + thread_idx_in_warp_y*8;

        uint32_t thread_offset_output_x = block_offset_output_x + thread_offset_blocktile_x;
        uint32_t thread_offset_output_y = block_offset_output_y + thread_offset_blocktile_y;

        uint32_t thread_offset_sts_k_x = (threadIdx.x%2)*4;
        uint32_t thread_offset_sts_k_y = threadIdx.x/2;

        uint32_t thread_offset_k_y = block_offset_output_y+thread_offset_sts_k_y;
        uint32_t thread_offset_k_x = thread_offset_sts_k_x;

        uint32_t thread_offset_sts_input_x = (threadIdx.x%32)*4;
        uint32_t thread_offset_sts_input_y = threadIdx.x/32;

        uint32_t thread_offset_input_x = block_offset_output_x+thread_offset_sts_input_x;
        uint32_t thread_offset_input_y = thread_offset_sts_input_y;


        uint32_t channel_size = k_h * k_w;
        uint32_t kernel_size = k_c * channel_size;

        uint32_t unpadded_input_w = padded_input_w - 2*padding_w;
        uint32_t unpadded_input_h = padded_input_h - 2*padding_h;

        uint32_t smem_load_idx = 0;
        uint32_t smem_store_idx = 0;
        uint32_t frag_load_idx = 0;
        uint32_t frag_store_idx = 0;

        auto LoadFromGlobal = [&]() {
            #pragma unroll
            for(uint32_t i=0; i<4; ++i) {
                bool k_guard = thread_offset_k_x+i<kernel_size && thread_offset_k_y<k_num;
                if(k_guard) {
                    ldgsts32(&k_smem[smem_store_idx][thread_offset_sts_k_y][thread_offset_sts_k_x+i], k+thread_offset_k_y*kernel_size+thread_offset_k_x+i, 1);
                } else {
                    k_smem[smem_store_idx][thread_offset_sts_k_y][thread_offset_sts_k_x+i] = 0.f;
                }
            }
            #pragma unroll
            for(uint32_t i=0; i<4; ++i) {
                uint32_t channel_idx = thread_offset_input_y / channel_size;
                uint32_t row_idx_in_window = (thread_offset_input_y-channel_idx*channel_size) / k_w;
                uint32_t col_idx_in_window = thread_offset_input_y-channel_idx*channel_size-row_idx_in_window*k_w;
                uint32_t row_idx = (thread_offset_input_x+i)/output_w*stride_h+row_idx_in_window;
                uint32_t col_idx = (thread_offset_input_x+i)%output_w*stride_w + col_idx_in_window;

                bool input_guard = thread_offset_input_x+i<output_h*output_w && thread_offset_input_y<kernel_size && (row_idx>=padding_h && row_idx<padded_input_h-padding_h) && (col_idx>=padding_w && col_idx<padded_input_w-padding_w);

                if(input_guard) {
                    ldgsts32(&input_smem[smem_store_idx][thread_offset_sts_input_y][thread_offset_sts_input_x+i], input+(row_idx-padding_h)*unpadded_input_w+(col_idx-padding_w)+channel_idx*unpadded_input_h*unpadded_input_w, 1);
                } else {
                    input_smem[smem_store_idx][thread_offset_sts_input_y][thread_offset_sts_input_x+i] = 0.f;
                } 
            }
        };

        auto LoadFromSmem = [&](uint32_t i_) {
            #pragma unroll
            for(uint32_t j=0; j<8; ++j) {
                k_frag[frag_store_idx][j] = k_smem[smem_load_idx][thread_offset_blocktile_y+j][i_];
                input_frag[frag_store_idx][j] = input_smem[smem_load_idx][i_][thread_offset_blocktile_x+j];
            }
        };

        auto ComputeThreadTile = [&]() {
            #pragma unroll
            for(uint32_t y=0; y<8; ++y) {
                for(uint32_t x=0; x<8; ++x) {
                    output_frag[y][x] += k_frag[frag_load_idx][y]*input_frag[frag_load_idx][x];
                }
            }
        };

        LoadFromGlobal();
        smem_store_idx ^= 1;

        wait();
        __syncthreads();

        for(uint32_t step=0; step<(kernel_size+7)/8-1; ++step) {
            thread_offset_k_x += 8;
            thread_offset_input_y += 8;

            LoadFromGlobal();
            smem_store_idx ^= 1;

            LoadFromSmem(0);
            frag_store_idx ^= 1;

            #pragma unroll
            for(uint32_t i=0; i<7; ++i) {
                LoadFromSmem(i+1);
                frag_store_idx ^= 1;
                
                ComputeThreadTile();
                frag_load_idx ^= 1;
            }

            ComputeThreadTile();
            frag_load_idx ^= 1;

            smem_load_idx ^= 1;

            wait();
            __syncthreads();
        }

        LoadFromSmem(0);
        frag_store_idx ^= 1;

        #pragma unroll
        for(uint32_t i=0; i<7; ++i) {
            LoadFromSmem(i+1);
            frag_store_idx ^= 1;
            
            ComputeThreadTile();
            frag_load_idx ^= 1;
        }

        ComputeThreadTile();

        #pragma unroll
        for(uint32_t y=0; y<8; ++y) {
            for(uint32_t x=0; x<8; ++x) {
                uint32_t entry_y = thread_offset_output_y+y;
                uint32_t entry_x = thread_offset_output_x+x;
                if(entry_y<k_num && entry_x<output_h*output_w) {
                    float tmp = output_frag[y][x]+(use_bias?b[entry_y]:0);
                    if(af == ActivationFunction::RELU) {
                        output[entry_y*(output_h*output_w)+entry_x] = tmp>0?tmp:0;
                    } else {
                        output[entry_y*(output_h*output_w)+entry_x] = tmp;
                    }
                }
            }
        }

    }


}