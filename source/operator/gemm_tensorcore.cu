#include "operator/gemm_tensorcore.cuh"


namespace lotus {

    dim3 MakeGemmGrid(uint32_t m, uint32_t n) 
    {
        return {(n+31)/32, (n+31)/32};
    }


    dim3 MakeGemmBlock() 
    {
        return {64, 2};
    }


    __global__ void Gemm(float const* a, float const* b, bool use_bias, float const* bias, float* c, uint32_t m, uint32_t n, uint32_t k, ActivationFunction af)
    {
        __shared__ __half a_tile[32][32];
        __shared__ float a_tile_buff[32][32];
        __shared__ __half b_tile[32][32];
        __shared__ float b_tile_buff[32][32];
        __shared__ float result[32][32];
        __shared__ float bias_seg[32];

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc_frag;
        nvcuda::wmma::fill_fragment(acc_frag, static_cast<float>(0));

        uint32_t warp_offset_tile_y = threadIdx.y*16;
        uint32_t warp_offset_tile_x = threadIdx.x/32*16;

        uint32_t thread_offset_tile_y = threadIdx.x%32;
        uint32_t thread_offset_tile_x = (threadIdx.x/32+threadIdx.y*2)*8;

        uint32_t thread_offset_a_y = blockIdx.y*32+thread_offset_tile_y;
        uint32_t thread_offset_a_x = thread_offset_tile_x;
        uint32_t thread_offset_b_y = blockIdx.x*32+thread_offset_tile_y; 
        uint32_t thread_offset_b_x = thread_offset_tile_x;

        bool use_128bits_loading = k%4==0;      

        auto LoadFromGlobal = [&]() {
            bool bound_check;
            if(use_128bits_loading) {
                bound_check = thread_offset_a_x<k && thread_offset_a_y<m;
                if(bound_check) {
                    ldgsts128(&a_tile_buff[thread_offset_tile_y][thread_offset_tile_x], a+thread_offset_a_y*k+thread_offset_a_x, 4);
                    bound_check = thread_offset_a_x+4<k;
                    if(bound_check) {
                        ldgsts128(&a_tile_buff[thread_offset_tile_y][thread_offset_tile_x+4], a+thread_offset_a_y*k+thread_offset_a_x+4, 4);
                    } else {
                        #pragma unroll
                        for(uint32_t i=0; i<4; ++i) {
                            a_tile_buff[thread_offset_tile_y][thread_offset_tile_x+4+i] = 0;
                        }
                    }
                } else {
                    #pragma unroll
                    for(uint32_t i=0; i<8; ++i) {
                        a_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i] = 0;
                    }
                }
                bound_check = thread_offset_b_x<k && thread_offset_b_y<n;
                if(bound_check) {
                    ldgsts128(&b_tile_buff[thread_offset_tile_y][thread_offset_tile_x], b+thread_offset_b_y*k+thread_offset_b_x, 4);
                    bound_check = thread_offset_b_x+4<k;
                    if(bound_check) {
                        ldgsts128(&b_tile_buff[thread_offset_tile_y][thread_offset_tile_x+4], b+thread_offset_b_y*k+thread_offset_b_x+4, 4);
                    } else {
                        #pragma unroll
                        for(uint32_t i=0; i<4; ++i) {
                            b_tile_buff[thread_offset_tile_y][thread_offset_tile_x+4+i] = 0;
                        }
                    }
                } else {
                    #pragma unroll
                    for(uint32_t i=0; i<8; ++i) {
                        b_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i] = 0;
                    }
                }

            } else {
                for(uint32_t i=0; i<8; ++i) {
                    bound_check = thread_offset_a_x+i<k && thread_offset_a_y<m;
                    if(bound_check) {
                        ldgsts32(&a_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i], a+thread_offset_a_y*k+thread_offset_a_x+i, 1);
                    } else {
                        a_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i] = 0;
                    }
                    bound_check = thread_offset_b_x+i<k && thread_offset_b_y<n;
                    if(bound_check) {
                        ldgsts32(&b_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i], b+thread_offset_b_y*k+thread_offset_b_x+i, 1);
                    } else {
                        b_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i] = 0;
                    }
                }
            }
        };

        auto SF2HF = [&]() {
            #pragma unroll 
            for(uint32_t i=0; i<8; ++i) {
                a_tile[thread_offset_tile_y][thread_offset_tile_x+i] = __float2half(a_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i]);
                b_tile[thread_offset_tile_y][thread_offset_tile_x+i] = __float2half(b_tile_buff[thread_offset_tile_y][thread_offset_tile_x+i]);
            }
        };

        auto ExeMMA = [&]() {
            nvcuda::wmma::load_matrix_sync(a_frag, &a_tile[warp_offset_tile_y][0], 32);
            nvcuda::wmma::load_matrix_sync(b_frag, &b_tile[warp_offset_tile_x][0], 32);
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            nvcuda::wmma::load_matrix_sync(a_frag, &a_tile[warp_offset_tile_y][16], 32);
            nvcuda::wmma::load_matrix_sync(b_frag, &b_tile[warp_offset_tile_x][16], 32);            
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        };
        if(use_bias) {
            if(threadIdx.y==0 && threadIdx.x/32==0) {
                if(blockIdx.x*32+threadIdx.x<n) {
                    ldgsts32(bias_seg+threadIdx.x, bias+blockIdx.x*32+threadIdx.x, 1);
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

        for(uint32_t step=0; step<(k+31)/32-1; ++step) {
            thread_offset_a_x += 32;
            thread_offset_b_x += 32;

            LoadFromGlobal();
            ExeMMA();

            wait();
            __syncthreads();
            SF2HF();
            __syncthreads();
        }

        ExeMMA();

        nvcuda::wmma::store_matrix_sync(&result[warp_offset_tile_y][warp_offset_tile_x], acc_frag, 32, nvcuda::wmma::mem_row_major);

        if(thread_offset_a_y<m) {
            #pragma unroll 
            for(uint32_t i=0; i<8; ++i) {
                if(blockIdx.x*32+thread_offset_tile_x+i<n) {
                    float tmp = result[thread_offset_tile_y][thread_offset_tile_x+i];
                    tmp += use_bias?bias_seg[thread_offset_tile_x+i]:0;
                    if(af == ActivationFunction::RELU) {
                        tmp = tmp>0?tmp:0;
                    } else if(af == ActivationFunction::SIGMOID) {
                        tmp = 1.f/(1.f+exp (-tmp));
                    } 
                    uint32_t entry = thread_offset_a_y*n+blockIdx.x*32+thread_offset_tile_x+i;
                    c[entry] = tmp;
                }
            }
        }
        
    }

}