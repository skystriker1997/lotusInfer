#include "operator/fused_gemv_add_bias.cuh"


namespace lotus {


     __device__ __forceinline__ void reduce_add(float* x_tile, float* a_tile, float& result) 
    {
        #pragma unroll
        for(uint32_t i=0; i<4; ++i) {
            a_tile[threadIdx.x*4+i] *= x_tile[threadIdx.x*4+i];
        }
        __syncthreads();

        #pragma unroll
        for(uint32_t n=128; n>0; n>>=2) {
            if(threadIdx.x<n) {
                if(n==2) {
                    if(threadIdx.x==0) {
                        result += (a_tile[0] + a_tile[1] + a_tile[2] + a_tile[3] + a_tile[4] + a_tile[5] + a_tile[6] + a_tile[7]);
                    }
                } else {
                    a_tile[threadIdx.x] += (a_tile[threadIdx.x+n]+a_tile[threadIdx.x+n*2]+a_tile[threadIdx.x+n*3]);
                }
            __syncthreads();
            }
        }
    }



    __global__ void sfgemva(const float *x, const float *a, const float* b, float *y, uint32_t a_w, uint32_t a_h, bool use_bias, ActivationFunction af) 
    {
        float result = 0;

        __shared__ float a_tile[2][8][128*4];
        __shared__ float x_tile[2][128*4];

        uint32_t offset_a_w = threadIdx.x * 4;
        uint32_t offset_a_h = blockIdx.x*8 + threadIdx.y;

        for(uint32_t i=0; i<4; ++i) {
            bool guard = a_w>(offset_a_w+i) && a_h>offset_a_h;

            if(guard) {
                ldgsts32(&a_tile[0][threadIdx.y][threadIdx.x*4+i], &a[offset_a_h*a_w+offset_a_w+i], 1);
                ldgsts32(&x_tile[0][threadIdx.x*4+i], x+offset_a_w+i, 1);
            } else {
                a_tile[0][threadIdx.y][threadIdx.x*4+i] = 0;
                x_tile[0][threadIdx.x*4+i] = 0;
            }
        }
        
        wait();
        __syncthreads();

        uint32_t load_idx = 0;
        uint32_t store_idx = 1;
        for(uint32_t k=0; k<(a_w+511)/512-1; ++k) {
            offset_a_w += 512;
            for(uint32_t i=0; i<4; ++i) {
                bool guard = a_w>(offset_a_w+i) && a_h>offset_a_h;
                if(guard) {
                    ldgsts32(&a_tile[store_idx][threadIdx.y][threadIdx.x*4+i], &a[offset_a_h*a_w+offset_a_w+i], 1);
                    ldgsts32(&x_tile[store_idx][threadIdx.x*4+i], x+offset_a_w+i, 1);
                } else {
                    a_tile[store_idx][threadIdx.y][threadIdx.x*4+i] = 0;
                    x_tile[store_idx][threadIdx.x*4+i] = 0;
                }
            }
            reduce_add(&x_tile[load_idx][0], &a_tile[load_idx][threadIdx.y][0], result);

            wait();
            __syncthreads();
            load_idx ^= 1;
            store_idx ^= 1;
        }

        reduce_add(&x_tile[load_idx][0], &a_tile[load_idx][threadIdx.y][0], result);

        if(threadIdx.x==0) {
            float tmp = result + (use_bias?b[offset_a_h]:0);
            if(af == ActivationFunction::RELU) {
                y[offset_a_h] = tmp>0?tmp:0;
            } else {
                y[offset_a_h] = tmp;
            }
        }
    };


}