#include "operator/fused_gemv_add.cuh"


namespace lotus {


     __device__ __forceinline__ void reduce_add(float* x_tile, float* a_tile, float& result) {
        
        #pragma unroll
        for(int i=0; i<4; ++i) {
            a_tile[threadIdx.x*4+i] *= x_tile[threadIdx.x*4+i];
        }
        __syncthreads();

        #pragma unroll
        for(int n=256; n>0; n>>=2) {
            if(threadIdx.x<n) {
                if(n==1) {
                    result += (a_tile[0] + a_tile[1] + a_tile[2] + a_tile[3]);
                } else {
                    a_tile[threadIdx.x] += (a_tile[threadIdx.x+n]+a_tile[threadIdx.x+n*2]+a_tile[threadIdx.x+n*3]);
                }
            }
            __syncthreads();
        }
    }


    __global__ void sfgemva(const float *x, const float *a, const float* b, float *y, int x_w, int a_h) {

        float result = 0;
        float bias = b[blockIdx.x];

        __shared__ float a_tile[2][256*4];
        __shared__ float x_tile[2][256*4];

        int offset_x = threadIdx.x * 4;

        for(int i=0; i<4; ++i) {
            bool guard = x_w>(offset_x+i);
            if(guard) {
                ldgsts32(&a_tile[0][threadIdx.x*4+i], &a[blockIdx.x*x_w+offset_x+i], 1);
                ldgsts32(&x_tile[0][threadIdx.x*4+i], x+offset_x+i, 1);
            } else {
                a_tile[0][threadIdx.x*4+i] = 0;
                x_tile[0][threadIdx.x*4+i] = 0;
            }
        }
        
        wait();
        __syncthreads();

        int load_idx = 0;
        int store_idx = 1;
        for(int k=0; k<(x_w+1023)/1024-1; ++k) {
            offset_x += 1024;
            for(int i=0; i<4; ++i) {
                bool guard = x_w>(offset_x+i);
                if(guard) {
                    ldgsts32(&a_tile[store_idx][threadIdx.x*4+i], &a[blockIdx.x*x_w+offset_x+i], 1);
                    ldgsts32(&x_tile[store_idx][threadIdx.x*4+i], x+offset_x+i, 1);
                } else {
                    a_tile[store_idx][threadIdx.x*4+i] = 0;
                    x_tile[store_idx][threadIdx.x*4+i] = 0;
                }
            }
            reduce_add(&x_tile[load_idx][0], &a_tile[load_idx][0], result);

            wait();
            __syncthreads();
            load_idx ^= 1;
            store_idx ^= 1;
        }

        reduce_add(&x_tile[load_idx][0], &a_tile[load_idx][0], result);

        if(threadIdx.x==0) {
            y[blockIdx.x] = result + bias;
        }


    };











}