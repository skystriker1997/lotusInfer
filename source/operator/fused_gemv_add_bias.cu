#include "operator/fused_gemv_add_bias.cuh"


namespace lotus {

    dim3 MakeSFgemvaGrid(uint32_t weight_h) {
        return {(weight_h+7)/8};
    };

    dim3 MakeSFgemvaBlock() {
        return {128, 8};
    };


    __device__ __forceinline__ void ReduceAdd(float* input_tile, float* weight_tile, float& result) 
    {  
        #pragma unroll
        for(uint32_t i=0; i<4; ++i) {
            weight_tile[threadIdx.x*4+i] *= input_tile[threadIdx.x*4+i];
        }
        __syncthreads();

        #pragma unroll
        for(uint32_t n=128; n>0; n>>=2) {
            if(threadIdx.x<n) {
                if(n==2) {
                    if(threadIdx.x==0) {
                        result += (weight_tile[0] + weight_tile[1] + weight_tile[2] + weight_tile[3] + weight_tile[4] + weight_tile[5] + weight_tile[6] + weight_tile[7]);
                    }
                } else {
                    weight_tile[threadIdx.x] += (weight_tile[threadIdx.x+n]+weight_tile[threadIdx.x+n*2]+weight_tile[threadIdx.x+n*3]);
                }
            }
            __syncthreads();
        }
    }




    __global__ void Fgemva(const float *input, const float *weight, const float* bias, float *output, uint32_t weight_h, uint32_t weight_w, bool use_bias, ActivationFunction af) 
    {
        float result = 0;

        __shared__ float weight_tile[2][8][128*4];
        __shared__ float input_tile[2][128*4];

        uint32_t offset_weight_x = threadIdx.x * 4;
        uint32_t offset_weight_y = blockIdx.x*8 + threadIdx.y;

        uint32_t load_idx = 0;
        uint32_t store_idx = 0;

        auto LoadFromGlobal = [&]() {
            for(uint32_t i=0; i<4; ++i) {
                bool guard = weight_w>(offset_weight_x+i) && weight_h>offset_weight_y;
                if(guard) {
                    ldgsts32(&weight_tile[0][threadIdx.y][threadIdx.x*4+i], &weight[offset_weight_y*weight_w+offset_weight_x+i], 1);
                    ldgsts32(&input_tile[0][threadIdx.x*4+i], input+offset_weight_x+i, 1);
                } else {
                    weight_tile[store_idx][threadIdx.y][threadIdx.x*4+i] = 0;
                    input_tile[store_idx][threadIdx.x*4+i] = 0;
                }
            }
        };
       
        LoadFromGlobal();
        store_idx ^= 1;

        wait();
        __syncthreads();


        for(uint32_t step=0; step<(weight_w+511)/512-1; ++step) {
            offset_weight_x += 512;

            LoadFromGlobal();
            store_idx ^= 1;

            ReduceAdd(&input_tile[load_idx][0], &weight_tile[load_idx][threadIdx.y][0], result);
            load_idx ^= 1;

            wait();
            __syncthreads();
        }

        ReduceAdd(&input_tile[load_idx][0], &weight_tile[load_idx][threadIdx.y][0], result);

        if(threadIdx.x==0) {
            result += use_bias?bias[offset_weight_y]:0.f;
            if(af == ActivationFunction::RELU) {
                output[offset_weight_y] = result>0?result:0;
            } else if(af == ActivationFunction::SIGMOID) {
                output[offset_weight_y] = 1.f/(1.f+exp (-result));
            } else {
                output[offset_weight_y] = result;
            }
        }
    };


}