#include "operator/conv2d.cuh"

namespace lotus {

    /*
    *
    * matrix a, b, and c are row-major
    * 
    * -----------------------------------------------------------------------------------------------------------------------------------------------
    * 
    * tile map:
    * 
    *                                          b_tile                            
    *                                                                               128 floats                    
    *                                         -|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-
    *                                 8 floats |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
    *                                         -|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-
    * 
    *                           8 floats                     
    *                          -|-----|-     --|-----|-----|-----|-----|-----|-----|-----|-----|-----------------------------------------------|-
    *                           |     |        | t0  |  t1 |  t2 |  t3 | t4  | t5  | t6  | t7  |                                               |
    *                           |-----|       -|-----|-----|-----|-----|-----|-----|-----|-----|                                               |
    *               a_tile      |     |        | t8  |  t9 | t10 | t11 | t12 | t13 | t14 | t15 |                                               |
    *                           |-----|       -|-----|-----|----- warp0 -----|-----|-----|-----|                 warp1                         |
    *                           |     |        | t16 | t17 | t18 | t19 | t20 | t21 | t22 | t23 |                                               |
    *                           |-----|       -|-----|-----|-----|-----|-----|-----|-----|-----|                                               |
    *                           |     |        | t24 | t25 | t26 | t27 | t28 | t29 | t30 | t31 |                                               |
    *                           |-----|       -|-----|-----|-----|-----|-----|-----|-----|-----|-----------------------------------------------|-
    *                           |     |        |                                               |                                               |
    *                           |-----|       -|                                               |                                               |
    *                           |     |        |                                               |                                               |
    *                           |-----|       -|                 warp2                         |                 warp3                         |
    *                           |     |        |                                               |                                               |
    *                           |-----|       -|                                               |                                               |     
    *                           |     |        |                                               |                                               |
    *              128 floats   |-----|       -|-------------------------------------------- block --------------------------------------------|-
    *                           |     |        |                                               |                                               |
    *                           |-----|       -|                                               |                                               |
    *                           |     |        |                                               |                                               |
    *                           |-----|       -|                 warp4                         |                 warp5                         |
    *                           |     |        |                                               |                                               |
    *                           |-----|       -|                                               |                                               |
    *                           |     |        |                                               |                                               |
    *                           |-----|       -|-----------------------------------------------|-----------------------------------------------|-
    *                           |     |        |                                               |                                               |
    *                           |-----|       -|                                               |                                               |
    *                           |     |        |                                               |                                               |
    *                           |-----|       -|                                               |                                               |
    *                           |     |        |                 warp6                         |                 warp7                         |
    *                           |-----|       -|                                               |                                               |
    *                           |     |        |                                               |                                               |
    *                          -|-----|-      -|-----------------------------------------------|-----------------------------------------------|-
    * 
    * --------------------------------------------------------------------------------------------------------------------------------------------------
    */



    __global__ void sconv2d( const float* x, 
                            const float* k, 
                            const float* b, 
                            float* y, 
                            const int k_num, const int k_h, const int k_w, const int k_c, 
                            const int x_w, const int x_h, const int x_c, 
                            const int y_w, const int y_h, const int y_c,
                            const int stride_h, const int stride_w,
                            const int padding_h, const int padding_w
                        )
    {
        __shared__  __align__(16 * 1024) float k_smem[2][128][8];
        __shared__  __align__(16 * 1024) float x_smem[2][8][128];

        float k_frag[2][8];
        float x_frag[2][8];

        float y_frag[8][8] = {0};


        int warp_idx = threadIdx.x / 32;

        int thread_idx_in_warp_w = (threadIdx.x%32) % 8;
        int thread_idx_in_warp_h = (threadIdx.x%32) / 8;

        int block_offset_y_w = blockIdx.x*128;
        int block_offset_y_h = blockIdx.y*128;

        int thread_offset_blocktile_w = (warp_idx%2)*64 + thread_idx_in_warp_w*8;
        int thread_offset_blocktile_h = (warp_idx/2)*32 + thread_idx_in_warp_h*8;

        int thread_offset_y_w = block_offset_y_w + thread_offset_blocktile_w;
        int thread_offset_y_h = block_offset_y_h + thread_offset_blocktile_h;

        int thread_offset_sts_k_w = (threadIdx.x%2)*4;
        int thread_offset_sts_k_h = threadIdx.x/2;

        int thread_offset_k_h = block_offset_y_h+thread_offset_sts_k_h;
        int thread_offset_k_w = thread_offset_sts_k_w;

        int thread_offset_sts_x_w = (threadIdx.x%32)*4;
        int thread_offset_sts_x_h = threadIdx.x/32;

        int thread_offset_x_w = block_offset_y_w+thread_offset_sts_x_w;
        int thread_offset_x_h = thread_offset_sts_x_h;


        int channel_size = k_h * k_w;
        int kernel_size = k_c * channel_size;

        int unpadded_x_w = x_w - 2*padding_w;
        int unpadded_x_h = x_h - 2*padding_h;

        #pragma unroll
        for(int i=0; i<4; ++i) {
           
            bool k_guard = thread_offset_k_w+i<kernel_size && thread_offset_k_h<k_num;
            if(k_guard) {
                ldgsts32(&k_smem[0][thread_offset_sts_k_h][thread_offset_sts_k_w+i], k+thread_offset_k_h*kernel_size+thread_offset_k_w+i, 1);
            } else {
                k_smem[0][thread_offset_sts_k_h][thread_offset_sts_k_w+i] = 0.f;
            }
        }


        #pragma unroll
        for(int i=0; i<4; ++i) {

            int channel_idx = thread_offset_x_h / channel_size;
            int row_idx_in_window = (thread_offset_x_h-channel_idx*channel_size) / k_w;
            int col_idx_in_window = thread_offset_x_h-channel_idx*channel_size-row_idx_in_window*k_w;
            int row_idx = (thread_offset_x_w+i)/y_w*stride_h+row_idx_in_window;
            int col_idx = (thread_offset_x_w+i)%y_w*stride_w + col_idx_in_window;

            bool x_guard = thread_offset_x_w+i<y_h*y_w && thread_offset_x_h<kernel_size && (row_idx>=padding_h && row_idx<x_h-padding_h) && (col_idx>=padding_w && col_idx<x_w-padding_w);

            if(x_guard) {
                ldgsts32(&x_smem[0][thread_offset_sts_x_h][thread_offset_sts_x_w+i], x+(row_idx-padding_h)*unpadded_x_w+(col_idx-padding_w)+channel_idx*unpadded_x_h*unpadded_x_w, 1);
            } else {
                x_smem[0][thread_offset_sts_x_h][thread_offset_sts_x_w+i] = 0.f;
            } 
        }

        wait();

        __syncthreads();

        int smem_load_idx = 0;
        int smem_store_idx = 1;
       

        for(int k_step=0; k_step<(kernel_size+7)/8-1; ++k_step) {
            thread_offset_k_w += 8;
            thread_offset_x_h += 8;

            #pragma unroll
            for(int i=0; i<4; ++i) {
            
                bool k_guard = thread_offset_k_w+i<kernel_size && thread_offset_k_h<k_num;
                if(k_guard) {
                    ldgsts32(&k_smem[smem_store_idx][thread_offset_sts_k_h][thread_offset_sts_k_w+i], k+thread_offset_k_h*kernel_size+thread_offset_k_w+i, 1);
                } else {
                    k_smem[smem_store_idx][thread_offset_sts_k_h][thread_offset_sts_k_w+i] = 0.f;
                }
            }

            #pragma unroll
            for(int i=0; i<4; ++i) {

                int channel_idx = thread_offset_x_h / channel_size;
                int row_idx_in_window = (thread_offset_x_h-channel_idx*channel_size) / k_w;
                int col_idx_in_window = thread_offset_x_h-channel_idx*channel_size-row_idx_in_window*k_w;
                int row_idx = (thread_offset_x_w+i)/y_w*stride_h+row_idx_in_window;
                int col_idx = (thread_offset_x_w+i)%y_w*stride_w + col_idx_in_window;

                bool x_guard = thread_offset_x_w+i<y_h*y_w && thread_offset_x_h<kernel_size && (row_idx>=padding_h && row_idx<x_h-padding_h) && (col_idx>=padding_w && col_idx<x_w-padding_w);

                if(x_guard) {
                    ldgsts32(&x_smem[smem_store_idx][thread_offset_sts_x_h][thread_offset_sts_x_w+i], x+(row_idx-padding_h)*unpadded_x_w+(col_idx-padding_w)+channel_idx*unpadded_x_h*unpadded_x_w, 1);
                } else {
                    x_smem[smem_store_idx][thread_offset_sts_x_h][thread_offset_sts_x_w+i] = 0.f;
                } 
            }


            #pragma unroll
            for(int i=0; i<8; ++i) {
                k_frag[0][i] = k_smem[smem_load_idx][thread_offset_blocktile_h+i][0];
                x_frag[0][i] = x_smem[smem_load_idx][0][thread_offset_blocktile_w+i];
            }

            int frag_load_idx = 0;
            int frag_store_idx = 1;

            #pragma unroll
            for(int i=0; i<7; ++i) {
                #pragma unroll
                for(int j=0; j<8; ++j) {
                    k_frag[frag_store_idx][j] = k_smem[smem_load_idx][thread_offset_blocktile_h+j][i+1];
                    x_frag[frag_store_idx][j] = x_smem[smem_load_idx][i+1][thread_offset_blocktile_w+j];
                }
                #pragma unroll
                for(int h=0; h<8; ++h) {
                    for(int w=0; w<8; ++w) {
                        y_frag[h][w] += k_frag[frag_load_idx][h]*x_frag[frag_load_idx][w];
                    }
                }
                frag_load_idx ^= 1;
                frag_store_idx ^= 1;
            }
            #pragma unroll
            for(int h=0; h<8; ++h) {
                for(int w=0; w<8; ++w) {
                    y_frag[h][w] += k_frag[frag_load_idx][h]*x_frag[frag_load_idx][w];
                }
            }

            wait();
            __syncthreads();

            smem_load_idx ^= 1;
            smem_store_idx ^= 1;
        }

        #pragma unroll
        for(int i=0; i<8; ++i) {
            k_frag[0][i] = k_smem[smem_load_idx][thread_offset_blocktile_h+i][0];
            x_frag[0][i] = x_smem[smem_load_idx][0][thread_offset_blocktile_w+i];
        }

        int frag_load_idx = 0;
        int frag_store_idx = 1;

        #pragma unroll
        for(int i=0; i<7; ++i) {
            #pragma unroll
            for(int j=0; j<8; ++j) {
                k_frag[frag_store_idx][j] = k_smem[smem_load_idx][thread_offset_blocktile_h+j][i+1];
                x_frag[frag_store_idx][j] = x_smem[smem_load_idx][i+1][thread_offset_blocktile_w+j];
            }
            #pragma unroll
            for(int h=0; h<8; ++h) {
                for(int w=0; w<8; ++w) {
                    y_frag[h][w] += k_frag[frag_load_idx][h]*x_frag[frag_load_idx][w];
                }
            }
            frag_load_idx ^= 1;
            frag_store_idx ^= 1;
        }
        #pragma unroll
        for(int h=0; h<8; ++h) {
            for(int w=0; w<8; ++w) {
                y_frag[h][w] += k_frag[frag_load_idx][h]*x_frag[frag_load_idx][w];
            }
        }

        #pragma unroll
        for(int h=0; h<8; ++h) {
            for(int w=0; w<8; ++w) {
                int i = thread_offset_y_h+h;
                int j = thread_offset_y_w+w;
                if(i<k_num && j<y_h*y_w) {
                    y[i*(y_h*y_w)+j] = y_frag[h][w]+b[i];
                }
            }
        }
    }


}