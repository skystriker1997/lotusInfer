#include "lotus_utils.hpp"
#include "operator/adaptive_avgpool2d.cuh"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"


bool check( xt::xarray<float>& x, float* y,
            uint32_t x_c, uint32_t x_h, uint32_t x_w,
            uint32_t k_h, uint32_t k_w,
            float stride_h, float stride_w,
            uint32_t y_h, uint32_t y_w
           ) 
{
    for(uint32_t p=0; p<x_c; ++p) {
        for(uint32_t i=0; i<y_h; ++i) {
            for(uint32_t j=0; j<y_w; j++) {
                auto x_frag = xt::view(x, xt::range(p, p+1), xt::range((uint32_t)std::round(i*stride_h), \
                                       (uint32_t)std::round(i*stride_h)+k_h), xt::range((uint32_t)std::round(j*stride_w), (uint32_t)std::round(j*stride_w)+k_w));
                float avg = xt::mean(x_frag)(0);
                if (std::fabs(avg - y[p*y_h*y_w+i*y_w+j]) / std::fabs(avg) > 1e-5f) {
                    printf("y[%d][%d][%d] not match, %f vs %f\n", p, i, j, avg, y[p*y_h*y_w+i*y_w+j]);
                    return false;
                }
            }
        }
    }
    return true;
}


int main() 
{
    using namespace lotus;

    uint32_t x_h = 257;
    uint32_t x_w = 257;
    uint32_t x_c = 3;
    uint32_t y_h = 64;
    uint32_t y_w = 64;

    uint32_t k_h = (x_h+y_h-1)/y_h;
    uint32_t k_w = (x_w+y_w-1)/y_w;
    float stride_h = (x_h-k_h)/y_h;
    float stride_w = (x_w-k_w)/y_w;

    uint32_t y_c = x_c;
   
    xt::random::seed(0);
    xt::xarray<float> x = xt::random::randint<int>({x_c, x_h, x_w}, 0, 100);


    float* d_x;
    float* d_y;
    float* h_y;

    cudaMalloc(&d_x,x_h*x_w*x_c*sizeof(float));
    cudaMalloc(&d_y, y_c*y_h*y_w*sizeof(float));
    cudaMallocHost(&h_y, y_c*y_h*y_w * sizeof(float));

    cudaMemcpy(d_x, x.data(), x_h*x_w*x_c*sizeof(float), cudaMemcpyHostToDevice);

    StreamPool pool(1);

    AdaptiveAvgpool2d<<<MakeAAP2dGrid(y_c, y_h, y_w), MakeAAP2dBlock(), 0, pool.Stream()>>>(d_x, d_y, 
                                                                                              k_h, k_w, 
                                                                                              x_c, x_h, x_w,
                                                                                              stride_h, stride_w,
                                                                                              y_h, y_w,
                                                                                              ActivationFunction::NONE);

    cudaMemcpy(h_y, d_y, y_c*y_h*y_w*sizeof(float), cudaMemcpyDeviceToHost);

    bool chk = check(x,
                     h_y,
                     x_c,x_h, x_w,
                     k_h, k_w,
                     stride_h, stride_w,
                     y_h, y_w);

    printf("Cube_Y check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFreeHost(h_y);

    return 0;
}