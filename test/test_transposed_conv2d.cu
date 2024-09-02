#include "lotus_utils.hpp"
#include "operator/transposed_conv2d.cuh"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"




bool check( xt::xarray<float>& x,
            xt::xarray<float>& kernel,
            bool use_bias, xt::xarray<float>& b,
            float* y,
            uint32_t x_h, uint32_t x_w, uint32_t x_c,
            uint32_t k_num, uint32_t k_c, uint32_t k_h, uint32_t k_w,
            uint32_t stride_h, uint32_t stride_w,
            uint32_t y_c, uint32_t y_h, uint32_t y_w,
            uint32_t padding_h, uint32_t padding_w
           ) 
{
    auto result = xt::xarray<double>::from_shape({y_c, y_h+2*padding_h, y_w+2*padding_w});
    for(uint32_t c=0; c<x_c; ++c) {
        for(uint32_t h=0; h<x_h; ++h) {
            for(uint32_t w=0; w<x_w; ++w) {
                for(uint32_t k=0; k<k_num; ++k) {
                    xt::xarray<float> frag_k = xt::view(kernel,xt::range(c,c+1),xt::range(k,k+1),xt::all(),xt::all());
                    xt::view(result,xt::range(k,k+1),xt::range(h*stride_h,h*stride_h+k_h),xt::range(w*stride_w,w*stride_w+k_w)) += frag_k.reshape({k_h,k_w})*x(c,h,w);
                }
            }
        }
    }

    if(use_bias) {
        for(uint32_t c=0; c<y_c; ++c) {
            xt::view(result, xt::range(c,c+1),xt::all(),xt::all()) += b(c);
        }
    }
   
    for(uint32_t c=0; c<y_c; ++c) {
        for(uint32_t h=0; h<y_h; ++h) {
            for(uint32_t w=0; w<y_w; ++w) {
                if (std::fabs(result(c, h+padding_h, w+padding_w) - y[c*y_h*y_w+h*y_w+w]) / std::fabs(result(c, h+padding_h, w+padding_w)) > 1e-5f) {
                    printf("y[%d][%d][%d] not match, %f vs %f\n", c, h, w, result(c, h+padding_h, w+padding_w), y[c*y_h*y_w+h*y_w+w]);
                    return false;
                }
            }
        }
    }
    return true;
}


int main() {
    using namespace lotus;

    uint32_t x_h = 64;
    uint32_t x_w = 64;
    uint32_t x_c = 16;
    uint32_t k_num = 8;
    uint32_t k_c = x_c;
    uint32_t k_h = 2;
    uint32_t k_w = 2;
    uint32_t stride_h = 3;
    uint32_t stride_w = 3;;
    uint32_t padding_h = 0;
    uint32_t padding_w = 0;

    uint32_t y_c = k_num;

    uint32_t y_h = (x_h-1)*stride_h+k_h-padding_h*2;
    uint32_t y_w = (x_w-1)*stride_w+k_w-padding_w*2;   

    xt::xarray<float> x = xt::random::randint<int>({x_c, x_h, x_w}, -10, 10);
    xt::xarray<float> k = xt::random::randint<int>({k_c, k_num,k_h,k_w}, -10, 10);
    xt::xarray<float> b = xt::random::randint<int>({k_num}, -10, 10);

    float* d_x;
    float* d_k;
    float* d_b;
    float* d_y;
    float* h_y;

    cudaMalloc(&d_x, x_h*x_w*x_c*sizeof(float));
    cudaMalloc(&d_k, k_num*k_h*k_w*k_c*sizeof(float));
    cudaMalloc(&d_b, k_num*sizeof(float));
    cudaMalloc(&d_y, y_c*y_h*y_w*sizeof(float));
    cudaMallocHost(&h_y, y_c*y_h*y_w * sizeof(float));

    cudaMemcpy(d_x, x.data(), x_h*x_w*x_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k.data(), k_num*k_h*k_w*k_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), k_num*sizeof(float), cudaMemcpyHostToDevice);

    StreamPool pool(1);

    TransConv2d<<<MakeTransposedConv2dGrid(y_c, y_h, y_w, padding_h, padding_w), MakeTransposedConv2dBlock(), 0, pool.Stream()>>>(d_x, d_k, true, d_b, d_y, 
                                                                                                                                  k_num, k_c, k_h, k_w, 
                                                                                                                                  x_c, x_h, x_w,
                                                                                                                                  y_c, y_h, y_w,
                                                                                                                                  stride_h, stride_w,
                                                                                                                                  padding_h, padding_w,
                                                                                                                                  ActivationFunction::NONE);

   
    cudaMemcpy(h_y, d_y, y_c*y_h*y_w * sizeof(float), cudaMemcpyDeviceToHost);

    bool chk = check(x, k, true, b, h_y,
                     x_h, x_w, x_c,
                     k_num, k_c, k_h, k_w,
                     stride_h, stride_w,
                     y_c, y_h, y_w,
                     padding_h, padding_w);

    printf("Cube_Y check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_x);
    cudaFree(d_k);
    cudaFree(d_b);
    cudaFree(d_y);
    cudaFreeHost(h_y);

    return 0;

}