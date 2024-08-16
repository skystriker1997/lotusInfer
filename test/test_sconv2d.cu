#include "operator/conv2d.cuh"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"



bool check( xt::xarray<float>& x,
            xt::xarray<float>& k,
            xt::xarray<float>& b,
            float* y,
            uint32_t padded_x_h,
            uint32_t padded_x_w,
            uint32_t x_c,
            uint32_t k_num,
            uint32_t k_c,
            uint32_t k_h,
            uint32_t k_w,
            uint32_t stride_h,
            uint32_t stride_w,
            uint32_t y_c,
            uint32_t y_h,
            uint32_t y_w,
            uint32_t padding_h,
            uint32_t padding_w
           ) 
{

    auto sum = [](const float& left, const float& right){return left + right;};
    xt::xarray<float> padded_x = xt::zeros<float>({x_c, padded_x_h, padded_x_w});
    xt::view(padded_x, xt::all(), xt::range(padding_h, padded_x_h-padding_h), xt::range(padding_w, padded_x_w-padding_w)) = x;

    for(uint32_t i=0; i<y_h; i++) {
        for(uint32_t j=0; j<y_w; j++) {
            auto x_frag = xt::view(padded_x, xt::all(), xt::range(i*stride_h, i*stride_h+k_h), xt::range(j*stride_w, j*stride_w+k_w));
            for(uint32_t q=0; q<k_num; ++q) {
                auto k_frag = xt::view(k, xt::range(q, q+1), xt::all(), xt::all(), xt::all());
                auto product = k_frag * x_frag;
                float result = xt::reduce(sum, product, {0,1,2,3})(0) + b(q);
                uint32_t target = q*(y_h*y_w) + i*y_w + j;
                if (std::fabs(result - y[target]) / std::fabs(result) > 1e-5f) {
                    printf("y[%d][%d][%d] not match, %f vs %f\n", q, i, j, result, y[target]);
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

    uint32_t padded_x_h = 257;
    uint32_t padded_x_w = 257;
    uint32_t x_c = 3;
    uint32_t k_num = 32;
    uint32_t k_c = 3;
    uint32_t k_h = 9;
    uint32_t k_w = 9;
    uint32_t stride_h = 2;
    uint32_t stride_w = 2;;
    uint32_t padding_h = 4;
    uint32_t padding_w = 4;

    uint32_t y_c = k_num;

    uint32_t y_h = (padded_x_h-k_h)/stride_h + 1;
    uint32_t y_w = (padded_x_w-k_w)/stride_w + 1;   

    uint32_t unpadded_x_h = padded_x_h - 2*padding_h;
    uint32_t unpadded_x_w = padded_x_w - 2*padding_w;
   
    xt::random::seed(0);
    xt::xarray<float> x = xt::random::randint<int>({x_c, unpadded_x_h, unpadded_x_w}, 0, 100);
    xt::xarray<float> k = xt::random::randint<int>({k_num, k_c,k_h,k_w}, 0, 100);
    xt::xarray<float> b = xt::random::randint<int>({k_num}, 0, 100);

    float* d_x;
    float* d_k;
    float* d_b;
    float* d_y;
    float* h_y;

    cudaMalloc(&d_x, unpadded_x_h*unpadded_x_w*x_c*sizeof(float));
    cudaMalloc(&d_k, k_num*k_h*k_w*k_c*sizeof(float));
    cudaMalloc(&d_b, k_num*sizeof(float));
    cudaMalloc(&d_y, y_c*y_h*y_w*sizeof(float));
    cudaMallocHost(&h_y, y_c*y_h*y_w * sizeof(float));

    cudaMemcpy(d_x, x.data(), unpadded_x_h*unpadded_x_w*x_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k.data(), k_num*k_h*k_w*k_c*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), k_num*sizeof(float), cudaMemcpyHostToDevice);

    StreamPool pool(1);

    sconv2d<<<CONV2D_GRID(y_c, y_h, y_w), CONV2D_BLOCK(), 0, pool.Stream()>>>(d_x, 
                                                                              d_k, 
                                                                              true, d_b, 
                                                                              d_y, 
                                                                              k_num, k_c, k_h, k_w, 
                                                                              x_c, padded_x_h, padded_x_w,
                                                                              y_c, y_h, y_w,
                                                                              stride_h, stride_w,
                                                                              padding_h, padding_w,
                                                                              ActivationFunction::NONE
                                                                              );

   

    cudaMemcpy(h_y, d_y, y_c*y_h*y_w * sizeof(float), cudaMemcpyDeviceToHost);

    bool chk = check(   x,
                        k,
                        b,
                        h_y,
                        padded_x_h,
                        padded_x_w,
                        x_c,
                        k_num,
                        k_c,
                        k_h,
                        k_w,
                        stride_h,
                        stride_w,
                        y_c,
                        y_h,
                        y_w,
                        padding_h,
                        padding_w
                    );

    printf("Cube_Y check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_x);
    cudaFree(d_k);
    cudaFree(d_b);
    cudaFree(d_y);
    cudaFreeHost(h_y);

    return 0;
}