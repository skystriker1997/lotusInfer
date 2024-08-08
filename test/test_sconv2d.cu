#include "operator/conv2d.cuh"
#include <cmath>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"



bool check( xt::xarray<float>& x,
            xt::xarray<float>& k,
            xt::xarray<float>& b,
            float* y,
            int x_h,
            int x_w,
            int x_c,
            int k_num,
            int k_c,
            int k_h,
            int k_w,
            int stride_h,
            int stride_w,
            int y_c,
            int y_h,
            int y_w,
            int padding_h,
            int padding_w
           ) 
{

    auto sum = [](const float& left, const float& right){return left + right;};
    xt::xarray<float> padded_x = xt::zeros<float>({x_c, x_h, x_w});
    xt::view(padded_x, xt::all(), xt::range(padding_h, x_h-padding_h), xt::range(padding_w, x_w-padding_w)) = x;

    for(int i=0; i<y_h; i++) {
        for(int j=0; j<y_w; j++) {
            auto x_frag = xt::view(padded_x, xt::all(), xt::range(i*stride_h, i*stride_h+k_h), xt::range(j*stride_w, j*stride_w+k_w));
            for(int q=0; q<k_num; ++q) {
                auto k_frag = xt::view(k, xt::range(q, q+1), xt::all(), xt::all(), xt::all());
                auto product = k_frag * x_frag;
                float result = xt::reduce(sum, product, {0,1,2,3})(0) + b(q);
                int target = q*(y_h*y_w) + i*y_w + j;
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

    int x_h = 257;
    int x_w = 257;
    int x_c = 3;
    int k_num = 32;
    int k_c = 3;
    int k_h = 9;
    int k_w = 9;
    int stride_h = 2;
    int stride_w = 2;;
    int padding_h = 4;
    int padding_w = 4;

    int y_c = k_num;

    int y_h = (x_h-k_h)/stride_h + 1;
    int y_w = (x_w-k_w)/stride_w + 1;   

    int unpadded_x_h = x_h - 2*padding_h;
    int unpadded_x_w = x_w - 2*padding_w;
   
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

    dim3 grid((y_h*y_w+127)/128, (y_c+127)/128);
    StreamPool pool(1);

    sconv2d<<<grid, 256, 0, pool.Stream()>>>(  d_x, 
                            d_k, 
                            d_b, 
                            d_y, 
                            k_num, k_h, k_w, k_c, 
                            x_w, x_h, x_c,
                            y_w, y_h, y_c,
                            stride_h, stride_w,
                            padding_h, padding_w
                          );

   

    cudaMemcpy(h_y, d_y, y_c*y_h*y_w * sizeof(float), cudaMemcpyDeviceToHost);

    bool chk = check(   x,
                        k,
                        b,
                        h_y,
                        x_h,
                        x_w,
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