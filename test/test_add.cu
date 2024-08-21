#include "operator/add.cuh"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"



bool check( xt::xarray<float>& x1,
            xt::xarray<float>& x2,
            const float* h_y,
            const uint32_t size
           ) 
{
    auto y = x1+x2;
    for(uint32_t i=0; i<size; ++i) {
        if (std::fabs(y(i) - h_y[i]) / std::fabs(y(i)) > 1e-5f) {
            printf("y[%d] not match, %f vs %f\n", i, y(i), h_y[i]);
            return false;
        }
    }
    return true;
}




int main() 
{
    using namespace lotus;
    
    uint32_t size = 10000;

    xt::random::seed(0);
    xt::xarray<float> x1 = xt::random::randn<float>({size});
    xt::xarray<float> x2 = xt::random::randn<float>({size});

    float* d_x1;
    float* d_x2;
    float* d_y;
    float* h_y;

    cudaMalloc(&d_x1, size*sizeof(float));
    cudaMalloc(&d_x2, size*sizeof(float));
    cudaMalloc(&d_y, size*sizeof(float));
    cudaMallocHost(&h_y, size*sizeof(float));

    cudaMemcpy(d_x1, x1.data(), size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, x2.data(), size*sizeof(float), cudaMemcpyHostToDevice);

    StreamPool pool(1);

    sadd<<<MakeAddGrid(size), MakeAddBlock(), 0, pool.Stream()>>>(d_x1, d_x2, d_y, size, ActivationFunction::NONE);

    cudaMemcpy(h_y, d_y, size*sizeof(float), cudaMemcpyDeviceToHost);

    bool chk = check(x1, x2, h_y, size);

    printf("vector_Y check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_y);
    cudaFreeHost(h_y);

    return 0;
}