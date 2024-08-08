#include "operator/fused_gemv_add.cuh"
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>




void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}



bool check(const float *h_x,
           const float *h_a,
           const float *h_b,
           const float *h_y,
           int x_w, int a_h) {
    for (int j = 0; j < a_h; ++j) {
        float sum = 0.f;
        for (int p = 0; p < x_w; ++p) {
            sum += h_x[p] * h_a[j * x_w + p];
        }
        sum += h_b[j];
        if (std::fabs(sum - h_y[j]) / std::fabs(sum) > 1e-5f) {
            printf("y[%d] not match, %f vs %f\n", j, sum, h_y[j]);
            return false;
        }
    }
    

    return true;
}


int main() {

    using namespace lotus;

    int x_w = 5099;
    int a_h = 1024;

    float *h_x, *h_a, *h_y, *h_b;
    cudaMallocHost(&h_x, x_w * sizeof(float));
    cudaMallocHost(&h_a, x_w * a_h * sizeof(float));
    cudaMallocHost(&h_y, a_h * sizeof(float));
    cudaMallocHost(&h_b, a_h * sizeof(float));

    random_init(h_a, x_w * a_h);
    random_init(h_x, x_w);
    random_init(h_b, a_h);

    float *d_x, *d_a, *d_y, *d_b;
    cudaMalloc(&d_x, x_w * sizeof(float));
    cudaMalloc(&d_a, x_w * a_h * sizeof(float));
    cudaMalloc(&d_y, a_h * sizeof(float));
    cudaMalloc(&d_b, a_h * sizeof(float));

    cudaMemcpy(d_a, h_a, x_w * a_h * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, x_w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, a_h * sizeof(float), cudaMemcpyHostToDevice);

    StreamPool pool(1);

    sfgemva<<<a_h, 256, 0, pool.Stream()>>>(d_x, d_a, d_b, d_y, x_w, a_h);

    cudaMemcpy(h_y, d_y, a_h * sizeof(float), cudaMemcpyDeviceToHost);

    bool chk = check(h_x, h_a, h_b, h_y, x_w, a_h);

    printf("vector_y check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_x);
    cudaFree(d_a);
    cudaFree(d_y);
    cudaFreeHost(h_x);
    cudaFreeHost(h_a);
    cudaFreeHost(h_y);

    return 0;
}

