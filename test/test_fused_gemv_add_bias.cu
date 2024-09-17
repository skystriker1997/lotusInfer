#include "operator/fused_gemv_add_bias.cuh"


void random_init(float *data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = float(rand()) / RAND_MAX;
    }
}



bool check(const float *h_x,
           const float *h_a,
           const float *h_b,
           const float *h_y,
           uint32_t a_w, uint32_t a_h) {
    for (uint32_t j = 0; j < a_h; ++j) {
        float sum = 0.f;
        for (uint32_t p = 0; p < a_w; ++p) {
            sum += h_x[p] * h_a[j * a_w + p];
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

    uint32_t a_w = 512;
    uint32_t a_h = 512;

    float *h_x, *h_a, *h_y, *h_b;
    cudaMallocHost(&h_x, a_w * sizeof(float));
    cudaMallocHost(&h_a, a_w * a_h * sizeof(float));
    cudaMallocHost(&h_y, a_h * sizeof(float));
    cudaMallocHost(&h_b, a_h * sizeof(float));

    random_init(h_a, a_w * a_h);
    random_init(h_x, a_w);
    random_init(h_b, a_h);

    float *d_x, *d_a, *d_y, *d_b;
    cudaMalloc(&d_x, a_w * sizeof(float));
    cudaMalloc(&d_a, a_w * a_h * sizeof(float));
    cudaMalloc(&d_y, a_h * sizeof(float));
    cudaMalloc(&d_b, a_h * sizeof(float));

    cudaMemcpy(d_a, h_a, a_w * a_h * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, a_w * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, a_h * sizeof(float), cudaMemcpyHostToDevice);

    StreamPool pool(1);

    Fgemva<<<MakeSFgemvaGrid(a_h), MakeSFgemvaBlock(), 0, pool.Stream()>>>(d_x, d_a, d_b, d_y, a_h, a_w, true, ActivationFunction::NONE);

    cudaMemcpy(h_y, d_y, a_h * sizeof(float), cudaMemcpyDeviceToHost);

    bool chk = check(h_x, h_a, h_b, h_y, a_w, a_h);

    printf("vector_y check: %s\n", chk ? "OK" : "Failed");

    cudaFree(d_x);
    cudaFree(d_a);
    cudaFree(d_y);
    cudaFreeHost(h_x);
    cudaFreeHost(h_a);
    cudaFreeHost(h_y);

    return 0;
}

