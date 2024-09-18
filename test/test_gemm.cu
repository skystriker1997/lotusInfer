#include "lotus_utils.hpp"
#include "operator/gemm_tensorcore.cuh"
#include <Eigen/Dense> 




int main() 
{
    using namespace lotus;
    using namespace Eigen;
    using MatrixRowMajor = Matrix <float,Dynamic,Dynamic,RowMajor>;
    int seq = 255;
    int input_features = 255;
    int output_features = 255;

    MatrixRowMajor input = MatrixXf(seq,input_features);
    MatrixRowMajor weight = MatrixXf(output_features,input_features);
    RowVectorXf bias = RowVectorXf(output_features);

    for(int i=0; i<seq; ++i) {
        for(int j=0; j<input_features; ++j) {
            if((i*j)%2==0) {
                input(i, j) = -1.f/j+100.f/i;
            } else {
                input(i, j) = 1.f/j-100.f/i;
            }
            
        }
    }

    for(int i=0; i<output_features; ++i) {
        for(int j=0; j<input_features; ++j) {
            if((i*j)%2==0) {
                weight(i, j) = 1.f/j-50.f/i;
            } else {
                weight(i, j) = -1.f/j+50.f/i;
            }            
        }
    }

    for(int i=0; i<output_features; ++i) {
        if(i%2==0) {
            bias(i) = 1;
        } else {
            bias(i) = -1;
        }
    }

    MatrixRowMajor output = (input*weight.transpose()).rowwise() + bias; 

    float *d_a, *d_b, *d_c, *d_bias, *h_c;
    cudaMalloc(&d_a, seq * input_features * sizeof(float));
    cudaMalloc(&d_b, output_features * input_features * sizeof(float));
    cudaMalloc(&d_c, seq * output_features * sizeof(float));
    cudaMalloc(&d_bias, output_features * sizeof(float));
    cudaMallocHost(&h_c, seq * output_features * sizeof(float));

    cudaMemcpy(d_a, input.data(), seq * input_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, weight.data(), output_features * input_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.data(), output_features * sizeof(float), cudaMemcpyHostToDevice);

    Gemm<<<MakeGemmGrid(seq, output_features), MakeGemmBlock()>>>(d_a, d_b, true, d_bias, d_c, seq, output_features, input_features, ActivationFunction::NONE);
    cudaMemcpy(h_c, d_c, seq * output_features * sizeof(float), cudaMemcpyDeviceToHost);

    for(uint32_t i=0; i<seq; i++) {
        for(uint32_t j=0; j<output_features; j++) {
            float cuda_result = h_c[i*output_features+j];
            float cpu_result = output(i,j);
            if (std::fabs(cpu_result - cuda_result) / std::fabs(cpu_result) > 1e-5f) {
                printf("Matrix C[%d][%d] not match, %f vs %f\n", i, j, cpu_result, cuda_result);
                return 0;
            }
        }
    }

    printf("Matrix C check OK\n");

    return 0;
}