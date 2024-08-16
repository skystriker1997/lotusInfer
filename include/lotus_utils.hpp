#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <fmt/core.h>
#include <memory>
#include <glog/logging.h>
#include <type_traits>
#include <cublas_v2.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string>
#include <queue>
#include <set>
#include <map>
#include <functional>
#include <iostream>
#include <fstream>


#define CUDA_CHECK(err)                                                                                                                         \
    do {                                                                                                                                        \
        cudaError_t _err = (err);                                                                                                               \
        CHECK(_err == cudaSuccess) << fmt::format("CUDA error {} at {}:{}\n", cudaGetErrorString(_err), __FILE__, __LINE__);                    \
    } while (0)


#define CUBLAS_CHECK(err)                                                                                                                       \
    do {                                                                                                                                        \
        cublasStatus_t _err = (err);                                                                                                            \
        CHECK(_err == CUBLAS_STATUS_SUCCESS) << fmt::format("CUBLAS error {} at {}:{}\n", cublasGetStatusString(_err), __FILE__, __LINE__);     \
    } while (0)


namespace lotus {

    enum ActivationFunction {
        NONE,
        RELU
    };


    template<typename T>
    std::enable_if_t<std::is_same_v<__half,T> || std::is_same_v<float,T>, std::shared_ptr<T>>
    MakeCudaShared(size_t n) {
        size_t size = n*sizeof(T);
        T* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        return std::shared_ptr<T>(ptr, [](T* ptr){CUDA_CHECK(cudaFree(ptr));});
    };


    __device__ __forceinline__
    void ldgsts128(const float* dst, const float* src, const int src_size) {
        asm volatile (
            "{.reg .u64 u64addr;\n"
            ".reg .u32 u32addr;\n"
            " cvta.to.shared.u64 u64addr, %0;\n"
            " cvt.u32.u64 u32addr, u64addr;\n"
            " cp.async.ca.shared.global [u32addr], [%1], 16, %2;\n}"
            ::"l"(dst), "l"(src), "r"(4*src_size)
        );
    };

    __device__ __forceinline__
    void ldgsts32(const float* dst, const float* src, const int src_size) {
        asm volatile (
            "{.reg .u64 u64addr;\n"
            ".reg .u32 u32addr;\n"
            " cvta.to.shared.u64 u64addr, %0;\n"
            " cvt.u32.u64 u32addr, u64addr;\n"
            " cp.async.ca.shared.global [u32addr], [%1], 4, %2;\n}"
            ::"l"(dst), "l"(src), "r"(4*src_size)
        );
    };




    __device__ __forceinline__
    void wait() {
        asm volatile (
            " cp.async.wait_all;\n"::
        );
    };


    class StreamPool {
        private:
        cublasHandle_t handle_;
        std::vector<cudaStream_t> streams_;
        size_t ptr_;

        public:
        StreamPool(size_t n) {
            CUBLAS_CHECK(cublasCreate(&handle_));
            for(size_t i=0; i<n; ++i) {
                streams_.emplace_back(nullptr);
                CUDA_CHECK(cudaStreamCreate(&streams_[i]));
            }
            CUBLAS_CHECK(cublasSetStream(handle_, streams_[0]));
            ptr_ = 0;
        };

        cublasHandle_t Handle() const {
            return handle_;
        }

        void SetStream() {
            ptr_++;
            if(ptr_ >= streams_.size()) {
                ptr_=0;
            } 
            CUBLAS_CHECK(cublasSetStream(handle_, streams_[ptr_]));
        }

        cudaStream_t Stream() {
            return streams_[ptr_];
        }

        ~StreamPool() {
            CUBLAS_CHECK(cublasDestroy(handle_));
            size_t n = streams_.size();
            for(size_t i=0; i<n; i++) {
                CUDA_CHECK(cudaStreamDestroy(streams_[i]));
            }
        };
    };


}