#pragma once

#include "layer.hpp"
#include "operator/conv2d.cuh"

namespace lotus {
    class Conv2d: public Layer {
        private:
        Tensor kernel_;
        Tensor bias_;
        uint32_t stride_h_;
        uint32_t stride_w_;
        uint32_t dilation_h_;
        uint32_t dilation_w_;
        uint32_t padding_h_; 
        uint32_t padding_w_;

        public:
        Conv2d( std::vector<char>& kernel, uint32_t k_num, uint32_t k_c, uint32_t k_h, uint32_t k_w, 
                std::vector<char>& bias, 
                uint32_t stride_h, uint32_t stride_w,
                uint32_t dilation_h, uint32_t dilation_w,
                uint32_t padding_h, uint32_t padding_w);
        
        void Forward() override;
        ~Conv2d() override = default;
    }


    std::shared_ptr<Conv2d> MakeConv2d(pnnx::Operator *opt)
}