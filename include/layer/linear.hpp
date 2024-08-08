#pragma once

#include "layer.hpp"
#include "operator/gemv.cuh"
#include "operator/gemm.cuh"

namespace lotus {
    class Linear: public Layer {
        private:
        Tensor weight_;
        Tensor bias_;
        bool use_bias_;

        public:
        Linear( std::vector<char>& weight, std::vector<int>& weight_shape, 
                std::vector<char>& bias, bool use_bias_);
        
        void Forward() override;
        ~Conv2d() override = default;
    }


    std::shared_ptr<Linear> MakeLinear(pnnx::Operator *opt)
}