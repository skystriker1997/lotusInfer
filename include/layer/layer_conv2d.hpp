#pragma once

#include "layer/layer.hpp"
#include "operator/conv2d.cuh"


namespace lotus {
    class Conv2dLayer: public Layer {
        private:
        Tensor kernel_;
        bool use_bias_;
        Tensor bias_;
        uint32_t stride_h_;
        uint32_t stride_w_;
        uint32_t padding_h_; 
        uint32_t padding_w_;

        public:
        Conv2dLayer(const std::string& name,
                    const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                    const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                    const std::vector<char>& kernel, 
                    const uint32_t k_num, const uint32_t k_c, const uint32_t k_h, const uint32_t k_w, 
                    const bool use_bias, const std::vector<char>& bias, 
                    const uint32_t stride_h, const uint32_t stride_w,
                    const uint32_t padding_h, const uint32_t padding_w);
                                 
        
        void Forward() override;
        ~Conv2dLayer() override = default;
    };


    std::shared_ptr<Conv2dLayer> MakeConv2dLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands);
}