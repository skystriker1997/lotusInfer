#pragma once

#include "layer/layer.hpp"
#include "operator/maxpool2d.cuh"


namespace lotus {
    class Maxpool2dLayer: public Layer {
        private:
        uint32_t kernel_h_;
        uint32_t kernel_w_;
        uint32_t stride_h_;
        uint32_t stride_w_;
        uint32_t padding_h_; 
        uint32_t padding_w_;

        public:
        Maxpool2dLayer(const std::string& name,
                       const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                       const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                       const uint32_t kernel_h, const uint32_t kernel_w, 
                       const uint32_t stride_h, const uint32_t stride_w,
                       const uint32_t padding_h, const uint32_t padding_w,
                       ActivationFunction af);
        
        void Forward() override;
        ~Maxpool2dLayer() override = default;
    };


    std::shared_ptr<Maxpool2dLayer> MakeMaxpool2dLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands);
}