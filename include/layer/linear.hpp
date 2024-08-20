#pragma once
#include "layer/layer.hpp"
#include "operator/fused_gemv_add_bias.cuh"

namespace lotus {
    class LinearLayer: public Layer {
        private:
        Tensor weight_;
        uint32_t in_features_;
        uint32_t out_features_;
        bool use_bias_;
        Tensor bias_;

        public:
        LinearLayer(
                    const std::string& name,
                    const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                    const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                    const std::vector<char>& weight, const uint32_t in_features, const uint32_t out_features,
                    const bool use_bias, const std::vector<char>& bias
                    );
                    

        void Forward() override;
        ~LinearLayer() override = default;
    };


    std::shared_ptr<LinearLayer> MakeLinearLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands);
}