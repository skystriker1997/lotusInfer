#pragma once
#include "layer/layer.hpp"
#include "operator/fused_gemv_add_bias.cuh"

namespace lotus {
    class LayerLinear: public Layer {
        private:
        Tensor weight_;
        uint32_t in_features_;
        uint32_t out_features_;
        bool use_bias_;
        Tensor bias_;
        ActivationFunction af_;

        public:
        LayerLinear(
                    const std::string& name,
                    const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                    const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                    const std::vector<char>& weight, const uint32_t in_features, const uint32_t out_features,
                    const bool use_bias, const std::vector<char>& bias
                    );
                    
        void SetActivation(ActivationFunction af);

        void Forward() override;
        ~LayerLinear() override = default;
    };


    std::shared_ptr<LayerLinear> MakeLayerLinear(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands);
}