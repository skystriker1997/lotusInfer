#pragma once
#include "layer/layer.hpp"


namespace lotus {
    class LayerFlatten: public Layer {
        public:
        LayerFlatten(
                    const std::string& name,
                    const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                    const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs
                    );
                    
        void Forward() override;
        ~LayerFlatten() override = default;
    };


    std::shared_ptr<LayerFlatten> MakeLayerFlatten(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands);
}