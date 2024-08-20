#pragma once
#include "layer/layer.hpp"


namespace lotus {
    class FlattenLayer: public Layer {
        public:
        FlattenLayer(
                    const std::string& name,
                    const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                    const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs
                    );
                    
        void Forward() override;
        ~FlattenLayer() override = default;
    };


    std::shared_ptr<FlattenLayer> MakeFlattenLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands);
}