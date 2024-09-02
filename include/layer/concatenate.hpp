#pragma once
#include "layer/layer.hpp"


namespace lotus {
    class CatLayer: public Layer {
        private:
        uint32_t dim_;
        public:
        CatLayer(const std::string& name,
                 const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                 const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& output,
                 uint32_t dim);
                    
        void Forward() override;
        ~CatLayer() override = default;
    };


    std::shared_ptr<CatLayer> MakeCatLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands);
}