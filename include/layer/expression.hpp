#pragma once
#include "layer/layer.hpp"
#include <regex>
#include "operator/add.cuh"

namespace lotus {
    class ExpressionLayer: public Layer {
        private:
        std::string expression_;
        void Parse();
        public:
        ExpressionLayer(
                    const std::string& name,
                    const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                    const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                    const std::string expression
                    );
                    
        void ParseExpression();
        void Forward() override;
        ~ExpressionLayer() override = default;
    };


    std::shared_ptr<ExpressionLayer> MakeExpressionLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands);
}