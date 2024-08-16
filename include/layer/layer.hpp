#pragma once

#include "lotus_utils.hpp"
#include "operand.hpp"
#include "pnnx/ir.h"


namespace lotus {
    class Layer {
        protected:
        std::string name_;
        std::vector<std::string> inputs_name_;
        std::vector<std::string> outputs_name_;
        std::vector<std::shared_ptr<Operand>> inputs_;
        std::vector<std::shared_ptr<Operand>> outputs_;

        public: 
        Layer() = default;
        Layer& operator=(const Layer& rhs) = delete;
        Layer& operator=(Layer&& rhs) = delete;
        const std::string& Name() {return name_;};
        const std::vector<std::string>& InputsName() {return inputs_name_;};
        const std::vector<std::string>& OutputsName() {return outputs_name_;};
        virtual void Forward() {};
        virtual ~Layer() = default;
    };
}
