#pragma once

#include "lotus_utils.hpp"
#include "operand.hpp"


namespace lotus {
    class Layer {
        private:
        std::string name_;
        std::vector<std::sring> inputs_name_;
        std::vector<std::sring> outputs_name_;
        std::vector<std::shared_ptr<Operand>> inputs_;
        std::vector<std::shared_ptr<Operand>> outputs_;

        public: 
        Layer() = default;

        Layer& operator=(const Layer& rhs) = delete;
        Layer& operator=(Layer&& rhs) = delete;
        void AttachInput(const std::shared_ptr<Operand>& input) {inputs_.emplace_back(input);};
        void AttachOutput(const std::shared_ptr<Operand>& output) {outputs_.emplace_back(output);};
        virtual void Forward() {};
        virtual ~Layer() = default;
    }
}
