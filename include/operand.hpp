#pragma once
#include "tensor.hpp"

namespace lotus {

    struct Operand {
        Tensor tensor_;
        std::string name_;
        std::string producer_;
        std::vector<std::string> consumers_;

    };
}
