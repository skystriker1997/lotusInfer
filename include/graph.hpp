#pragma once

#include "layer/layer_adaptive_avgpool2d.hpp"
#include "layer/layer_maxpool2d.hpp"
#include "layer/flatten.hpp"
#include "layer/linear.hpp"
#include "layer/layer_conv2d.hpp"
#include "layer/expression.hpp"
#include "layer/concatenate.hpp"
#include "layer/layer_transposed_conv2d.hpp"


namespace lotus {

    class Graph {
    private:

        std::map<std::string, std::shared_ptr<Layer>> layers_;

        std::map<std::string, std::shared_ptr<Operand>> operands_;

        std::vector< std::shared_ptr<Layer>> topo_sorted_layers_;

        std::string input_operand_;

        std::string output_operand_;

        void TopoSortLayers();



    public:
        Graph(const std::string& param_path, const std::string& bin_path, uint32_t batch_size);

        ~Graph() = default;

        void InitialiseInput(const std::vector<float>& input);

        std::shared_ptr<Operand> Output();

        void Forward();

    };

}