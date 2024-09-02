#include "graph.hpp"

namespace lotus {

    void Graph::TopoSortLayers() {

        std::queue<std::shared_ptr<Layer>> topo_queue;
        std::map<std::string, int> in_degrees;

        for(auto &[name, layer]: layers_) {
            bool first = true;
            int in_degree = 0;
            for(const auto &input: layer->InputsName()) {
                if(input != input_operand_) {
                    first = false;
                    in_degree++;
                }
            }
            if(first)
                topo_queue.push(layer);
            in_degrees.insert({layer->Name(), in_degree});
        }

        while(!topo_queue.empty()) {
            auto &layer = topo_queue.front();
            for(const auto &output: layer->OutputsName()) {
                for(auto &consumer: operands_[output]->consumers_) {
                    int &in_degree = in_degrees[consumer];
                    in_degree--;
                    if(!in_degree) {
                        topo_queue.push(layers_[consumer]);
                    }
                }
            }
            topo_sorted_layers_.emplace_back(layer);
            topo_queue.pop();
        }

        CHECK(topo_sorted_layers_.size() == layers_.size()) << "the pnnx graph is supposed to be directed acyclic";
    };


    
    Graph::Graph(const std::string& param_path, const std::string& bin_path, uint32_t batch_size) {

        pnnx::Graph pnnx_graph;

        CHECK(pnnx_graph.load(param_path, bin_path) == 0) << "incorrect model parameter path or model coefficient path";

        for(pnnx::Operand *opd: pnnx_graph.operands) {
            if(opd->producer->type == "pnnx.Input") {
                CHECK(input_operand_.empty()) << "the graph is supposed to accept 1 input";
                input_operand_ = opd->name;
            }
           
            if(opd->consumers[0]->type == "pnnx.Output") {
                CHECK(output_operand_.empty()) << "the graph is supposed to generate 1 output";
                output_operand_ = opd->name;
                std::cout << "output operand is " << output_operand_ << std::endl;
            }

            auto operand = std::make_shared<Operand>();

            CHECK(opd->type == 1) << "lotusInfer supports only float32 up to now";
            std::vector<uint32_t> shape(opd->shape.size());
            for(size_t i=0; i<opd->shape.size(); ++i) {
                if(i==0) {
                    shape[0] = batch_size;
                } else {
                    shape[i] = opd->shape[i];
                }
            }
            
            operand->tensor_ = Tensor(shape);
            operand->name_ = opd->name;
            operand->producer_ = opd->producer->name;
            for(pnnx::Operator* consumer : opd->consumers) {
                operand->consumers_.emplace_back(consumer->name);
            }

            operands_.insert({operand->name_, operand});
        }

        for(pnnx::Operator *opt: pnnx_graph.ops) {
            if(opt->type == "pnnx.Input" || opt->type == "pnnx.Output" || opt->type == "nn.ReLU" || opt->type == "F.sigmoid")
                continue;
            CHECK(layers_.find(opt->name) == layers_.end()) << "duplicate layer names";
            if(opt->type == "nn.Conv2d") {
                layers_.insert({opt->name, MakeConv2dLayer(opt, operands_)});
            } else if(opt->type == "nn.MaxPool2d") {
                layers_.insert({opt->name, MakeMaxpool2dLayer(opt, operands_)});
            } else if(opt->type == "nn.AdaptiveAvgPool2d") {
                layers_.insert({opt->name, MakeAdaptiveAvgpool2dLayer(opt, operands_)});
            } else if(opt->type == "torch.flatten") {
                layers_.insert({opt->name, MakeFlattenLayer(opt, operands_)});
            } else if(opt->type == "nn.Linear") {
                layers_.insert({opt->name, MakeLinearLayer(opt, operands_)});
            } else if(opt->type == "pnnx.Expression") {
                layers_.insert({opt->name, MakeExpressionLayer(opt, operands_)});
            } else if(opt->type == "torch.cat") {
                layers_.insert({opt->name, MakeCatLayer(opt, operands_)});
            } else if(opt->type == "nn.ConvTranspose2d") {
                layers_.insert({opt->name, MakeTransposedConv2dLayer(opt, operands_)});
            } else {
                CHECK(false) << "lotusInfer does not support layer " << opt->type << " up to now";
            }
        }

        TopoSortLayers();

    }


    void Graph::InitialiseInput(const std::vector<float>& input) {
        operands_[input_operand_]->tensor_.AssignData(input);
    };


    std::shared_ptr<Operand> Graph::Output() {
        return operands_[output_operand_];
    };


    void Graph::Forward() {
        for(const auto &layer: topo_sorted_layers_) {
            std::cout << "layer " << layer->Name() << " started" << std::endl;
            layer->Forward();
            std::cout << "layer " << layer->Name() << " completed" << std::endl;
        }
    };

}