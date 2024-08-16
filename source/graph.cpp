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

        CHECK(topo_sorted_layers_.size() == layers_.size()) << "the pnnx graph is not a directed acyclic one";
    };


    
    Graph::Graph(const std::string& param_path, const std::string& bin_path, uint32_t batch_size) {

        pnnx::Graph pnnx_graph;

        // std::cout << "pnnx is to load the model" << std::endl;

        CHECK(pnnx_graph.load(param_path, bin_path) == 0) << "graph fails to load the model and coefficients";

        // std::cout << "pnnx loaded the model" << std::endl;

        for(pnnx::Operand *opd: pnnx_graph.operands) {

            if(opd->producer->type == "nn.ReLU") {
                continue;
            }

            auto operand = std::make_shared<Operand>();

            CHECK(opd->type == 1) << "lotusInfer does not support data type other than float32";
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
                std::string type = consumer->type;
                if(type != "nn.ReLU") {
                    operand->consumers_.emplace_back(consumer->name);
                } else {
                    for(pnnx::Operator* _consumer : consumer->outputs[0]->consumers) {
                        operand->consumers_.emplace_back(_consumer->name);
                    }
                }
            }

            operands_.insert({operand->name_, operand});

            if(opd->producer->type == "pnnx.Input") {
                CHECK(input_operand_.empty()) << "graph supports only one input";
                input_operand_ = opd->name;
            }
           
            if(opd->consumers[0]->type == "pnnx.Output") {
                CHECK(output_operand_.empty()) << "graph supports only one output";
                output_operand_ = opd->name;
            }
        }

        for(pnnx::Operator *opt: pnnx_graph.ops) {
            if(opt->type == "pnnx.Input" || opt->type == "pnnx.Output" || opt->type == "nn.ReLU")
                continue;
            CHECK(layers_.find(opt->name) == layers_.end()) << "duplicate layer names";
            if(opt->type == "nn.Conv2d") {
                layers_.insert({opt->name, MakeLayerConv2d(opt, operands_)});
            } else if(opt->type == "nn.MaxPool2d") {
                layers_.insert({opt->name, MakeLayerMaxpool2d(opt, operands_)});
            } else if(opt->type == "nn.AdaptiveAvgPool2d") {
                layers_.insert({opt->name, MakeLayerAdaptiveAvgpool2d(opt, operands_)});
            } else if(opt->type == "torch.flatten") {
                layers_.insert({opt->name, MakeLayerFlatten(opt, operands_)});
            } else if(opt->type == "nn.Linear") {
                layers_.insert({opt->name, MakeLayerLinear(opt, operands_)});
            } else {
                CHECK(false) << fmt::format("lotusInfer does not support layer {} up to now", opt->type);
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