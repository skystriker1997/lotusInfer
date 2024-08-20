#include "layer/flatten.hpp"


namespace lotus {

    FlattenLayer::FlattenLayer(
                            const std::string& name,
                            const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                            const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs
                            )
    {   
        name_ = name;
        inputs_name_ = inputs_name;
        outputs_name_ = outputs_name;
        inputs_ = inputs;
        outputs_ = outputs;
    };

        
    void FlattenLayer::Forward() {
        auto x_batch = inputs_[0];
        auto y_batch = outputs_[0];

        Tensor& in_tensor = x_batch->tensor_;
        Tensor& out_tensor = y_batch->tensor_;

        std::vector<uint32_t> flattened_shape(out_tensor.DimSize());
        for(size_t i=0; i<out_tensor.DimSize(); ++i) {
            flattened_shape[i] = out_tensor.Dim(i);
        }
        out_tensor = in_tensor;
        out_tensor.Reshape(flattened_shape);
    };


    std::shared_ptr<FlattenLayer> MakeFlattenLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands) {
        CHECK(opt->inputs.size()==1) << "flatten layer gets more than 1 input";  
        CHECK(opt->outputs.size()==1) << "flatten layer gets more than 1 output";

        std::string input_name;
        if(opt->inputs[0]->producer->type=="nn.ReLU") {
            input_name = opt->inputs[0]->producer->inputs[0]->name;
        } else {
            input_name = opt->inputs[0]->name;
        }
        std::vector<std::string> inputs_name = {input_name};
        std::vector<std::string> outputs_name = {opt->outputs[0]->name};
        auto input = operands.find(inputs_name[0]);
        CHECK(input != operands.end()) << "flatten layer fails to find the input operand";
        auto output = operands.find(outputs_name[0]);
        CHECK(output != operands.end()) << "flatten layer fails to find the output operand";

        std::vector<std::shared_ptr<Operand>> inputs = {input->second};
        std::vector<std::shared_ptr<Operand>> outputs = {output->second};

        return std::make_shared<FlattenLayer>(opt->name,
                                             inputs_name, outputs_name,
                                             inputs, outputs);
    };
}