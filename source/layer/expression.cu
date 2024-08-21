#include "layer/expression.hpp"


namespace lotus {

    ExpressionLayer::ExpressionLayer(const std::string& name,
                                     const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                                     const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                                     const std::string expression
                                    ) 
    {
        name_ = name;
        inputs_name_ = inputs_name;
        outputs_name_ = outputs_name;
        inputs_ = inputs;
        outputs_ = outputs;
        expression_ = expression;
        Parse();
    }

    void ExpressionLayer::Parse() {
        std::regex pattern(R"(add\(@(\d+),@(\d+)\))");
        std::smatch matches;
        CHECK(std::regex_search(expression_, matches, pattern)) << "Expression layer only supports addition of two operands up to now";
    }


    void ExpressionLayer::Forward() {
        auto x1_batch = inputs_[0];
        auto x2_batch = inputs_[1];
        auto y_batch = outputs_[0];

        size_t batch_size = y_batch->tensor_.Dim(0);

        StreamPool pool(batch_size);
        for(int i=0; i<batch_size; ++i) {
            Tensor x1 = x1_batch->tensor_.Element(i);
            Tensor x2 = x2_batch->tensor_.Element(i);
            Tensor y = y_batch->tensor_.Element(i);

            if(i != 0) {
                pool.SetStream();
            }

            sadd<<<MakeAddGrid(y.Size()), MakeAddBlock(), 0, pool.Stream()>>>(x1.Data(), x2.Data(), y.Data(), y.Size(), af_);
            
        }
        cudaDeviceSynchronize();
    }

    std::shared_ptr<ExpressionLayer> MakeExpressionLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands) {
        CHECK(opt->inputs.size()==2) << "expression layer only accepts 2 inputs up to now";  
        CHECK(opt->outputs.size()==1) << "expression layer gets more than 1 output";

        auto expr = opt->params.find("expr");
        CHECK(expr != opt->params.end()) << "expression layer fails to find parameter 'expr'";

        std::vector<std::string> inputs_name(2);

        if(opt->inputs[0]->producer->type=="nn.ReLU") {
            inputs_name[0] = opt->inputs[0]->producer->inputs[0]->name;
        } else {
            inputs_name[0] = opt->inputs[0]->name;
        }

        if(opt->inputs[1]->producer->type=="nn.ReLU") {
            inputs_name[1] = opt->inputs[1]->producer->inputs[0]->name;
        } else {
            inputs_name[1] = opt->inputs[1]->name;
        }

        std::vector<std::string> outputs_name = {opt->outputs[0]->name};

        auto input1 = operands.find(inputs_name[0]);
        CHECK(input1 != operands.end()) << "expression layer fails to find the input operand1";

        auto input2 = operands.find(inputs_name[1]);
        CHECK(input2 != operands.end()) << "expression layer fails to find the input operand2";

        auto output = operands.find(outputs_name[0]);
        CHECK(output != operands.end()) << "expression layer fails to find the output operand";

        std::vector<std::shared_ptr<Operand>> inputs = {input1->second, input2->second};
        std::vector<std::shared_ptr<Operand>> outputs = {output->second};

        return std::make_shared<ExpressionLayer>(opt->name, inputs_name, outputs_name, inputs, outputs, expr->second.s);
    };

}