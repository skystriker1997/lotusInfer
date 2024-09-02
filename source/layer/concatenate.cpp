#include "layer/concatenate.hpp"


namespace lotus {

    CatLayer::CatLayer(const std::string& name,
                       const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                       const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                       uint32_t dim)
    {   
        name_ = name;
        inputs_name_ = inputs_name;
        outputs_name_ = outputs_name;
        inputs_ = inputs;
        outputs_ = outputs;
        dim_ = dim;
    };

        
    void CatLayer::Forward() {
        auto x1_batch = inputs_[0];
        auto x2_batch = inputs_[1];
        auto y_batch = outputs_[0];

        size_t batch_size = y_batch->tensor_.Dim(0);

        for(int i=0; i<batch_size; ++i) {
            Tensor x1 = x1_batch->tensor_.Element(i);
            Tensor x2 = x2_batch->tensor_.Element(i);
            Tensor y = y_batch->tensor_.Element(i);

            CUDA_CHECK(cudaMemcpy(y.Data(), x1.Data(), x1.Size()*sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(y.Data()+x1.Size(), x2.Data(), x2.Size()*sizeof(float), cudaMemcpyDeviceToDevice));
        }
    };


    std::shared_ptr<CatLayer> MakeCatLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands) {
        CHECK(opt->inputs.size()==2) << "concatenation layer is supposed to accept 2 inputs";  
        CHECK(opt->outputs.size()==1) << "concatenation layer is supposed to generate 1 output";

        auto dim = opt->params.find("dim");
        CHECK(dim != opt->params.end()) << "concatenation layer missing parameter 'dim'";
        CHECK(dim->second.i == 1) << "concatenation layer supports only dim with value of 1";

        std::vector<std::string> inputs_name(2);
        for(int i=0; i<2; ++i) {
            inputs_name[i] = opt->inputs[i]->name;
        }

        std::vector<std::string> outputs_name = {opt->outputs[0]->name};

        auto input1 = operands.find(inputs_name[0]);
        CHECK(input1 != operands.end()) << "concatenation layer missing input operand1";

        auto input2 = operands.find(inputs_name[1]);
        CHECK(input2 != operands.end()) << "concatenation layer missing input operand2";

        auto output = operands.find(outputs_name[0]);
        CHECK(output != operands.end()) << "concatenation layer missing output operand";

        std::vector<std::shared_ptr<Operand>> inputs = {input1->second, input2->second};
        std::vector<std::shared_ptr<Operand>> outputs = {output->second};

        return std::make_shared<CatLayer>(opt->name, inputs_name, outputs_name, inputs, outputs, dim->second.i);
    };
}