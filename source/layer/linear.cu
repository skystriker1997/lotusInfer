#include "layer/linear.hpp"


namespace lotus {

    LayerLinear::LayerLinear(
                            const std::string& name,
                            const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                            const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                            const std::vector<char>& weight, const uint32_t in_features, const uint32_t out_features,
                            const bool use_bias, const std::vector<char>& bias
                            )
    {   
        name_ = name;
        inputs_name_ = inputs_name;
        outputs_name_ = outputs_name;
        inputs_ = inputs;
        outputs_ = outputs;
        weight_ = Tensor({out_features, in_features}, weight);
        in_features_ = in_features;
        out_features_ = out_features;
        use_bias_ = use_bias;
        if(use_bias_) {
            bias_ = Tensor({out_features}, bias);
        } else {
            bias_ = Tensor();
        }
        af_ = ActivationFunction::NONE;
    };


    void LayerLinear::SetActivation(ActivationFunction af) {
        af_ = af;
    };

        
    void LayerLinear::Forward() {
        auto x_batch = inputs_[0];
        auto y_batch = outputs_[0];

        size_t batch_size = x_batch->tensor_.Dim(0);
        StreamPool pool(batch_size);
        for(int i=0; i<x_batch->tensor_.Dim(0); ++i) {
            Tensor x = x_batch->tensor_.Element(i);
            Tensor y = y_batch->tensor_.Element(i);

            if(i != 0) {
                pool.SetStream();
            }

            sfgemva<<<FGEMVA_GRID(out_features_), FGEMVA_BLOCK(), 0, pool.Stream()>>>(x.Data(), weight_.Data(), bias_.Data(), y.Data(), out_features_, in_features_, use_bias_, af_);

            
        }
        cudaDeviceSynchronize();
    };


    std::shared_ptr<LayerLinear> MakeLayerLinear(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands) {
        CHECK(opt->inputs.size()==1) << "linear layer gets more than 1 input";  
        CHECK(opt->outputs.size()==1) << "linear layer gets more than 1 output";

        auto out_features = opt->params.find("out_features");
        CHECK(out_features != opt->params.end()) << "linear layer fails to find parameter 'out_features'";

        auto in_features = opt->params.find("in_features");
        CHECK(in_features != opt->params.end()) << "linear layer fails to find parameter 'in_features'";
    
        auto use_bias = opt->params.find("bias");
        CHECK(use_bias != opt->params.end()) << "linear layer fails to find parameter 'bias'";

        auto weight = opt->attrs.find("weight");
        CHECK(weight != opt->attrs.end()) << "linear layer fails to find attribute 'kernel'";

        auto bias = opt->attrs.find("bias");
        if(use_bias->second.b) {
            CHECK(bias != opt->attrs.end()) << "linear layer fails to find attribute 'bias'";
        } 

        std::string input_name;
        if(opt->inputs[0]->producer->type=="nn.ReLU") {
            input_name = opt->inputs[0]->producer->inputs[0]->name;
        } else {
            input_name = opt->inputs[0]->name;
        }
        std::vector<std::string> inputs_name = {input_name};
        std::vector<std::string> outputs_name = {opt->outputs[0]->name};
        auto input = operands.find(inputs_name[0]);
        CHECK(input != operands.end()) << "linear layer fails to find the input operand";
        auto output = operands.find(outputs_name[0]);
        CHECK(output != operands.end()) << "linear layer fails to find the output operand";

        CHECK(input->second->tensor_.DimSize()==2) << "linear layer only supports vector matrix multiplication";
        std::vector<std::shared_ptr<Operand>> inputs = {input->second};
        std::vector<std::shared_ptr<Operand>> outputs = {output->second};

        std::vector<char> empty_bias {};

        return std::make_shared<LayerLinear>(opt->name,
                                             inputs_name, outputs_name,
                                             inputs, outputs,
                                             weight->second.data, in_features->second.i, out_features->second.i,
                                             use_bias->second.b, use_bias->second.b?bias->second.data:empty_bias
                                             );
    };
}