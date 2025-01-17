#include "layer/linear.hpp"



namespace lotus {

    LinearLayer::LinearLayer(const std::string& name,
                             const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                             const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                             const std::vector<char>& weight, const uint32_t in_features, const uint32_t out_features,
                             const bool use_bias, const std::vector<char>& bias,
                             ActivationFunction af)
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
        af_ = af;
    };


        
    void LinearLayer::Forward() {
        auto x_batch = inputs_[0];
        auto y_batch = outputs_[0];

        size_t batch_size = x_batch->tensor_.Dim(0);
        StreamPool pool(batch_size);
        for(int i=0; i<batch_size; ++i) {
            Tensor x = x_batch->tensor_.Element(i);
            Tensor y = y_batch->tensor_.Element(i);

            if(i != 0) {
                pool.SetStream();
            }

            Gemm<<<MakeGemmGrid(x.Dim(0), out_features_), MakeGemmBlock(), 0, pool.Stream()>>>(x.Data(), weight_.Data(), use_bias_, bias_.Data(), y.Data(), x.Dim(0), out_features_, in_features_, af_);
            
        }
        cudaDeviceSynchronize();
    };


    std::shared_ptr<LinearLayer> MakeLinearLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands) {
        CHECK(opt->inputs.size()==1) << "linear layer is supposed to accept 1 input";  
        CHECK(opt->outputs.size()==1) << "linear layer is supposed to generate 1 output";

        auto out_features = opt->params.find("out_features");
        CHECK(out_features != opt->params.end()) << "linear layer missing parameter 'out_features'";

        auto in_features = opt->params.find("in_features");
        CHECK(in_features != opt->params.end()) << "linear layer missing parameter 'in_features'";
    
        auto use_bias = opt->params.find("bias");
        CHECK(use_bias != opt->params.end()) << "linear layer missing parameter 'bias'";

        auto weight = opt->attrs.find("weight");
        CHECK(weight != opt->attrs.end()) << "linear layer missing attribute 'kernel'";

        auto bias = opt->attrs.find("bias");
        if(use_bias->second.b) {
            CHECK(bias != opt->attrs.end()) << "linear layer missing attribute 'bias'";
        } 

        std::vector<std::string> inputs_name = {opt->inputs[0]->name};

        std::string output_name;
        ActivationFunction af;
        if(opt->outputs[0]->consumers[0]->type=="nn.ReLU") {
            af = ActivationFunction::RELU;
            output_name = opt->outputs[0]->consumers[0]->outputs[0]->name;
        } else if(opt->outputs[0]->consumers[0]->type=="F.sigmoid") {
            af = ActivationFunction::SIGMOID;
            output_name = opt->outputs[0]->consumers[0]->outputs[0]->name;
        } else {
            af = ActivationFunction::NONE;
            output_name = opt->outputs[0]->name;
        }
        std::vector<std::string> outputs_name = {output_name};

        auto input = operands.find(inputs_name[0]);
        CHECK(input != operands.end()) << "linear layer missing the input operand";
        auto output = operands.find(outputs_name[0]);
        CHECK(output != operands.end()) << "linear layer missing the output operand";

        std::vector<std::shared_ptr<Operand>> inputs = {input->second};
        std::vector<std::shared_ptr<Operand>> outputs = {output->second};

        std::vector<char> empty_bias {};

        return std::make_shared<LinearLayer>(opt->name,
                                             inputs_name, outputs_name,
                                             inputs, outputs,
                                             weight->second.data, in_features->second.i, out_features->second.i,
                                             use_bias->second.b, use_bias->second.b?bias->second.data:empty_bias,
                                             af);
    };
}