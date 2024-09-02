#include "layer/layer_transposed_conv2d.hpp"


namespace lotus {

    TransposedConv2dLayer::TransposedConv2dLayer(const std::string& name,
                                                 const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                                                 const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                                                 const std::vector<char>& kernel, 
                                                 const uint32_t k_num, const uint32_t k_c, const uint32_t k_h, const uint32_t k_w, 
                                                 const bool use_bias, const std::vector<char>& bias, 
                                                 const uint32_t stride_h, const uint32_t stride_w,
                                                 const uint32_t padding_h, const uint32_t padding_w,
                                                 ActivationFunction af)
    {   
        name_ = name;
        inputs_name_ = inputs_name;
        outputs_name_ = outputs_name;
        inputs_ = inputs;
        outputs_ = outputs;
        kernel_ = Tensor({k_num, k_c, k_h, k_w}, kernel);
        use_bias_ = use_bias;
        if(use_bias_) {
            bias_ = Tensor({k_num}, bias);
        } else {
            bias_ = Tensor();
        }
        stride_h_ = stride_h;
        stride_w_ = stride_w;
        padding_h_ = padding_h;
        padding_w_ = padding_w;
        af_ = af;
    };



    void TransposedConv2dLayer::Forward() {
        auto x_batch = inputs_[0];
        auto y_batch = outputs_[0];

        size_t batch_size = x_batch->tensor_.Dim(0);
        StreamPool pool(batch_size);
        for(uint32_t i=0; i<batch_size; ++i) {
            if(i != 0) {
                pool.SetStream();
            }

            Tensor x = x_batch->tensor_.Element(i);
            Tensor y = y_batch->tensor_.Element(i);

            uint32_t x_c = x.Dim(0);
            uint32_t x_h = x.Dim(1);
            uint32_t x_w = x.Dim(2);

            uint32_t y_c = y.Dim(0);
            uint32_t y_h = y.Dim(1);
            uint32_t y_w = y.Dim(2);

            TransConv2d<<<MakeTransposedConv2dGrid(y_c, y_h, y_w, padding_h_, padding_w_),MakeTransposedConv2dBlock(), 0, pool.Stream()>>>(x.Data(), 
                                                                                                                                kernel_.Data(), 
                                                                                                                                use_bias_, bias_.Data(), 
                                                                                                                                y.Data(), 
                                                                                                                                kernel_.Dim(0), kernel_.Dim(1), kernel_.Dim(2), kernel_.Dim(3), 
                                                                                                                                x_c, x_h, x_w,  
                                                                                                                                y_c, y_h, y_w,
                                                                                                                                stride_h_, stride_w_,
                                                                                                                                padding_h_, padding_w_,
                                                                                                                                af_);

        }
        cudaDeviceSynchronize();
    };


    std::shared_ptr<TransposedConv2dLayer> MakeTransposedConv2dLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands) {
        CHECK(opt->inputs.size()==1) << "transposed conv2d layer is supposed to accept 1 input";  
        CHECK(opt->outputs.size()==1) << "transposed conv2d layer is supposed to generate 1 output";

        auto groups = opt->params.find("groups");
        CHECK(groups != opt->params.end()) << "transposed conv2d layer missing parameter 'groups'";
        CHECK(groups->second.i == 1) << "transposed conv2d layer supports only kernel groups with value of 1";

        auto dilation = opt->params.find("dilation");
        CHECK(dilation != opt->params.end()) << "transposed conv2d layer missing parameter 'dilation'";
        CHECK(dilation->second.ai[0]==1 && dilation->second.ai[1]==1) << "transposed conv2d layer supports only dilation with a value of 1";

        auto in_channels = opt->params.find("in_channels");
        CHECK(in_channels != opt->params.end()) << "transposed conv2d layer missing parameter 'in_channels'";

        auto out_channels = opt->params.find("out_channels");
        CHECK(out_channels != opt->params.end()) << "transposed conv2d layer missing parameter 'out_channels'";

        auto padding = opt->params.find("padding");
        CHECK(padding != opt->params.end()) << "transposed conv2d layer missing parameter 'padding'";
        CHECK(padding->second.ai[0]==0 && padding->second.ai[1]==0) << "transposed conv2d layer supports only input padding with a value of 0";

        auto output_padding = opt->params.find("output_padding");
        CHECK(padding != opt->params.end()) << "transposed conv2d layer missing parameter 'output_padding'";
    
        auto use_bias = opt->params.find("bias");
        CHECK(use_bias != opt->params.end()) << "transposed conv2d layer missing parameter 'bias'";

        auto stride = opt->params.find("stride");
        CHECK(stride != opt->params.end()) << "transposed conv2d layer missing parameter 'stride'";

        auto kernel_size = opt->params.find("kernel_size");
        CHECK(kernel_size != opt->params.end()) << "transposed conv2d layer missing parameter 'kernel_size'";

        auto kernel = opt->attrs.find("weight");
        CHECK(kernel != opt->attrs.end()) << "transposed conv2d layer missing attribute 'kernel'";

        auto bias = opt->attrs.find("bias");
        if(use_bias->second.b) {
            CHECK(bias != opt->attrs.end()) << "transposed conv2d layer missing attribute 'bias'";
        }

        uint32_t k_num = out_channels->second.i;
        uint32_t k_c = in_channels->second.i;
        uint32_t k_h = kernel_size->second.ai[0];
        uint32_t k_w = kernel_size->second.ai[1];
        uint32_t stride_h = stride->second.ai[0];
        uint32_t stride_w = stride->second.ai[1];
        uint32_t padding_h = output_padding->second.ai[0];
        uint32_t padding_w = output_padding->second.ai[1];    

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
        CHECK(input != operands.end()) << "transposed conv2d layer missing the input operand";
        auto output = operands.find(outputs_name[0]);
        CHECK(output != operands.end()) << "transposed conv2d layer missing the output operand";

        std::vector<std::shared_ptr<Operand>> inputs = {input->second};
        std::vector<std::shared_ptr<Operand>> outputs = {output->second};

        std::vector<char> empty_bias {};

        return std::make_shared<TransposedConv2dLayer>(opt->name,
                                                       inputs_name, outputs_name,
                                                       inputs, outputs,
                                                       kernel->second.data,
                                                       k_num, k_c, k_h, k_w,
                                                       use_bias->second.b, use_bias->second.b?bias->second.data:empty_bias,
                                                       stride_h, stride_w,
                                                       padding_h, padding_w,
                                                       af);
    };
}