#include "layer/layer_maxpool2d.hpp"

namespace lotus {

    Maxpool2dLayer::Maxpool2dLayer( const std::string& name,
                                    const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                                    const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                                    const uint32_t kernel_h, const uint32_t kernel_w, 
                                    const uint32_t stride_h, const uint32_t stride_w,
                                    const uint32_t padding_h, const uint32_t padding_w
                                   )
    {   
        name_ = name;
        inputs_name_ = inputs_name;
        outputs_name_ = outputs_name;
        inputs_ = inputs;
        outputs_ = outputs;
        kernel_h_ = kernel_h;
        kernel_w_ = kernel_w;
        stride_h_ = stride_h;
        stride_w_ = stride_w;
        padding_h_ = padding_h;
        padding_w_ = padding_w;
    };


    void Maxpool2dLayer::Forward() {
        auto x_batch = inputs_[0];
        auto y_batch = outputs_[0];

        size_t batch_size = x_batch->tensor_.Dim(0);
        StreamPool pool(batch_size);
        
        for(int i=0; i<batch_size; ++i) {
            if(i != 0) {
                pool.SetStream();
            }
            Tensor x = x_batch->tensor_.Element(i);
            Tensor y = y_batch->tensor_.Element(i);

            uint32_t padded_x_h = x.Dim(1) + 2*padding_h_;
            uint32_t padded_x_w = x.Dim(2) + 2*padding_w_;

            uint32_t x_c = x.Dim(0);
            uint32_t y_h = y.Dim(1);
            uint32_t y_w = y.Dim(2);

            smaxpool2d<<<MakeMP2dGrid(x_c, y_h, y_w), MakeMP2dBlock(), 0, pool.Stream()>>>(x.Data(), 
                                                                                           y.Data(),
                                                                                           kernel_h_, kernel_w_,
                                                                                           x_c, padded_x_h, padded_x_w,
                                                                                           padding_h_, padding_w_, 
                                                                                           stride_h_, stride_w_,
                                                                                           y_h, y_w);
        }
        cudaDeviceSynchronize();
    };


    std::shared_ptr<Maxpool2dLayer> MakeMaxpool2dLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands) {
        CHECK(opt->inputs.size()==1) << "maxpool2d layer gets more than 1 input";  
        CHECK(opt->outputs.size()==1) << "maxpool2d layer gets more than 1 output";

        auto dilation = opt->params.find("dilation");
        CHECK(dilation != opt->params.end()) << "maxpool2d layer fails to find parameter 'dilation'";
        CHECK(dilation->second.ai[0]==1 && dilation->second.ai[1]==1) << "maxpool2d layer does not support kernel dilation greater than 1";

        auto padding = opt->params.find("padding");
        CHECK(padding != opt->params.end()) << "maxpool2d layer fails to find parameter 'padding'";

        auto stride = opt->params.find("stride");
        CHECK(stride != opt->params.end()) << "maxpool2d layer fails to find parameter 'stride'";

        auto kernel_size = opt->params.find("kernel_size");
        CHECK(kernel_size != opt->params.end()) << "maxpool2d layer fails to find parameter 'kernel_size'";

        auto ceil_mode = opt->params.find("ceil_mode");
        CHECK(ceil_mode != opt->params.end()) << "maxpool2d layer fails to find parameter 'ceil_mode'";
        CHECK(ceil_mode->second.b==false) << "maxpool2d layer does not support ceil mode";

        auto return_indices = opt->params.find("return_indices");
        CHECK(return_indices != opt->params.end()) << "maxpool2d layer fails to find parameter 'return_indices'";
        CHECK(return_indices->second.b==false) << "maxpool2d layer does not support returning indices";

        uint32_t k_h = kernel_size->second.ai[0];
        uint32_t k_w = kernel_size->second.ai[1];
        uint32_t stride_h = stride->second.ai[0];
        uint32_t stride_w = stride->second.ai[1];
        uint32_t padding_h = padding->second.ai[0];
        uint32_t padding_w = padding->second.ai[1];    

        std::string input_name;
        if(opt->inputs[0]->producer->type=="nn.ReLU") {
            input_name = opt->inputs[0]->producer->inputs[0]->name;
        } else {
            input_name = opt->inputs[0]->name;
        }
        std::vector<std::string> inputs_name = {input_name};
        std::vector<std::string> outputs_name = {opt->outputs[0]->name};
        auto input = operands.find(inputs_name[0]);
        CHECK(input != operands.end()) << "maxpool2d layer fails to find the input operand";
        auto output = operands.find(outputs_name[0]);
        CHECK(output != operands.end()) << "maxpool2d layer fails to find the output operand";

        std::vector<std::shared_ptr<Operand>> inputs = {input->second};
        std::vector<std::shared_ptr<Operand>> outputs = {output->second};

        return std::make_shared<Maxpool2dLayer>(opt->name,
                                                inputs_name, outputs_name,
                                                inputs, outputs,
                                                k_h, k_w,
                                                stride_h, stride_w,
                                                padding_h, padding_w);
    };
}