#include "layer/layer_adaptive_avgpool2d.hpp"

namespace lotus {

    AdaptiveAvgpool2dLayer::AdaptiveAvgpool2dLayer( const std::string& name,
                                                    const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                                                    const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                                                    const uint32_t output_h, const uint32_t output_w)
    {   
        name_ = name;
        inputs_name_ = inputs_name;
        outputs_name_ = outputs_name;
        inputs_ = inputs;
        outputs_ = outputs;
        output_h_ = output_h;
        output_w_ = output_w;
    };


    void AdaptiveAvgpool2dLayer::Forward() {
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

            uint32_t x_c = x.Dim(0);
            uint32_t x_h = x.Dim(1);
            uint32_t x_w = x.Dim(2);

            uint32_t kernel_h = (x_h+output_h_-1)/output_h_;
            uint32_t kernel_w = (x_w+output_w_-1)/output_w_;

            float stride_h = (x_h-kernel_h)/output_h_;
            float stride_w = (x_w-kernel_w)/output_w_;

            sadaptive_avgpool2d<<<ADAPTIVE_AVGPOOL2D_GRID(x_c, output_h_, output_w_), ADAPTIVE_AVGPOOL2D_BLOCK(), 0, pool.Stream()>>>(x.Data(), y.Data(), 
                                                                                                                                      kernel_h, kernel_w, 
                                                                                                                                      x_c, x_h, x_w, 
                                                                                                                                      stride_h, stride_w, 
                                                                                                                                      output_h_, output_w_);

        }
        cudaDeviceSynchronize();
    };


    std::shared_ptr<AdaptiveAvgpool2dLayer> MakeAdaptiveAvgpool2dLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands) {

        CHECK(opt->inputs.size()==1) << "adaptive average pooling layer gets more than 1 input";  
        CHECK(opt->outputs.size()==1) << "adaptive average pooling layer gets more than 1 output";

        auto output_size = opt->params.find("output_size");
        CHECK(output_size != opt->params.end()) << "adaptive average pooling layer fails to find parameter 'output_size'";

        uint32_t output_h = output_size->second.ai[0];
        uint32_t output_w = output_size->second.ai[1]; 

        std::string input_name;
        if(opt->inputs[0]->producer->type=="nn.ReLU") {
            input_name = opt->inputs[0]->producer->inputs[0]->name;
        } else {
            input_name = opt->inputs[0]->name;
        }
        std::vector<std::string> inputs_name = {input_name};
        std::vector<std::string> outputs_name = {opt->outputs[0]->name};
        auto input = operands.find(inputs_name[0]);
        CHECK(input != operands.end()) << "adaptive average pooling layer fails to find the input operand";
        auto output = operands.find(outputs_name[0]);
        CHECK(output != operands.end()) << "adaptive average pooling layer fails to find the output operand";

        std::vector<std::shared_ptr<Operand>> inputs = {input->second};
        std::vector<std::shared_ptr<Operand>> outputs = {output->second};

        return std::make_shared<AdaptiveAvgpool2dLayer>(opt->name,
                                                        inputs_name, outputs_name,
                                                        inputs, outputs,
                                                        output_h, output_w
                                                        );
    };
}