#include "layer/layer_adaptive_avgpool2d.hpp"


namespace lotus {

    AdaptiveAvgpool2dLayer::AdaptiveAvgpool2dLayer(const std::string& name,
                                                   const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                                                   const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                                                   const uint32_t output_h, const uint32_t output_w,
                                                   ActivationFunction af)
    {   
        name_ = name;
        inputs_name_ = inputs_name;
        outputs_name_ = outputs_name;
        inputs_ = inputs;
        outputs_ = outputs;
        output_h_ = output_h;
        output_w_ = output_w;
        af_ = af;
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

            AdaptiveAvgpool2d<<<MakeAAP2dGrid(x_c, output_h_, output_w_), MakeAAP2dBlock(), 0, pool.Stream()>>>(x.Data(), y.Data(), 
                                                                                                                  kernel_h, kernel_w, 
                                                                                                                  x_c, x_h, x_w, 
                                                                                                                  stride_h, stride_w, 
                                                                                                                  output_h_, output_w_,
                                                                                                                  af_);

        }
        cudaDeviceSynchronize();
    };


    std::shared_ptr<AdaptiveAvgpool2dLayer> MakeAdaptiveAvgpool2dLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands) {

        CHECK(opt->inputs.size()==1) << "adaptive average pooling layer is supposed to accept 1 input";  
        CHECK(opt->outputs.size()==1) << "adaptive average pooling layer is supposed to generate 1 output";

        auto output_size = opt->params.find("output_size");
        CHECK(output_size != opt->params.end()) << "adaptive average pooling layer missing parameter 'output_size'";

        uint32_t output_h = output_size->second.ai[0];
        uint32_t output_w = output_size->second.ai[1]; 
        
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
        CHECK(input != operands.end()) << "adaptive average pooling layer missing input operand";
        auto output = operands.find(outputs_name[0]);
        CHECK(output != operands.end()) << "adaptive average pooling layer missing output operand";

        std::vector<std::shared_ptr<Operand>> inputs = {input->second};
        std::vector<std::shared_ptr<Operand>> outputs = {output->second};

        return std::make_shared<AdaptiveAvgpool2dLayer>(opt->name,
                                                        inputs_name, outputs_name,
                                                        inputs, outputs,
                                                        output_h, output_w,
                                                        af);
    };
}