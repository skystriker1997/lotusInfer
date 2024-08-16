#pragma once

#include "layer/layer.hpp"
#include "operator/maxpool2d.cuh"


namespace lotus {
    class LayerMaxpool2d: public Layer {
        private:
        uint32_t kernel_h_;
        uint32_t kernel_w_;
        uint32_t stride_h_;
        uint32_t stride_w_;
        uint32_t padding_h_; 
        uint32_t padding_w_;

        public:
        LayerMaxpool2d(const std::string& name,
                                 const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                                 const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                                 const uint32_t kernel_h, const uint32_t kernel_w, 
                                 const uint32_t stride_h, const uint32_t stride_w,
                                 const uint32_t padding_h, const uint32_t padding_w);
        
        void Forward() override;
        ~LayerMaxpool2d() override = default;
    };


    std::shared_ptr<LayerMaxpool2d> MakeLayerMaxpool2d(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands);
}