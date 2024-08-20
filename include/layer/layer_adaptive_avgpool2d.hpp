#pragma once

#include "layer/layer.hpp"
#include "operator/adaptive_avgpool2d.cuh"


namespace lotus {
    class AdaptiveAvgpool2dLayer: public Layer {
        private:
        uint32_t output_h_;
        uint32_t output_w_;

        public:
        AdaptiveAvgpool2dLayer( const std::string& name,
                                const std::vector<std::string>& inputs_name, const std::vector<std::string>& outputs_name,
                                const std::vector<std::shared_ptr<Operand>>& inputs, const std::vector<std::shared_ptr<Operand>>& outputs,
                                const uint32_t output_h, const uint32_t output_w);
        
        void Forward() override;
        ~AdaptiveAvgpool2dLayer() override = default;
    };


    std::shared_ptr<AdaptiveAvgpool2dLayer> MakeAdaptiveAvgpool2dLayer(pnnx::Operator *opt, const std::map<std::string, std::shared_ptr<Operand>>& operands);
}