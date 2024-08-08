#pragma once

#include "lotus_utils.hpp"
#include <vector>
#include <numeric>



namespace lotus {

    class Tensor {
        private:
        std::vector<uint32_t> shape_;
        std::vector<uint32_t> strides_;
        std::shared_ptr<float> data_;
        uint32_t offset_;
        bool is_batch_;
        
        public:
        Tensor() = delete;
        Tensor(const std::vector<uint32_t>& shape);
        Tensor(const std::vector<uint32_t>& shape, std::vector<char>& data); 
        Tensor(const Tensor& other) = default;
        Tensor(Tensor&& other) noexcept;

        ~Tensor() = default;
        Tensor& operator=(const Tensor& rhs) = default;
        Tensor& operator=(Tensor&& rhs) noexcept;

        size_t DimSize() const;
        uint32_t Dim(uint32_t i) const;
        uint32_t Stride(uint32_t i) const;
        float* Data();
        Tensor Element(uint32_t i);

        static std::vector<uint32_t> ComputeStrides(const std::vector<uint32_t>& shape);
    };

}
    