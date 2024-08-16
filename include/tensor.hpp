#pragma once
#include "lotus_utils.hpp"



namespace lotus {

    class Tensor {
        private:
        std::vector<uint32_t> shape_;
        std::shared_ptr<float> data_;
        uint32_t offset_;
        uint32_t size_;
        
        public:
        Tensor() = default;
        Tensor(const std::vector<uint32_t>& shape);
        Tensor(const std::vector<uint32_t>& shape, const std::vector<char>& data); 
        Tensor(const Tensor& other) = default;
        Tensor(Tensor&& other) noexcept;

        ~Tensor() = default;
        Tensor& operator=(const Tensor& rhs) = default;
        Tensor& operator=(Tensor&& rhs) noexcept;

        void AssignData(const std::vector<float>& data);
        void Reshape(const std::vector<uint32_t>& shape);
        size_t DimSize() const;
        uint32_t Dim(size_t i) const;
        uint32_t Size() const;
        float* Data();
        Tensor Element(uint32_t i);

    };

}
    