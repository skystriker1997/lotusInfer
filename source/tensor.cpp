#include "tensor.hpp"

namespace lotus {

    Tensor::Tensor(const std::vector<uint32_t>& shape) {
        strides_ = ComputeStrides(shape);
        uint32_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
        data_ = MakeCudaShared<float>(size);
        shape_ = shape;
        offset_ = 0;
        is_batch_ = true;
    }


    Tensor::Tensor(const std::vector<uint32_t>& shape, std::vector<char>& data) {
        strides_ = ComputeStrides(shape);
        uint32_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
        std::vector<float> vec_fp32;
        CHECK(data.size() % 4 == 0) << fmt::format("error : data size is not divisibale by the size of fp32! at {}:{}", __FILE__, __LINE__);
        CHECK(data.size() / 4 == size) << fmt::format("error : shape and data are not matching! at {}:{}", __FILE__, __LINE__);
        for(size_t i=0; i<data.size()/4; i++) {
            float fp32 = *((float*)data.data()+i);
            vec_fp32.emplace_back(fp32);
        }
        data_ = MakeCudaShared<float>(size);
        CUDA_CHECK(cudaMemcpy(data_.get(), &vec_fp32[0], size*4, cudaMemcpyHostToDevice));
        shape_ = shape;
        offset_ = 0;
        is_batch_ = true;
    }


    Tensor::Tensor(Tensor&& other) noexcept {
        data_ = std::move(other.data_);
        strides_ = std::move(other.strides_);
        shape_ = std::move(other.shape_);
        offset_ = other.offset_;
        is_batch_ = other.is_batch_;
    }


    Tensor& Tensor::operator=(Tensor&& rhs) noexcept {
        data_ = std::move(rhs.data_);
        strides_ = std::move(rhs.strides_);
        shape_ = std::move(rhs.shape_);
        offset_ = rhs.offset_;
        is_batch_ = rhs.is_batch_;
        return *this;
    }


    size_t Tensor::DimSize() const {
        return shape_.size();
    };


    uint32_t Tensor::Dim(uint32_t i) const {
        CHECK(i<shape_.size()) << fmt::format("error : index out of range at {}:{}", __FILE__, __LINE__);
        return shape_[i];
    };


    uint32_t Tensor::Stride(uint32_t i) const {
        CHECK(i<shape_.size()) << fmt::format("error : index out of range at {}:{}", __FILE__, __LINE__);
        return strides_[i];
    };


    float* Tensor::Data() {
        return data_.get() + offset_;
    };


    Tensor Tensor::Element(uint32_t i) {
        CHECK(!shape_.empty()) << fmt::format("error : fetch element from a scalar at {}:{}", __FILE__, __LINE__);
        CHECK(i<shape_[0]) << fmt::format("error : index out of range at {}:{}", __FILE__, __LINE__);
        Tensor tmp = *this;
        tmp.shape_.erase(tmp.shape_.begin());
        tmp.strides_.erase(tmp.strides_.begin());
        tmp.offset_ += strides_[0]*i;
        return tmp;
    };


    std::vector<uint32_t> Tensor::ComputeStrides(const std::vector<uint32_t>& shape) {
        size_t len = shape.size();
        CHECK(len > 0) << fmt::format("empty shape of tensor! at {}:{}", __FILE__, __LINE__);
        size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
        CHECK(size > 0) << fmt::format("non-positive dimension value! at {}:{}", __FILE__, __LINE__);
        std::vector<uint32_t> strides = shape;

        for(size_t i=0; i<len-1; i++) {
            strides[i] = std::accumulate(shape.begin()+i+1, shape.end(), 1, std::multiplies<uint32_t>());
        }

        strides[len-1] = 1;
    
        return strides;
    }


}