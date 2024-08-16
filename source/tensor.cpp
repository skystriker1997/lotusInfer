#include "tensor.hpp"

namespace lotus {

    Tensor::Tensor(const std::vector<uint32_t>& shape) {
        size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
        data_ = MakeCudaShared<float>(size_);
        shape_ = shape;
        offset_ = 0;
    }


    Tensor::Tensor(const std::vector<uint32_t>& shape, const std::vector<char>& data) {
        size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
        std::vector<float> vec_fp32;
        CHECK(data.size() % 4 == 0) << fmt::format("error : data size is not divisibale by the size of fp32! at {}:{}", __FILE__, __LINE__);
        CHECK(data.size() / 4 == size_) << fmt::format("error : shape and data are not matching! at {}:{}", __FILE__, __LINE__);
        for(size_t i=0; i<data.size()/4; i++) {
            float fp32 = *((float*)data.data()+i);
            vec_fp32.emplace_back(fp32);
        }
        data_ = MakeCudaShared<float>(size_);
        CUDA_CHECK(cudaMemcpy(data_.get(), &vec_fp32[0], size_*4, cudaMemcpyHostToDevice));
        shape_ = shape;
        offset_ = 0;
    }


    Tensor::Tensor(Tensor&& other) noexcept {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        offset_ = other.offset_;
        size_ = other.size_;
    }


    Tensor& Tensor::operator=(Tensor&& rhs) noexcept {
        data_ = std::move(rhs.data_);
        shape_ = std::move(rhs.shape_);
        offset_ = rhs.offset_;
        size_ = rhs.size_;
        return *this;
    }

    void Tensor::AssignData(const std::vector<float>& data) {
        CHECK(size_ == data.size()) << "error to assign unmatching data size: " << std::to_string(size_) << " vs " << std::to_string(data.size());
        CUDA_CHECK(cudaMemcpy(data_.get()+offset_, &data[0], size_*4, cudaMemcpyHostToDevice));
    }


    void Tensor::Reshape(const std::vector<uint32_t>& shape) {
        CHECK(size_ == std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>())) << "tensor fails to reshape";
        shape_ = shape;
    }


    size_t Tensor::DimSize() const {
        return shape_.size();
    };


    uint32_t Tensor::Dim(size_t i) const {
        CHECK(i<shape_.size()) << fmt::format("error : index out of range at {}:{}", __FILE__, __LINE__);
        return shape_[i];
    };


    uint32_t Tensor::Size() const {
        return size_;
    }


    float* Tensor::Data() {
        return data_.get() + offset_;
    };


    Tensor Tensor::Element(uint32_t i) {
        CHECK(!shape_.empty()) << fmt::format("error : fetch element from a scalar at {}:{}", __FILE__, __LINE__);
        CHECK(i<shape_[0]) << fmt::format("error : index out of range at {}:{}", __FILE__, __LINE__);
        Tensor tmp = *this;
        tmp.shape_.erase(tmp.shape_.begin());
        uint32_t stride = std::accumulate(shape_.begin()+1, shape_.end(), 1, std::multiplies<uint32_t>());
        tmp.offset_ += stride*i;
        tmp.size_ = stride;
        return tmp;
    };

}