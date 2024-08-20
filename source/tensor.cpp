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
        CHECK(data.size() % 4 == 0) << "the size of std::vector<char> is not divisibale by the size of fp32";
        CHECK(data.size() / 4 == size_) << "the size of std::vector<char> is not as supposed by the tensor constructor: " << size_ << " vs " << data.size()/4;
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
        CHECK(size_ == data.size()) << "the size of std::vector<float> is not as supposed by the tensor data assignment: " << size_ << " vs " << data.size();
        CUDA_CHECK(cudaMemcpy(data_.get()+offset_, &data[0], size_*4, cudaMemcpyHostToDevice));
    }


    void Tensor::Reshape(const std::vector<uint32_t>& shape) {
        CHECK(size_ == std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>())) << "tensor fails to reshape because of unmatching size";
        shape_ = shape;
    }


    size_t Tensor::DimSize() const {
        return shape_.size();
    };


    uint32_t Tensor::Dim(size_t i) const {
        CHECK(i<shape_.size()) << "index out of range when try to get the dimension size";
        return shape_[i];
    };


    uint32_t Tensor::Size() const {
        return size_;
    }


    float* Tensor::Data() {
        return data_.get() + offset_;
    };


    Tensor Tensor::Element(uint32_t i) {
        CHECK(!shape_.empty()) << "cannot fetch any element from a scalar";
        CHECK(i<shape_[0]) << "index out of range when try to fetch an element from a tensor";
        Tensor tmp = *this;
        tmp.shape_.erase(tmp.shape_.begin());
        uint32_t stride = std::accumulate(shape_.begin()+1, shape_.end(), 1, std::multiplies<uint32_t>());
        tmp.offset_ += stride*i;
        tmp.size_ = stride;
        return tmp;
    };

    void Tensor::Print() {
        std::vector<float> tmp(size_);
        CUDA_CHECK(cudaMemcpy(&tmp[0], this->Data(), size_*sizeof(float), cudaMemcpyDeviceToHost));
        auto data = xt::adapt(tmp, shape_);
        std::cout << data << std::endl;
    }

}