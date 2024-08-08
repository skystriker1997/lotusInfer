#include "layer/conv2d.hpp"

namespace lotus {
    Conv2d::Conv2d( std::vector<char>& kernel, uint32_t k_num, uint32_t k_c, uint32_t k_h, uint32_t k_w, 
                std::vector<char>& bias, 
                uint32_t stride_h, uint32_t stride_w,
                uint32_t dilation_h, uint32_t dilation_w,
                uint32_t padding_h, uint32_t padding_w)
    {
        kernel_ = Tensor({k_num, k_c, k_h, k_w}, kernel);
        bias_ = Tensor({k_num}, bias);
        stride_h_ = stride_h;
        stride_w_ = stride_w;
        dilation_h_ = dilation_h;
        dilation_w_ = dilation_w;
        padding_h_ = padding_h;
        padding_w_ = padding_w;
    };
        
    void Conv2d::Forward() {
        auto x_batch = inputs_[0];
        auto y_batch = outputs_[0];

        size_t batch_size = x_batch->tensor_.Dim(0);
        StreamPool pool(batch_size);
        for(int i=0; i<x_batch->tensor_.Dim(0); ++i) {
            Tensor x = x_batch->tensor_.Element(i);
            Tensor y = y_batch->tensor_.Element(i);
            int padded_x_h = x.Dim(1) + 2*padding_h;
            int padded_x_w = x.Dim(2) + 2*padding_w;

            int y_c = y.Dim(0);
            int y_h = y.Dim(1);
            int y_w = y.Dim(2);

            int x_c = x.Dim(0);

            dim3 grid((y_h*y_w+127)/128, (y_c+127)/128);

            sconv2d<<<grid, 256, 0, pool.Stream()>>>( x.Data(), 
                                                    kernel_.Data(), 
                                                    bias_.Data(), 
                                                    y.Data(), 
                                                    kernel_.Dim(0), kernel_.Dim(2), kernel_.Dim(3), kernel_.Dim(1), 
                                                    padded_x_w, padded_x_h, x_c, 
                                                    y_w, y_h, y_c,
                                                    stride_h, stride_w,
                                                    dilation_h, dilation_w,
                                                    padding_h, padding_w
                                                    )
            pool.SetStream();
        }
        cudaDeviceSynchronize();
    };
}