#include "graph.hpp"
#include <opencv2/opencv.hpp>
#include "xtensor/xio.hpp"
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>
#include <chrono>


void PreprocessImage(cv::Mat &image, std::vector<float> &result) {

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(256, 256));

    image.convertTo(image, CV_32FC3, 1.f/255.f);

    cv::Scalar mean; 
    cv::Scalar stddev;
    cv::meanStdDev(image, mean, stddev);

    cv::Mat channels[3];
    cv::split(image, channels); 
    
    for(int i=0; i<3; ++i) {
        channels[i] = (channels[i]-mean[i]) / stddev[i];
    }
    memcpy(&result[0], channels[0].data, sizeof(float)*(256*256));
    memcpy(&result[256*256], channels[1].data, sizeof(float)*(256*256));
    memcpy(&result[256*256*2], channels[2].data, sizeof(float)*(256*256));
}



int main(int argc, char* argv[]) {

    std::string image_path = argv[1];
    std::string param_path = argv[2];
    std::string bin_path = argv[3];

    cv::Mat image = cv::imread(image_path);

    CHECK(!image.empty()) << "error loading image";

    std::vector<float> data(256*256*3);
    PreprocessImage(image, data);

    lotus::Graph graph(param_path, bin_path, 1);

    graph.InitialiseInput(data);

    auto start = std::chrono::high_resolution_clock::now();
    graph.Forward();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds to complete inference" << std::endl;

    std::shared_ptr<lotus::Operand> output = graph.Output();

    std::vector<float> tmp(256*256);
    
    CUDA_CHECK(cudaMemcpy(&tmp[0], output->tensor_.Data(), 256*256*sizeof(float), cudaMemcpyDeviceToHost));

    cv::Mat result(256, 256, CV_32F, tmp.data());
    cv::imwrite("abnormal_organization.png", result*255.f);
    
    return 0;
}