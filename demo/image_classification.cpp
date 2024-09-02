#include "graph.hpp"
#include <opencv2/opencv.hpp>
#include "xtensor/xio.hpp"
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>
#include <chrono>


void PreprocessImage(cv::Mat &image, std::vector<float> &result) {

    cv::resize(image, image, cv::Size(256, 256));

    // cv::Mat rgb_image;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3, 1.f / 255.f);
    
    int center_x = image.cols / 2;
    int center_y = image.rows / 2;

    int x = center_x - 224 / 2;
    int y = center_y - 224 / 2;

    x = std::max(0, x);
    y = std::max(0, y);

    cv::Rect crop_region(x, y, 224, 224);

    cv::Mat image_crop = image(crop_region);

    cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406); 
    cv::Scalar stddev = cv::Scalar(0.229, 0.224, 0.225); 

    cv::Mat channels[3];
    cv::split(image_crop, channels); 

    channels[0] = (channels[0] - mean[0]) / stddev[0]; 
    channels[1] = (channels[1] - mean[1]) / stddev[1]; 
    channels[2] = (channels[2] - mean[2]) / stddev[2]; 

    memcpy(&result[0], channels[0].data, sizeof(float)*(224*224));
    memcpy(&result[224*224], channels[1].data, sizeof(float)*(224*224));
    memcpy(&result[224*224*2], channels[2].data, sizeof(float)*(224*224));

}



int main(int argc, char* argv[]) {

    std::string image_path = argv[1];
    std::string param_path = argv[2];
    std::string bin_path = argv[3];
    std::string image_classes = argv[4];

    cv::Mat image = cv::imread(image_path);

    CHECK(!image.empty()) << "error loading image";

    std::vector<float> data(224*224*3);
    PreprocessImage(image, data);

    // time
    lotus::Graph graph(param_path, bin_path, 1);

    graph.InitialiseInput(data);

    auto start = std::chrono::high_resolution_clock::now();
    graph.Forward();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds to complete inference" << std::endl;

    std::shared_ptr<lotus::Operand> output = graph.Output();

    std::vector<float> tmp(1000);
    
    CUDA_CHECK(cudaMemcpy(&tmp[0], output->tensor_.Data(), 1000 * sizeof(float), cudaMemcpyDeviceToHost));

    auto logits = xt::adapt(tmp, {1000});

    float maximum = xt::amax(logits)();

    xt::xarray<float> exp = xt::exp(logits - maximum);

    xt::xarray<float> probabilities = exp / xt::sum(exp)();

    std::ifstream input_file(image_classes);  
    std::vector<std::string> classes;        
    std::string line;

    CHECK(input_file) << "error opening image classes file";

    while (std::getline(input_file, line)) {
        classes.push_back(line);
    }

    input_file.close();

    for(int i=0; i<5; ++i) {
        auto index = xt::argmax(probabilities)();
        std::cout << "with " << probabilities(index)*100 << " percent likelyhood to be " << classes[index] << std::endl;
        probabilities(index) = 0;
    }
    
    return 0;
}