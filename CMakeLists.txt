cmake_minimum_required(VERSION 3.17)

project(lotusinfer LANGUAGES CXX CUDA)

add_definitions(-DHAVE_CBLAS=1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)
find_package(Catch2 3 REQUIRED)
find_package (glog 0.7.1 REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)


include_directories(include/)


file(GLOB OPT "source/operator/*.cu")
add_library(operator ${OPT})
target_link_libraries(operator PUBLIC glog::glog fmt::fmt CUDA::cudart CUDA::cublas)

add_library(tensor source/tensor.cpp)
target_link_libraries(tensor PUBLIC glog::glog fmt::fmt CUDA::cudart)

add_subdirectory(test)


