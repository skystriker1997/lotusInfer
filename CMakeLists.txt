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


include_directories(include/)


file(GLOB OPT "source/operator/*.cu")
add_library(operator ${OPT})
target_link_libraries(operator PUBLIC glog::glog fmt::fmt CUDA::cudart CUDA::cublas)


add_library(tensor source/tensor.cpp)
target_include_directories(tensor PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(tensor PUBLIC glog::glog fmt::fmt CUDA::cudart xtensor)


file(GLOB LAYER "source/layer/*.cu" "source/layer/*.cpp")
add_library(layer ${LAYER})
target_link_libraries(layer PUBLIC operator tensor)


add_library(pnnx source/pnnx/ir.cpp source/pnnx/storezip.cpp source/pnnx/utils.cpp)


add_library(graph source/graph.cpp)
target_link_libraries(graph PUBLIC layer pnnx)


add_subdirectory(test)
add_subdirectory(demo/image_classification)


