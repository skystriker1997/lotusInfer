find_package( OpenCV REQUIRED )

add_executable(infer_alexnet infer_alexnet.cpp)
target_include_directories(infer_alexnet PUBLIC ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(infer_alexnet PUBLIC ${OpenCV_LIBS} graph)
