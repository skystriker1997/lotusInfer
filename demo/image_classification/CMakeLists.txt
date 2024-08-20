find_package( OpenCV REQUIRED )

add_executable(infer_cnn infer_cnn.cpp)
target_include_directories(infer_cnn PUBLIC ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(infer_cnn PUBLIC ${OpenCV_LIBS} graph)


