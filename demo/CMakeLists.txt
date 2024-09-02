find_package( OpenCV REQUIRED )

add_executable(image_classification image_classification.cpp)
target_include_directories(image_classification PUBLIC ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(image_classification PUBLIC ${OpenCV_LIBS} graph)

add_executable(image_segmentation image_segmentation.cpp)
target_include_directories(image_segmentation PUBLIC ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(image_segmentation PUBLIC ${OpenCV_LIBS} graph)


