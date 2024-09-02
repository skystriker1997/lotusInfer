find_package(xtensor REQUIRED)

add_executable(test_fgemva test_fused_gemv_add_bias.cu)
target_link_libraries(test_fgemva PUBLIC operator)

add_executable(test_conv2d test_conv2d.cu)
target_include_directories(test_conv2d PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(test_conv2d PUBLIC operator xtensor)

add_executable(test_maxpool2d test_maxpool2d.cu)
target_include_directories(test_maxpool2d PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(test_maxpool2d PUBLIC operator xtensor)

add_executable(test_adaptive_avgpool2d test_adaptive_avgpool2d.cu)
target_include_directories(test_adaptive_avgpool2d PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(test_adaptive_avgpool2d PUBLIC operator xtensor)

add_executable(test_add test_add.cu)
target_include_directories(test_add PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(test_add PUBLIC operator xtensor)

add_executable(test_transposed_conv2d test_transposed_conv2d.cu)
target_include_directories(test_transposed_conv2d PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(test_transposed_conv2d PUBLIC operator xtensor)




