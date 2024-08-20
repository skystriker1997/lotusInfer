find_package(xtensor REQUIRED)

add_executable(test_sfgemva test_sfused_gemv_add_bias.cu)
target_link_libraries(test_sfgemva PUBLIC operator)


add_executable(test_sconv2d test_sconv2d.cu)
target_include_directories(test_sconv2d PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(test_sconv2d PUBLIC operator xtensor)


add_executable(test_maxpool2d test_maxpool2d.cu)
target_include_directories(test_maxpool2d PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(test_maxpool2d PUBLIC operator xtensor)

add_executable(test_adaptive_avgpool2d test_adaptive_avgpool2d.cu)
target_include_directories(test_adaptive_avgpool2d PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(test_adaptive_avgpool2d PUBLIC operator xtensor)


add_executable(test_add test_add.cu)
target_include_directories(test_add PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(test_add PUBLIC operator xtensor)


