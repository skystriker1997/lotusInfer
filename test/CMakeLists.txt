# add_executable(test_sgemm test_sgemm.cu)
# target_link_libraries(test_sgemm PUBLIC operator)


add_executable(test_sfgemva test_sfused_gemv_add.cu)
target_link_libraries(test_sfgemva PUBLIC operator)


add_executable(test_sconv2d test_sconv2d.cu)
target_include_directories(test_sconv2d PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(test_sconv2d PUBLIC operator xtensor xtensor-blas ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})