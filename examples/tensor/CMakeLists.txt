add_executable(nenb-test nenb.cpp)
target_compile_options(nenb-test PRIVATE -g)
target_compile_options(nenb-test PRIVATE -O0)
target_link_libraries(nenb-test PRIVATE ggml)

add_executable(sgemm-test sgemm.cpp)
target_compile_options(sgemm-test PRIVATE -g)
target_compile_options(sgemm-test PRIVATE -O0)
target_link_libraries(sgemm-test PRIVATE ggml)