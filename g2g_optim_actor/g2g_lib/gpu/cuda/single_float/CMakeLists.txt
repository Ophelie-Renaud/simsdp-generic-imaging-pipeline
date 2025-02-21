project(gpu_skytosky_single)

FIND_PACKAGE(CUDA REQUIRED)
find_package(OpenMP)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-DSKYTOSKY_SINGLE")
set(CUDA_NVCC_FLAGS "-DSKYTOSKY_SINGLE -O3")

cuda_add_library(gpu_skytosky_single SHARED ../interpolation_wrapper.cu)
target_link_libraries(gpu_skytosky_single)
