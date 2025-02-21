project(gpu_skytosky_double)

FIND_PACKAGE(CUDA REQUIRED)
find_package(OpenMP)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-DSKYTOSKY_DOUBLE")
set(CUDA_NVCC_FLAGS "-DSKYTOSKY_DOUBLE -O3")

cuda_add_library(gpu_skytosky_double SHARED ../interpolation_wrapper.cu)
target_link_libraries(gpu_skytosky_double)
