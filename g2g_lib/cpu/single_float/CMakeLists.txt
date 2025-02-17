project(cpu_skytosky_single)

find_package(OpenMP)
#find_package(Boost REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-DSKYTOSKY_SINGLE -Wall -mavx -pthread -fopenmp -O3 --std=c++11")

add_library(cpu_skytosky_single SHARED ../interpolation_wrapper.cpp)
target_link_libraries(cpu_skytosky_single)

