project(cpu_skytosky_double)

find_package(OpenMP)
#find_package(Boost REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-DSKYTOSKY_DOUBLE -Wall -mavx -pthread -fopenmp -O3 --std=c++11")

add_library(cpu_skytosky_double SHARED ../interpolation_wrapper.cpp)
target_link_libraries(cpu_skytosky_double)

