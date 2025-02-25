cmake_minimum_required(VERSION 3.10.2)

# *******************************************
# ************* CMake Content ***************
# *******************************************
# This CMake create a workspace containing the following projects
#
# Programs
#  - SEP
#  - SEP GPU version



# Add definition for relative path into project
add_definitions( -DPROJECT_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/..")
set(PROJECT_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/..")


if (DEFINED BUFFER_NAME AND DEFINED NB_BIT AND DEFINED TEST_SUITE)
    set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D${BUFFER_NAME}=${NB_BIT} -DTEST_SUITE")
    message("${ADDITIONAL_C_FLAGS}")
endif()


project(SEP_Pipeline_GPU C CXX CUDA)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -g -O3 -Wall ${ADDITIONAL_C_FLAGS}")

find_package(CUDA 10.1 REQUIRED)
message(STATUS ">>> Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

set(
CUDA_NVCC_FLAGS
${CUDA_NVCC_FLAGS};
-std=c++11 -arch=sm_60 -Xptxas -O3 -restrict -D_FORCE_INLINES -D_DEBUG -lineinfo
)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/generated)
include_directories(${PROJECT_ROOT_PATH}/include ${PROJECT_ROOT_PATH}/generated)


file(
    GLOB
    source_files_GPU
    ../src/*.c
    ../src/GPU/*.cu
    ../generated/*.cpp
    ../generated/*.c
)

# message("source_files_GPU : ${source_files_GPU}")
cuda_add_executable(SEP_Pipeline_GPU ${source_files_GPU})
target_link_libraries(SEP_Pipeline_GPU m cudart cusolver cufft) # math, CUDA FFT and SVD