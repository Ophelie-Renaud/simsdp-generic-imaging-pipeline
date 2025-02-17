cmake_minimum_required(VERSION 3.10)
project(g2g CXX)

include_directories(common)

option(ENABLE_NATIVE_TUNING "Enable architecture-specific tuning" ON)
option(ENABLE_FAST_MATH "Enable fast math optimizations" ON)

if(ENABLE_NATIVE_TUNING)
    set(TUNING_FLAGS "-march=native -mtune=native -std=c++11")
    message("Set -march and -mtune=native compiler flags")
else()
    set(TUNING_FLAGS "")
endif()

if(ENABLE_FAST_MATH)
    set(FAST_MATH_FLAG "-ffast-math")
    message("Set -ffast-math compiler flag")
else()
    set(FAST_MATH_FLAG "")
endif()

add_subdirectory(cpu)

# Vérifier si CUDA est installé avant d'ajouter le sous-dossier
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    add_subdirectory(gpu/cuda)
    message("CUDA found, enabling GPU support")
else()
    message("CUDA not found, skipping GPU support")
endif()

