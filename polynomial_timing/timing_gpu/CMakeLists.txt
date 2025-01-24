cmake_minimum_required(VERSION 3.10)

# *******************************************
# ************* CMake Content ***************
# *******************************************
# This CMake create a workspace containing the following projects
#
# Programs
#  - SEP
#  - SEP GPU version



# Add definition for relative path into project
add_definitions( -DPROJECT_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

#TEST_SUITE
if (DEFINED TEST_SUITE)
    set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D TEST_SUITE=${TEST_SUITE}")
    message("${ADDITIONAL_C_FLAGS}")
endif()

#PSF_STORAGE_SIZE
if (DEFINED PSF_STORAGE_SIZE)
    set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D PSF_STORAGE_SIZE=${PSF_STORAGE_SIZE}")
    message("${ADDITIONAL_C_FLAGS}")

    if (DEFINED PSF_STORAGE_TYPE)
        set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D PSF_STORAGE_TYPE=${PSF_STORAGE_TYPE}")
        message("${ADDITIONAL_C_FLAGS}")
    endif()
endif()

#VIS_UVW_COORD_STORAGE_SIZE
if (DEFINED VIS_UVW_COORD_STORAGE_SIZE)
    set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D VIS_UVW_COORD_STORAGE_SIZE=${VIS_UVW_COORD_STORAGE_SIZE}")
    message("${ADDITIONAL_C_FLAGS}")

    if (DEFINED VIS_UVW_COORD_STORAGE_TYPE)
        set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D VIS_UVW_COORD_STORAGE_TYPE=${VIS_UVW_COORD_STORAGE_TYPE}")
        message("${ADDITIONAL_C_FLAGS}")
    endif()
endif()



#GAINS_APPLY_STORAGE_SIZE and FIFO_POP_VIS_STORAGE_SIZE
if (DEFINED GAINS_APPLY_STORAGE_SIZE)
    set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D GAINS_APPLY_STORAGE_SIZE=${GAINS_APPLY_STORAGE_SIZE}")
    set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D FIFO_POP_VIS_STORAGE_SIZE=${GAINS_APPLY_STORAGE_SIZE}")
    message("${ADDITIONAL_C_FLAGS}")

    if (DEFINED GAINS_APPLY_STORAGE_TYPE)
        set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D GAINS_APPLY_STORAGE_TYPE=${GAINS_APPLY_STORAGE_TYPE}")
        message("${ADDITIONAL_C_FLAGS}")
    endif()
endif()

#GRIDDING_STORAGE_SIZE
if (DEFINED GRIDDING_STORAGE_SIZE)
    set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D GRIDDING_STORAGE_SIZE=${GRIDDING_STORAGE_SIZE}")
    message("${ADDITIONAL_C_FLAGS}")

    if (DEFINED GRIDDING_STORAGE_TYPE)
        set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D GRIDDING_STORAGE_TYPE=${GRIDDING_STORAGE_TYPE}")
        message("${ADDITIONAL_C_FLAGS}")
    endif()
endif()

#FFT_C2C_STORAGE_SIZE
if (DEFINED FFT_C2C_STORAGE_SIZE)
    set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D FFT_C2C_STORAGE_SIZE=${FFT_C2C_STORAGE_SIZE}")
    message("${ADDITIONAL_C_FLAGS}")

    if (DEFINED FFT_C2C_STORAGE_TYPE)
        set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D FFT_C2C_STORAGE_TYPE=${FFT_C2C_STORAGE_TYPE}")
        message("${ADDITIONAL_C_FLAGS}")
    endif()
endif()

#CUFFT_STORAGE_SIZE
if (DEFINED CUFFT_STORAGE_SIZE)
    set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D CUFFT_STORAGE_SIZE=${CUFFT_STORAGE_SIZE}")
    message("${ADDITIONAL_C_FLAGS}")

    if (DEFINED CUFFT_STORAGE_TYPE)
        set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D CUFFT_STORAGE_TYPE=${CUFFT_STORAGE_TYPE}")
        message("${ADDITIONAL_C_FLAGS}")
    endif()
endif()

#FFT_C2R_STORAGE_SIZE
if (DEFINED FFT_C2R_STORAGE_SIZE)
    set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D FFT_C2R_STORAGE_SIZE=${FFT_C2R_STORAGE_SIZE}")
    message("${ADDITIONAL_C_FLAGS}")

    if (DEFINED FFT_C2R_STORAGE_TYPE)
        set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D FFT_C2R_STORAGE_TYPE=${FFT_C2R_STORAGE_TYPE}")
        message("${ADDITIONAL_C_FLAGS}")
    endif()
endif()

#CONV_CORREC_STORAGE_SIZE
if (DEFINED CONV_CORREC_STORAGE_SIZE)
    set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D CONV_CORREC_STORAGE_SIZE=${CONV_CORREC_STORAGE_SIZE}")
    message("${ADDITIONAL_C_FLAGS}")

    if (DEFINED CONV_CORREC_STORAGE_TYPE)
        set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D CONV_CORREC_STORAGE_TYPE=${CONV_CORREC_STORAGE_TYPE}")
        message("${ADDITIONAL_C_FLAGS}")
    endif()
endif()

#DFT_STORAGE_SIZE
if (DEFINED DFT_STORAGE_SIZE)
    set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D DFT_STORAGE_SIZE=${DFT_STORAGE_SIZE}")
    message("${ADDITIONAL_C_FLAGS}")

    if (DEFINED DFT_STORAGE_TYPE)
        set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -D DFT_STORAGE_TYPE=${DFT_STORAGE_TYPE}")
        message("${ADDITIONAL_C_FLAGS}")
    endif()
endif()



project(SEP_Pipeline C)

# Set to 1 to activate debug compilation (0 for release)
if(NOT ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
        MESSAGE("Generate Debug project")
        #			set(CMAKE_BUILD_TYPE Debug)
        #			set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Debug)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -pg -Wall ${ADDITIONAL_C_FLAGS}")
    else()
        MESSAGE("Generate Release project")
        #			set(CMAKE_BUILD_TYPE Release)
        #			set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release)
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -O3 ${ADDITIONAL_C_FLAGS}")
    endif()

    #add libmath during non visual studio builds
    set(CMAKE_EXTRA_LIB m)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

set(LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/lib/cmake_modules/")


# *******************************************
# ************ Pthread LIBRARY **************
# *******************************************
if (WIN32)
    set(THREADS_USE_PTHREADS_WIN32 true)
    # pthread included AFTER Sdl to avoid unnecessary warnings
    file(GLOB PTHREADDIR "${LIBS_DIR}/pthread-[\\.|0-9]*")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PTHREADDIR})
    Find_Package ( THREADS REQUIRED )
else()
    Find_Package ( Threads )
endif()

if(NOT THREADS_FOUND)
    MESSAGE(FATAL_ERROR "Pthread not found !")
endif()


if (WIN32)
    file(GLOB
            PTHREADS_DLL
            ${CMAKE_PREFIX_PATH}/lib/*.dll
    )

    MESSAGE("Copy Pthreads DLLs into ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    if(NOT ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
        file(COPY ${PTHREADS_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    else()
        file(COPY ${PTHREADS_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug)
        file(COPY ${PTHREADS_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release)
    endif()
endif()

# *******************************************
# ************ OpenMP LIBRARY ***************
# *******************************************
OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
    FIND_PACKAGE(OpenMP)
    IF(OPENMP_FOUND)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ENDIF()
ENDIF()

# *******************************************
# ************ FFTW LIBRARY *****************
# *******************************************

#set (FFTW_INCLUDE_DIRS /usr/lib/x86_64-linux-gnu /usr/include/)
#Find_Package (FFTW)
#if(NOT FFTW_FOUND)
#   MESSAGE(FATAL_ERROR "FFTW not found !")
#endif()

find_library(FFTW_LIB fftw3)
if (NOT FFTW_LIB)
    message(FATAL_ERROR "FFTW library not found.")
endif()

include_directories(/usr/include)
link_libraries(${FFTW_LIB})



# *******************************************
# ************ Lapack LIBRARY **************
# *******************************************

SET(ENV{MKLROOT} /opt/intel/mkl)
#set(BLA_VENDOR Intel10_64lp)
set(BLA_VENDOR Generic)

Find_Package(LAPACK REQUIRED)
if(NOT LAPACK_FOUND)
    MESSAGE(FATAL_ERROR "LAPACK not found !")
endif()


# *******************************************
# ***** SEP_Pipeline CPU Version  ***********
# *******************************************
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/generated ${THREADS_PTHREADS_INCLUDE_DIR} ${FFTW_INCLUDE_DIRS})

file(
        GLOB
        source_files_CPU
        ./src/CPU/*.c
        ./generated/*.c
        ./src/*.c
        main.c
)

file(
        GLOB
        header_files
        ./include/*.h
        ./generated/*.h
)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -g -Wall -mcmodel=large")
#message("source_files_CPU : ${source_files_CPU}")

add_executable( SEP_Pipeline ${source_files_CPU} ${header_files})
target_link_libraries(SEP_Pipeline ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_EXTRA_LIB} fftw3 fftw3f lapack)


# *******************************************
# ***** SEP_Pipeline GPU Version  ***********
# *******************************************

if(CMAKE_CUDA_COMPILER)

    project(SEP_Pipeline_GPU CXX CUDA)

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -g -g -Wall ${ADDITIONAL_C_FLAGS}")

    find_package(CUDA 10.1 REQUIRED)
    message(STATUS ">>> Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")

    set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS};
            -std=c++11 -arch=sm_60 -Xptxas -O3 -restrict -D_FORCE_INLINES -D_DEBUG -lineinfo
    )

    file(
            GLOB
            source_files_GPU
            ./src/*.c
            ./src/GPU/*.cu
            ./generated/*.cpp
            ./generated/*.c
            main.c
    )

    file(
            GLOB
            header_files
            ./include/*.h
            ./generated/*.h
    )

    #    message("source_files_GPU : ${source_files_GPU}")
    #    message("header_files : ${header_files}")
    cuda_add_executable(SEP_Pipeline_GPU ${source_files_GPU})
    target_link_libraries(SEP_Pipeline_GPU m cudart cusolver cufft) # math, CUDA FFT and SVD
endif()


