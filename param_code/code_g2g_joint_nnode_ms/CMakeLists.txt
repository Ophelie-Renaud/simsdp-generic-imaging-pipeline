#cmake_minimum_required(VERSION 2.8.12.2)
cmake_minimum_required(VERSION 3.10)

# *******************************************
# ************* CMake Content ***************
# *******************************************
# This CMake create a workspace containing the following projects
#
# Programs
#  - SEP


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



project(G2G_Pipeline C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

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

# -----------------------------------------
# FFTW LIBRARY
# -----------------------------------------
message(STATUS "Building on system: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Host: ${CMAKE_HOST_SYSTEM_NAME}")

execute_process(
        COMMAND hostname
        OUTPUT_VARIABLE HOSTNAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Detected hostname: ${HOSTNAME}")

if(HOSTNAME MATCHES "ruche")
    message(STATUS "Configuration pour Ruche détectée")
    set(FFTW_ROOT "/gpfs/softs/spack/opt/spack/linux-centos7-cascadelake/intel-19.0.3.199/fftw-3.3.8-xxxx")
    set(FFTW_INCLUDE_DIR "${FFTW_ROOT}/include")
    set(FFTW_LIBRARY "${FFTW_ROOT}/lib/libfftw3f.so")
else()
    message(STATUS "Configuration locale détectée")
    find_path(FFTW_INCLUDE_DIR fftw3.h PATHS /usr/include /usr/local/include /usr/lib/x86_64-linux-gnu/include)
    find_library(FFTW_LIBRARY NAMES fftw3f PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
endif()

if(NOT FFTW_INCLUDE_DIR OR NOT FFTW_LIBRARY)
    message(FATAL_ERROR "FFTW not found! Set FFTW_INCLUDE_DIR and FFTW_LIBRARY manually.")
else()
    message(STATUS "FFTW found:")
    message(STATUS "  Includes: ${FFTW_INCLUDE_DIR}")
    message(STATUS "  Library: ${FFTW_LIBRARY}")
endif()

set(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIR})
set(FFTW_LIBRARIES ${FFTW_LIBRARY})


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
# ************ casacore LIBRARY **************
# *******************************************
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")


# Ajouter les répertoires d'inclusion (Casacore, Boost et Python)
include_directories(/usr/local/include)

# Ajouter le répertoire des bibliothèques (modifie ces chemins si nécessaire)
link_directories(/usr/local/lib)

# *******************************************
# ************** MPI LIBRARY ****************
# *******************************************
OPTION(USE_MPI "Enable MPI" ON)
if(USE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_FOUND)
        message(STATUS "MPI found:")
        message(STATUS "  Include dir: ${MPI_C_INCLUDE_PATH}")
        message(STATUS "  Libraries: ${MPI_C_LIBRARIES}")
        include_directories(${MPI_C_INCLUDE_PATH})
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_COMPILE_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_C_LINK_FLAGS}")
    else()
        message(FATAL_ERROR "MPI not found!")
    endif()
endif()



# *******************************************
# ***** G2G_Pipeline CPU Version  ***********
# *******************************************
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/generated ${THREADS_PTHREADS_INCLUDE_DIR} ${FFTW_INCLUDE_DIRS})

file(
        GLOB
        source_files_CPU
        ./src/CPU/*.c
        ./generated/*.c
        ./src/*.c
        ./generated/sub/*.c
)

file(
        GLOB
        header_files
        ./include/*.h
        ./generated/*.h
        ./generated/sub/*.h
)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -g -Wall -mcmodel=large")
add_library(casa_wrapper STATIC
        src/casacore_wrapper.cpp
        include/casacore_wrapper.h
)
target_link_libraries(casa_wrapper casa_casa casa_tables casa_ms casa_measures casa_fits casa_mirlib -lfftw3)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcmodel=large")

add_executable( G2G_Pipeline ${source_files_CPU} ${header_files}
        generated/main.c)
target_link_libraries(G2G_Pipeline ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_EXTRA_LIB} fftw3 fftw3f lapack ${MPI_C_LIBRARIES} m casa_wrapper)


