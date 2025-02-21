cmake_minimum_required(VERSION 3.30)
project(untitled3)

set(CMAKE_CXX_STANDARD 20)

# Inclure le répertoire racine du projet pour le fichier .h
include_directories(${CMAKE_SOURCE_DIR})

# Lier la bibliothèque partagée .so depuis la racine du projet
link_directories(${CMAKE_SOURCE_DIR})

add_executable(untitled3 main.cpp)

# Lier la bibliothèque avec ton exécutable
target_link_libraries(untitled3 cpu_skytosky_single)
