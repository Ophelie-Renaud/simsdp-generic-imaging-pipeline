# Définir la version de CMake
cmake_minimum_required(VERSION 3.10)

# Nom du projet
project(ReadMS)

# Spécifier la norme C++ utilisée
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")


# Ajouter les répertoires d'inclusion (Casacore, Boost et Python)
include_directories(/usr/local/include)

# Ajouter le répertoire des bibliothèques (modifie ces chemins si nécessaire)
link_directories(/usr/local/lib)

# Ajouter le fichier source
set(SOURCE_FILES main.c read_ms.cpp)

# Créer l'exécutable
add_executable(read_ms ${SOURCE_FILES})

# Lier les bibliothèques nécessaires (Casacore et Boost)
target_link_libraries(read_ms
        casa_casa
        casa_tables
        boost_system
        boost_filesystem
)

# Ajouter un chemin pour les bibliothèques partagées (si nécessaire)
# Ceci est utile si les bibliothèques Casacore et Boost sont dans un répertoire personnalisé
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
