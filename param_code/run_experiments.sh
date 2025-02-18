#!/bin/bash

# Liste des valeurs de paramètres à tester
NUM_VIS_values=(10 11 12 13 14 15 16 17 18 19 20)
GRID_SIZE_values=(10 11 12 13 14 15 16 17 18 19 20)
NUM_MINOR_CYCLE_values=(1 2 3 4 5)

# Vérifier si le dossier build existe, sinon le créer
if [ ! -d "build" ]; then
    mkdir build
fi

# Aller dans le dossier build et générer le projet avec CMake
cd build
cmake ..
make -j$(nproc)  # Compilation avec tous les cœurs disponibles
cd ..

# Vérifier si la compilation a réussi
if [ ! -f build/mon_programme ]; then
    echo "Erreur : l'exécutable 'mon_programme' n'a pas été généré."
    exit 1
fi

# Fichier de log
LOG_FILE="log_execution_time.txt"
echo "NUM_VIS GRID_SIZE NUM_MINOR_CYCLE Execution_Time(ms)" > $LOG_FILE  # En-tête du fichier

# Boucle sur les différents paramètres
for NUM_VIS in "${NUM_VIS_values[@]}"
do
    for GRID_SIZE in "${GRID_SIZE_values[@]}"
    do
        for NUM_MINOR_CYCLE in "${NUM_MINOR_CYCLE_values[@]}"
        do
            echo "Exécution avec NUM_VIS=$NUM_VIS, GRID_SIZE=$GRID_SIZE, NUM_MINOR_CYCLE=$NUM_MINOR_CYCLE"
            
            # Début du chrono
            start_time=$(date +%s%N)
            
            # Exécution de ton programme avec les paramètres
            ./build/sep $NUM_VIS $GRID_SIZE $NUM_MINOR_CYCLE

	    # Fin du chrono
            end_time=$(date +%s%N)
            
            # Calcul du temps en millisecondes
            execution_time=$(echo "scale=3; ($end_time - $start_time) / 1000000" | bc)
            
            # Enregistrer les résultats dans le fichier log
            echo "$NUM_VIS $GRID_SIZE $NUM_MINOR_CYCLE $execution_time" >> $LOG_FILE
        done
    done
        done
    done
done

