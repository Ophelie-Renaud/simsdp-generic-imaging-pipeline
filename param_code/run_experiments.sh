#!/bin/bash

# Liste des valeurs de paramètres à tester
NUM_VIS_values=(10 11 12 13 14 15 16 17 18 19 20)
GRID_SIZE_values=(10 11 12 13 14 15 16 17 18 19 20)
NUM_MINOR_CYCLE_values=(1 2 3 4 5)

# Boucle sur les différents paramètres
for NUM_VIS in "${NUM_VIS_values[@]}"
do
    for GRID_SIZE in "${GRID_SIZE_values[@]}"
    do
        for NUM_MINOR_CYCLE in "${NUM_MINOR_CYCLE_values[@]}"
        do
            echo "Exécution avec NUM_VIS=$NUM_VIS, GRID_SIZE=$GRID_SIZE, NUM_MINOR_CYCLE=$NUM_MINOR_CYCLE"
            
            # Compiler ton programme (si nécessaire)
            gcc -o mon_programme communication.c Core0.c dump.c fifo.c mac_barrier.c preesm_md5.c main.c -lm

            # Exécution de ton programme avec les paramètres
            ./mon_programme $NUM_VIS $GRID_SIZE $NUM_MINOR_CYCLE
            
            # Enregistrer les résultats dans un fichier log
            echo "Exécution avec NUM_VIS=$NUM_VIS, GRID_SIZE=$GRID_SIZE, NUM_MINOR_CYCLE=$NUM_MINOR_CYCLE terminée à $(date)" >> log_memoryScripts.txt
        done
    done
done

