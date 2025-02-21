#!/bin/bash

# Liste des valeurs de paramètres à tester
NANT=512
NUM_BASELINES=$((NANT * (NANT - 1) / 2))
NUM_VIS_values=(10)
GRID_SIZE_values=(512)
NUM_MINOR_CYCLE_values=(50)



# Aller dans le dossier du code
cd code_g2g/
#cd code_dft/
#cd code_fft/
cmake .
make 

# Vérifier si la compilation a réussi
if [ ! -f SEP_Pipeline ]; then
    echo "Erreur : l'exécutable 'SEP_Pipeline' n'a pas été généré."
    exit 1
fi

# Fichier de log
LOG_FILE="log_execution_time.txt"
echo "NUM_VISIBILITIES;GRID_SIZE;NUM_MINOR_CYCLE;DurationII;Latency" > $LOG_FILE  # En-tête du fichier

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
            ./SEP_Pipeline $NUM_VIS $GRID_SIZE $NUM_MINOR_CYCLE

	    # Fin du chrono
            end_time=$(date +%s%N)
            
            # Calcul du temps en millisecondes
            execution_time=$(echo "scale=3; ($end_time - $start_time) / 1000000" | bc)
            
            # Enregistrer les résultats dans le fichier log
            echo "$NUM_VIS;$GRID_SIZE;$NUM_MINOR_CYCLE;$execution_time;$execution_time" >> $LOG_FILE
        done
    done
done

