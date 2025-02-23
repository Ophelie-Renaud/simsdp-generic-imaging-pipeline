#!/bin/bash

# Vérification du nombre d'arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <mode>"
    echo "Modes disponibles : g2g, fft, dft, g2g_clean"
    exit 1
fi

MODE=$1

# Déterminer le dossier et le fichier de sortie en fonction du mode
case "$MODE" in
    g2g) DIR="code_g2g"; OUTPUT_FILE="g2g.csv" ;;
    g2g_clean) DIR="code_g2g_clean"; OUTPUT_FILE="g2g_clean.csv" ;;
    fft) DIR="code_fft"; OUTPUT_FILE="fft.csv" ;;
    dft) DIR="code_dft"; OUTPUT_FILE="dft.csv" ;;
    *) 
        echo "Erreur : mode invalide '$MODE'. Choisissez parmi : g2g, fft, dft, g2g_clean."
        exit 1
        ;;
esac

echo "Mode sélectionné : $MODE"
echo "Dossier : $DIR"
echo "Fichier de sortie : $OUTPUT_FILE"


# Liste des valeurs de paramètres à tester
NANT=512
NUM_BASELINES=$((NANT * (NANT - 1) / 2))
NUM_VIS_values=($((10 * NUM_BASELINES)) $((15 * NUM_BASELINES)) $((20 * NUM_BASELINES)) $((25 * NUM_BASELINES)) $((30 * NUM_BASELINES)))
GRID_SIZE_values=(512 1024 1536 2048 2560)
NUM_MINOR_CYCLE_values=(50 100 150 200 250)


# Aller dans le dossier du code
cd "$DIR" || { echo "Erreur : impossible d'accéder au dossier '$DIR'."; exit 1; }


cmake .
make 

# Vérifier si la compilation a réussi
if [ ! -f SEP_Pipeline ]; then
    echo "Erreur : l'exécutable 'SEP_Pipeline' n'a pas été généré."
    exit 1
fi

# Fichier de log
echo "NUM_VISIBILITIES;GRID_SIZE;NUM_MINOR_CYCLES;DurationII;Latency" > $OUTPUT_FILE  # En-tête du fichier

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
            echo "$NUM_VIS;$GRID_SIZE;$NUM_MINOR_CYCLE;$execution_time;$execution_time" >> $OUTPUT_FILE
        done
    done
done

echo "Exécution terminée. Résultats enregistrés dans $DIR/$OUTPUT_FILE"
