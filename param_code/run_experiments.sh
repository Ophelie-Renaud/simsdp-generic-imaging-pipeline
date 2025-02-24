#!/bin/bash

# Vérification du nombre d'arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <mode>"
    echo "Modes disponibles : g2g, fft, dft, g2g_clean, all"
    exit 1
fi

MODE=$1

MODES=(
    [g2g]="code_g2g g2g.csv"
    [g2g_clean]="code_g2g_clean g2g_clean.csv"
    [fft]="code_fft fft.csv"
    [dft]="code_dft dft.csv"
)

run_mode() {
    DIR=$1
    OUTPUT_FILE=$2
    
    echo "Mode sélectionné : $MODE"
    echo "Dossier : $DIR"
    echo "Fichier de sortie : $OUTPUT_FILE"
    
    cd "$DIR" || { echo "Erreur : impossible d'accéder au dossier '$DIR'."; exit 1; }
    # Suppression de l'output file s'il existe
    [ -f "$OUTPUT_FILE" ] && rm "$OUTPUT_FILE"
    
    # Suppression de CMakeCache.txt s'il existe
    [ -f "CMakeCache.txt" ] && rm "CMakeCache.txt"
    
    cmake .
    make
    
    if [ ! -f SEP_Pipeline ]; then
        echo "Erreur : l'exécutable 'SEP_Pipeline' n'a pas été généré."
        exit 1
    fi
    
    echo "NUM_VISIBILITIES;GRID_SIZE;NUM_MINOR_CYCLES;DurationII;Latency" > $OUTPUT_FILE
    
    NANT=512
    NUM_BASELINES=$((NANT * (NANT - 1) / 2))
    NUM_VIS_values=($((10 * NUM_BASELINES)) $((15 * NUM_BASELINES)) $((20 * NUM_BASELINES)))
    GRID_SIZE_values=(512 1024 1536)
    NUM_MINOR_CYCLE_values=(50 100 150)
    
    for NUM_VIS in "${NUM_VIS_values[@]}"
    do
        for NUM_MINOR_CYCLE in "${NUM_MINOR_CYCLE_values[@]}"
        do
            for GRID_SIZE in "${GRID_SIZE_values[@]}"
            do
                echo "Exécution avec NUM_VIS=$NUM_VIS, GRID_SIZE=$GRID_SIZE, NUM_MINOR_CYCLE=$NUM_MINOR_CYCLE"
                start_time=$(date +%s%N)
                ./SEP_Pipeline $NUM_VIS $GRID_SIZE $NUM_MINOR_CYCLE
                end_time=$(date +%s%N)
                execution_time=$(echo "scale=3; ($end_time - $start_time) / 1000000" | bc)
                echo "$NUM_VIS;$GRID_SIZE;$NUM_MINOR_CYCLE;$execution_time;$execution_time" >> $OUTPUT_FILE
            done
        done
    done
    
    cd ..
}

if [ "$MODE" == "all" ]; then
    for key in "${!MODES[@]}"; do
        run_mode ${MODES[$key]}
    done
else
    if [[ -n "${MODES[$MODE]}" ]]; then
        run_mode ${MODES[$MODE]}
    else
        echo "Erreur : mode invalide '$MODE'. Choisissez parmi : g2g, fft, dft, g2g_clean, all."
        exit 1
    fi
fi

echo "Exécution terminée."
