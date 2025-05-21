#!/bin/bash

# pb run all fait tourner que dft je sais pas pourquoi

# Vérification du nombre d'arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <mode>"
    echo "Modes disponibles : g2g, fft, dft, g2g_clean, all"
    exit 1
fi

MODE=$1

declare -A MODES  # Déclaration du tableau associatif
MODES=(
    [g2g]="code_g2g_nnode_ms g2g.csv G2G_Pipeline"
    [fft]="code_fft_nnode_ms fft.csv FFT_Pipeline"
    [dft]="code_dft_nnode_ms dft.csv DFT_Pipeline"
)

run_mode() {
    DIR=$1
    OUTPUT_FILE=$2
    EXECUTABLE=$3
    

    
    echo "NUM_VISIBILITIES;GRID_SIZE;NUM_MINOR_CYCLES;DurationII;Latency" > "$DIR/$OUTPUT_FILE"
    
    NANT=512
    NUM_BASELINES=$((NANT * (NANT - 1) / 2))
    NUM_VIS_values=(10⁴ 10⁵ 10⁶) #SB155.rebin = 1477782
    GRID_SIZE_values=(512 1024 1536 2048 4096) #DFT crash at 4096
    NUM_MINOR_CYCLE_values=(1 10 100 1000)
    MEASUREMENTSET="/home/orenaud/Desktop/nancep/VirA-SB155/SB155.rebin.MS"
    OUTPUT="/home/orenaud/Desktop/output/"
    NODE=1
    
    for NUM_VIS in "${NUM_VIS_values[@]}"
    do
        for NUM_MINOR_CYCLE in "${NUM_MINOR_CYCLE_values[@]}"
        do
            for GRID_SIZE in "${GRID_SIZE_values[@]}"
            do
                echo "Exécution avec NUM_VIS=$NUM_VIS, GRID_SIZE=$GRID_SIZE, NUM_MINOR_CYCLE=$NUM_MINOR_CYCLE"
                start_time=$(date +%s%N)
                "$DIR"/"$EXECUTABLE" $NUM_VIS $GRID_SIZE $NUM_MINOR_CYCLE $NODE "$MEASUREMENTSET" "$OUTPUT"
                end_time=$(date +%s%N)
                execution_time=$(echo "scale=3; ($end_time - $start_time) / 1000000" | bc)
                echo "$NUM_VIS;$GRID_SIZE;$NUM_MINOR_CYCLE;$execution_time;$execution_time" >> $DIR/$OUTPUT_FILE
            done
        done
    done
    echo "Exécution terminée de $MODE. Résultats enregistrés dans $DIR/$OUTPUT_FILE"
    #cd ..
}

if [ "$MODE" == "all" ]; then
     for key in "${!MODES[@]}"; do
        eval set -- ${MODES[$key]}
        run_mode "$1" "$2" "$3"
    done
else
    if [[ -n "${MODES[$MODE]}" ]]; then
        eval set -- ${MODES[$MODE]}
        run_mode "$1" "$2" "$3"
    else
        echo "Erreur : mode invalide '$MODE'. Choisissez parmi : g2g, fft, dft, all."
        exit 1
    fi
fi

echo "Exécution terminée."
