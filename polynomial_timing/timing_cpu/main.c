#include <stdio.h>
#include <stdlib.h>
#include "include/timings.h"

int main(int argc, char *argv[]) {
    printf("Welcome to the main that encapsulate all computations!\n");
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <function_name> [params...]\n", argv[0]);
        return 1;
    }

    // Identifier la fonction à appeler
    const char *function_name = argv[1];

    if (strcmp(function_name, "time_constant_setups") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s time_constant_setups <NUM_SAMPLES> \n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        time_constant_setups(NUM_SAMPLES);

    } else if (strcmp(function_name, "time_gridsize_setups") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s time_gridsize_setups <NUM_SAMPLES> <GRID_SIZE>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int GRID_SIZE = atoi(argv[3]);
        time_gridsize_setups(NUM_SAMPLES, GRID_SIZE);
    }else if (strcmp(function_name, "time_visibility_setups") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s time_visibility_setups <NUM_SAMPLES> <NUM_VISIBILITIES>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int NUM_VISIBILITIES = atoi(argv[3]);
        time_visibility_setups(NUM_SAMPLES, NUM_VISIBILITIES);
    } else if (strcmp(function_name, "time_save_output") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s time_save_output <NUM_SAMPLES> <GRID_SIZE>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int GRID_SIZE = atoi(argv[3]);
        time_save_output(NUM_SAMPLES, GRID_SIZE);
    }else if (strcmp(function_name, "time_dft") == 0) {
        if (argc != 6) {
            fprintf(stderr, "Usage: %s time_dft <NUM_MINOR_CYCLES> <NUM_VISIBILITIES> <NUM_ACTUAL_VISIBILITIES>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int NUM_MINOR_CYCLES = atoi(argv[3]);
        int NUM_VISIBILITIES = atoi(argv[4]);
        int NUM_ACTUAL_VISIBILITIES = atoi(argv[5]);
        time_dft(NUM_SAMPLES, NUM_MINOR_CYCLES,NUM_VISIBILITIES,NUM_ACTUAL_VISIBILITIES);
    }else if (strcmp(function_name, "time_gains_application") == 0) {
        if (argc != 5) {
            fprintf(stderr, "Usage: %s time_gains_application <NUM_SAMPLES> <NUM_VISIBILITIES><NUM_ACTUAL_VISIBILITIES>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int NUM_VISIBILITIES = atoi(argv[3]);
        int NUM_ACTUAL_VISIBILITIES = atoi(argv[4]);
        time_gains_application(NUM_SAMPLES, NUM_VISIBILITIES,NUM_ACTUAL_VISIBILITIES);
    }else if (strcmp(function_name, "time_substraction") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s time_substraction <NUM_SAMPLES> <NUM_VISIBILITIES>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int NUM_VISIBILITIES = atoi(argv[3]);
        time_substraction(NUM_SAMPLES,NUM_VISIBILITIES);
    }else if (strcmp(function_name, "time_add_visibilities") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s time_add_visibilities <NUM_SAMPLES> <NUM_VISIBILITIES>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int NUM_VISIBILITIES = atoi(argv[3]);
        time_add_visibilities(NUM_SAMPLES, NUM_VISIBILITIES);
    }else if (strcmp(function_name, "time_prolate") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s time_prolate <NUM_SAMPLES> <GRID_SIZE>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int GRID_SIZE = atoi(argv[3]);
        time_prolate(NUM_SAMPLES, GRID_SIZE);
    }else if (strcmp(function_name, "time_finegrid") == 0) {
        if (argc != 5) {
            fprintf(stderr, "Usage: %s time_finegrid <NUM_SAMPLES> <NUM_VISIBILITIES> <NUM_ACTUAL_VISIBILITIES>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int NUM_VISIBILITIES = atoi(argv[3]);
        int NUM_ACTUAL_VISIBILITIES = atoi(argv[4]);
        time_finegrid(NUM_SAMPLES, NUM_VISIBILITIES, NUM_ACTUAL_VISIBILITIES);
    }else if (strcmp(function_name, "time_subtract_ispace") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s time_subtract_ispace <NUM_SAMPLES> <GRID_SIZE> \n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int GRID_SIZE = atoi(argv[3]);
        time_subtract_ispace(NUM_SAMPLES, GRID_SIZE);
    }else if (strcmp(function_name, "time_fft_shift") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s time_fft_shift <NUM_SAMPLES> <GRID_SIZE> \n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int GRID_SIZE = atoi(argv[3]);
        time_fft_shift(NUM_SAMPLES, GRID_SIZE);
    }else if (strcmp(function_name, "time_fft") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s time_fft <NUM_SAMPLES> <GRID_SIZE> \n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int GRID_SIZE = atoi(argv[3]);
        time_fft(NUM_SAMPLES, GRID_SIZE);
    }else if (strcmp(function_name, "time_hogbom") == 0) {
        if (argc != 5) {
            fprintf(stderr, "Usage: %s time_hogbom <NUM_SAMPLES> <GRID_SIZE> <NUM_MINOR_CYCLES>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int GRID_SIZE = atoi(argv[3]);
        int NUM_MINOR_CYCLES = atoi(argv[4]);
        time_hogbom(NUM_SAMPLES, GRID_SIZE, NUM_MINOR_CYCLES);
    }else if (strcmp(function_name, "time_std_gridding") == 0) {
        if (argc != 5) {
            fprintf(stderr, "Usage: %s time_std_gridding <NUM_SAMPLES> <GRID_SIZE> <NUM_VISIBILITIES>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int GRID_SIZE = atoi(argv[3]);
        int NUM_VISIBILITIES = atoi(argv[4]);
        time_std_gridding(NUM_SAMPLES, GRID_SIZE, NUM_VISIBILITIES);
    }else if (strcmp(function_name, "time_dft_gridding") == 0) {
        if (argc != 5) {
            fprintf(stderr, "Usage: %s time_dft_gridding <NUM_SAMPLES> <GRID_SIZE> <NUM_VISIBILITIES>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int GRID_SIZE = atoi(argv[3]);
        int NUM_VISIBILITIES = atoi(argv[4]);
        time_dft_gridding(NUM_SAMPLES, GRID_SIZE, NUM_VISIBILITIES);
    }else if (strcmp(function_name, "time_degrid") == 0) {
        if (argc != 5) {
            fprintf(stderr, "Usage: %s time_degrid <NUM_SAMPLES> <GRID_SIZE> <NUM_VISIBILITIES>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int GRID_SIZE = atoi(argv[3]);
        int NUM_VISIBILITIES = atoi(argv[4]);
        time_degrid(NUM_SAMPLES, GRID_SIZE, NUM_VISIBILITIES);
    }else if (strcmp(function_name, "time_s2s") == 0) {
        if (argc != 5) {
            fprintf(stderr, "Usage: %s time_s2s <NUM_SAMPLES> <GRID_SIZE> <NUM_VISIBILITIES>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int GRID_SIZE = atoi(argv[3]);
        int NUM_VISIBILITIES = atoi(argv[4]);
        time_s2s_degrid(NUM_SAMPLES, GRID_SIZE, NUM_VISIBILITIES);
    }else if (strcmp(function_name, "time_psf_host_set_up") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s time_psf_host_set_up <NUM_SAMPLES> <GRID_SIZE>\n", argv[0]);
            return 1;
        }
        int NUM_SAMPLES = atoi(argv[2]);
        int GRID_SIZE = atoi(argv[3]);
        time_psf_host_set_up(NUM_SAMPLES, GRID_SIZE);
    }else{
        fprintf(stderr, "Error: Unknown function '%s'\n", function_name);
        return 1;
    }

    return 0;
}

