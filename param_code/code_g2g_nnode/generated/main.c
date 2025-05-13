//
// Created by orenaud on 5/4/25.
//
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include "sub/sub.h"
int main(int argc, char *argv[]) {
  // Vérification du nombre d'arguments passés
  if (argc != 5) {
    printf("Usage: %s <NUM_VIS> <GRID_SIZE> <NUM_MINOR_CYCLE> <NUM_NODE>\n", argv[0]);
    return 1;
  }
  // Récupérer les paramètres passés
  int NUM_VIS = atoi(argv[1]);
  int GRID_SIZE = atoi(argv[2]);
  int NUM_MINOR_CYCLE = atoi(argv[3]);
    int NUM_NODE = atoi(argv[4]);

    int nb_proc;
    int proc_nb;
    char mpi_hostname[MPI_MAX_PROCESSOR_NAME];
    int resultlen;
    int i;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_nb);

    if (proc_nb == 0)
    {
        printf("Test MPI sur %d processus\n\n", nb_proc);
    }
    if (NUM_NODE>nb_proc) {
        printf("Le nombre de node demandé dépasse le nombre disponible, le nomnbre de processus est seuillé à %d \n\n", nb_proc);
        NUM_NODE = nb_proc;
    }
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Get_processor_name(mpi_hostname, &resultlen);

    for (i=0; i<NUM_NODE; ++i)
    {
        if (proc_nb == i)
        {
            printf("Le processus de rang %3d s'execute sur %s\n ", i, mpi_hostname);
            sub(NUM_VIS, GRID_SIZE, NUM_MINOR_CYCLE);
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();

    return 0;
}