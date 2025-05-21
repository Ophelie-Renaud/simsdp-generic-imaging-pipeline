//
// Created by orenaud on 5/4/25.
//
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <stddef.h>

#include "deconvolution_run.h"
#include "sub/sub.h"
int main(int argc, char *argv[]) {
  // Vérification du nombre d'arguments passés
  if (argc != 7) {
    printf("Usage: %s <NUM_VIS> <GRID_SIZE> <NUM_MINOR_CYCLE> <NUM_NODE> <MS_PATH> <OUT_PATH>\n", argv[0]);
    return 1;
  }
  // Récupérer les paramètres passés
  int NUM_VIS = atoi(argv[1]);
  int GRID_SIZE = atoi(argv[2]);
  int NUM_MINOR_CYCLE = atoi(argv[3]);
    int NUM_NODE = atoi(argv[4]);
  const char* MS_PATH = argv[5];
    const char* OUT_PATH = argv[6];

    int nb_proc;
    int proc_nb;
    char mpi_hostname[MPI_MAX_PROCESSOR_NAME];
    int resultlen;

    int num_gridding_kernel = 17;
    int total_gridding_kernel_samples = 108800;
    int num_baseline = 130816;
    int num_receivers = 512;
    int max_sources = NUM_MINOR_CYCLE;

    //max values
   int grid_size = 512;//32768;


    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &nb_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_nb);
    MPI_Get_processor_name(mpi_hostname, &resultlen);

    if (proc_nb == 0)
    {
        printf("Test MPI sur %d processus\n\n", nb_proc);
    }
    if (NUM_NODE>nb_proc) {
        printf("Le nombre de node demandé dépasse le nombre disponible, le nomnbre de processus est seuillé à %d \n\n", nb_proc);
        NUM_NODE = nb_proc;
    }

    // Définition MPI_INT2
    MPI_Datatype MPI_INT2;
    {
      int blocklengths[2] = {1, 1};
      MPI_Aint offsets[2] = {offsetof(int2, x), offsetof(int2, y)};
      MPI_Datatype types[2] = {MPI_INT, MPI_INT};
      MPI_Type_create_struct(2, blocklengths, offsets, types, &MPI_INT2);
      MPI_Type_commit(&MPI_INT2);
    }

    // Définition MPI_PRECISION3
    MPI_Datatype MPI_PRECISION3;
    {
      int block_lengths[3] = {1, 1, 1};
      MPI_Aint displacements[3] = {
          offsetof(PRECISION3, x),
          offsetof(PRECISION3, y),
          offsetof(PRECISION3, z)
      };
      MPI_Datatype types[3] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT}; // ou MPI_DOUBLE si PRECISION = double
      MPI_Type_create_struct(3, block_lengths, displacements, types, &MPI_PRECISION3);
      MPI_Type_commit(&MPI_PRECISION3);
    }

    //Allocation des buffers locaux
    Config config = {};
    PRECISION* clean_psf = (PRECISION*)malloc(sizeof(PRECISION) * grid_size * grid_size);
    PRECISION* psf = (PRECISION*)malloc(sizeof(PRECISION) * grid_size * grid_size);
    int2* psf_halfdims = (int2*)malloc(sizeof(int2) * 1);
    int2* gridding_kernel_supports = (int2*)malloc(sizeof(int2) * num_gridding_kernel);
    PRECISION2* gridding_kernels = (PRECISION2*)malloc(sizeof(PRECISION2) * total_gridding_kernel_samples);
    int2* degridding_kernel_supports = (int2*)malloc(sizeof(int2) * num_gridding_kernel);
    PRECISION2* degridding_kernels = (PRECISION2*)malloc(sizeof(PRECISION2) * total_gridding_kernel_samples);
    int2* receiver_pairs = (int2*)malloc(sizeof(int2) * num_baseline);
    PRECISION3* vis_uvw_coords = (PRECISION3*)malloc(sizeof(PRECISION3) * NUM_VIS);
    PRECISION2* gains = (PRECISION2*)malloc(sizeof(PRECISION2) * num_receivers);
    PRECISION2* measured_visibilities = (PRECISION2*)malloc(sizeof(PRECISION2) * NUM_VIS);
    PRECISION* prolate = (PRECISION*)malloc(sizeof(PRECISION) * grid_size / 2);
    PRECISION* image_estimate = (PRECISION*)malloc(sizeof(PRECISION) * grid_size * grid_size);
    PRECISION3* source_list = (PRECISION3*)malloc(sizeof(PRECISION3) * max_sources);
    int* num_source = (int*)malloc(sizeof(int) * 1);
    //int* iter_major = (int*)malloc(sizeof(int) * 1);
    PRECISION* delta_image = (PRECISION*)malloc(sizeof(PRECISION) * grid_size * grid_size);
    PRECISION* image_model = (PRECISION*)malloc(sizeof(PRECISION) * grid_size * grid_size);

    // Allocation des buffer globaux
    //Config config_global = {};
    PRECISION* clean_psf_global = (PRECISION*)malloc(sizeof(PRECISION) * grid_size * grid_size*NUM_NODE);
    PRECISION* delta_image_global = (PRECISION*)malloc(sizeof(PRECISION) * grid_size * grid_size*NUM_NODE);
    PRECISION* image_model_global = (PRECISION*)malloc(sizeof(PRECISION) * grid_size * grid_size*NUM_NODE);
    int2* psf_halfdims_global = (int2*)malloc(sizeof(int2) * 1*NUM_NODE);
    PRECISION* image_estimate_global = (PRECISION*)malloc(sizeof(PRECISION) * grid_size * grid_size*NUM_NODE);
    PRECISION3* source_list_global = (PRECISION3*)malloc(sizeof(PRECISION3) * max_sources*NUM_NODE);
    int* num_source_global = (int*)malloc(sizeof(int) * 1*NUM_NODE);

    //fflush(stdout);
    //MPI_Barrier(MPI_COMM_WORLD);




    if (proc_nb <NUM_NODE) {
        printf("Le processus de rang %3d s'exécute sur le node %s, sur le ms %s\n", proc_nb, mpi_hostname, MS_PATH);
        set_up(NUM_VIS, GRID_SIZE, NUM_MINOR_CYCLE, MS_PATH,OUT_PATH, config,clean_psf,psf_halfdims, gridding_kernel_supports, gridding_kernels,psf, receiver_pairs, vis_uvw_coords,degridding_kernel_supports, degridding_kernels,gains,measured_visibilities,prolate);

        for (int iter_major=0;iter_major<5;iter_major++){
            printf("\n ** >> Lancement du cycle majeur %d << **\n\n", iter_major);

            delta(NUM_VIS, GRID_SIZE, NUM_MINOR_CYCLE, MS_PATH,OUT_PATH, config,image_estimate,source_list,num_source, gridding_kernel_supports, gridding_kernels,psf, receiver_pairs, vis_uvw_coords,degridding_kernel_supports, degridding_kernels,gains,measured_visibilities,prolate,iter_major,delta_image,image_model);

           // MPI_Gather(config, 1, MPI_FLOAT,config_global, 1, MPI_FLOAT,0, MPI_COMM_WORLD);
            MPI_Gather(clean_psf, 1, MPI_FLOAT,clean_psf_global, GRID_SIZE * GRID_SIZE, MPI_FLOAT,0, MPI_COMM_WORLD);
            MPI_Gather(delta_image, 1, MPI_FLOAT, delta_image_global, GRID_SIZE * GRID_SIZE, MPI_FLOAT,0, MPI_COMM_WORLD);
            MPI_Gather(image_model, 1, MPI_FLOAT,image_model_global, GRID_SIZE * GRID_SIZE, MPI_FLOAT,0, MPI_COMM_WORLD);
            MPI_Gather(psf_halfdims, 1, MPI_INT2,psf_halfdims_global, 1, MPI_INT2,0, MPI_COMM_WORLD);
            //MPI_Gather(iter_major, 1, MPI_FLOAT,
              // iter_major_global, 1, MPI_FLOAT,
               //0, MPI_COMM_WORLD);

            if (proc_nb == 0) {
                printf("\n ** >> Lancement des cycle mineur de l'iteration %d << **\n\n", iter_major);
                hogbom_clean(grid_size, NUM_MINOR_CYCLE, NUM_MINOR_CYCLE, delta_image_global,
        clean_psf_global, psf_halfdims_global, &config, image_model_global,
        num_source_global, source_list_global, image_estimate_global); // psi_0_hogbom_clean_0
                //psi(NUM_VIS, GRID_SIZE, NUM_MINOR_CYCLE, MS_PATH,OUT_PATH,config,clean_psf_global,delta_image_global,image_model_global,psf_halfdims_global,iter_major,image_estimate_global,source_list_global,num_source_global);
            }
            MPI_Scatter(image_estimate_global, GRID_SIZE * GRID_SIZE, MPI_FLOAT,image_estimate, GRID_SIZE * GRID_SIZE, MPI_FLOAT,0, MPI_COMM_WORLD);
            MPI_Scatter(source_list_global, max_sources, MPI_FLOAT,source_list, max_sources, MPI_FLOAT,0, MPI_COMM_WORLD);
            MPI_Scatter(num_source_global, 1, MPI_INT,num_source, 1, MPI_INT,0, MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    // Libération mémoire (important!)
    free(clean_psf);
    free(psf);
    free(psf_halfdims);
    free(gridding_kernel_supports);
    free(gridding_kernels);
    free(degridding_kernel_supports);
    free(degridding_kernels);
    free(receiver_pairs);
    free(vis_uvw_coords);
    free(gains);
    free(measured_visibilities);
    free(prolate);
    free(image_estimate);
    free(source_list);
    free(num_source);
    free(delta_image);
    free(image_model);

    free(clean_psf_global);
    free(delta_image_global);
    free(image_model_global);
    free(psf_halfdims_global);
    free(image_estimate_global);
    free(source_list_global);
    free(num_source_global);

    MPI_Type_free(&MPI_INT2);
    MPI_Type_free(&MPI_PRECISION3);

    MPI_Finalize();

    return 0;
}