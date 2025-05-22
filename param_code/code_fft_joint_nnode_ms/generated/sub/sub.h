//
// Created by orenaud on 5/4/25.
//

#ifndef SUB_H
#define SUB_H

#include "preesm.h"

void sub(int num_vis, int grid_size, int num_minor_cycle, const char* ms_path, const char* out_path) ;
void set_up(int NUM_VIS, int GRID_SIZE,int NUM_MINOR_CYCLE,const char* MS_PATH,const char* OUT_PATH,
    Config config,PRECISION *clean_psf,int2 *psf_halfdims, int2 *gridding_kernel_supports,PRECISION2 *gridding_kernels,PRECISION *psf,int2 *receiver_pairs,PRECISION3 *vis_uvw_coords,int2 *degridding_kernel_supports,PRECISION2 *degridding_kernels,PRECISION2 *gains,PRECISION2 *measured_visibilities,PRECISION *prolate);
void delta(int NUM_VIS, int GRID_SIZE,int NUM_MINOR_CYCLE,const char* MS_PATH,const char* OUT_PATH, Config config,
PRECISION *image_estimate,PRECISION3 *source_list,int *num_source,
    int2 *gridding_kernel_supports,PRECISION2 *gridding_kernels,PRECISION *psf,int2 *receiver_pairs,PRECISION3 *vis_uvw_coords,int2 *degridding_kernel_supports,PRECISION2 *degridding_kernels,PRECISION2 *gains,PRECISION2 *measured_visibilities,PRECISION *prolate, int iter_major,
    PRECISION *delta_image,PRECISION *image_out);
void psi(int NUM_VIS,int GRID_SIZE, int NUM_MINOR_CYCLE, const char* MS_PATH,const char* OUT_PATH,Config config,PRECISION *clean_psf,PRECISION *delta_image,PRECISION *image_model,int2 *psf_halfdims,int iter_major,PRECISION *image_estimate,PRECISION3 *source_list,int *num_source);

#endif //SUB_H
