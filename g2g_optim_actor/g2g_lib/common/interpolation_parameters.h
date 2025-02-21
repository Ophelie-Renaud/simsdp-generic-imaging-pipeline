#pragma once

#include <complex>
#include "spec_types.h"

struct interpolation_parameters{
    //Antennas specs
    size_t nrows;

    //Observation specs
    size_t Nx;
    size_t half_Nx;
    size_t Ny;
    size_t half_Ny;
    size_t Nchan;
    size_t spw_selected;
    size_t oversampling_factor;
    size_t half_support_function;
    size_t full_support_function;
    //W-proj Convolution functions specs
    size_t nb_w_planes;
    size_t nb_vis_polarization;
    size_t nb_grid_polarization;
    int* grid_channel_idx;
    size_t no_grid_index;
    size_t grid_channel_width;
    size_t no_chan_spw;
    size_t nb_grid_chan;
    size_t polarization_step;
    
    double cell_size_l;
    double cell_size_m;
    cell_scale_type u_scale;
    cell_scale_type v_scale;
    //cell_scale_type

    // s2s
    size_t do_s2s;
    int* len_s2s_coo;
    int* a_coo;
    float* u_coo;
    float* v_coo;
    int* ch_coo;
    float* uvach_coo;
    size_t tot_size_coo;

    //Frequencies
    freq_type *chan_wavelength;
    //double *chan_wavelength;

    //Grids
    //std::complex<fft_grid_type> *input_grid;
    //std::complex<fft_grid_type> *output_grid;
    fft_grid_type* __restrict__ input_grid;
    fft_grid_type* __restrict__ output_grid;
    fft_grid_type* __restrict__ psf_grid;


    //Data specs
    //bool *flags;
    //unsigned int *data_desc_id;
    std::complex<visibility_type> * __restrict__ visibilities;
    weight_type * __restrict__ visibility_weight;
    //double *visibility_weight;
    uvw_coordinates_type * __restrict__ uvw_coordinates;
    //double *uvw_coordinates;

    //Convolution functions specs
    conv_function_type *gridding_conv_function;
    size_t filter_size;
    conv_function_type *filter_AA_2D; // 2D AA convolution kernel
    size_t filter_AA_2D_size;
    size_t filter_choice;
    float max_w;
    norm_weight_type *conv_norm_weight;


    //S2S specs
    

};


