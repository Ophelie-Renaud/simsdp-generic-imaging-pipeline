#ifndef LIBCPU_SKYTOSKY_SINGLE_H
#define LIBCPU_SKYTOSKY_SINGLE_H
#include <stddef.h>


typedef struct interpolation_parameters{
    // Atennas specs
    size_t nrows;
    // Observation specs
    size_t Nx;
    size_t half_Nx;
    size_t Ny;
    size_t half_Ny;
    size_t Nchan;
    size_t spw_selected;
    size_t oversampling_factor;
    size_t half_support_function;
    size_t full_support_function;
    size_t nb_w_planes;
    size_t nb_vis_polarization;
    size_t nb_grid_polarization;
    size_t *grid_channel_idx;
    size_t no_grid_index;
    size_t grid_channel_width;
    size_t no_chan_spw; // Number of nb_channels x nb_spw
    size_t nb_grid_chan;
    size_t polarization_step;
    double cell_size_l;
    double cell_size_m;
    float* u_scale;
    float* v_scale;
    // S2S
    size_t do_s2s;
    int* len_s2s_coo;
    int* a_coo;
    float* u_coo;
    float* v_coo;
    int* ch_coo;
    float* uvach_coo;
    size_t tot_size_coo;
    // Frequencies
    float* chan_wavelength;
    // Grids
    float* input_grid;
    float* output_grid;
    float* psf_grid;
    // Data specs
    float* visibilities;
    float* visibility_weight;
    float* uvw_coordinates;
    // Convolution functions specs
    float* gridding_conv_function;
    size_t filter_size;
    float* filter_AA_2D;
    size_t filter_AA_2D_size;
    size_t filter_choice;
    float max_w;
    float* conv_norm_weight;
    // W-proj Convolution functions specs
} interpolation_parameters;

// Déclarations des fonctions exportées
extern "C"{
    void s2s_single_pola(interpolation_parameters &parameters);
    }
void degridding_quad_pola(void);
void dgg_init_s2s(void);
void free_params(void);
void gridding_psf(void);
void gridding_quad_pola(void);
void init(void);
//void s2s_quad_pola(void);
//void s2s_single_pola(void);
void s2s_single_pola(interpolation_parameters &parameters);
void get_sky2sky_matrix_v0(interpolation_parameters&);
void get_sky2sky_matrix_v1(interpolation_parameters&);
void get_sky2sky_matrix_v3(interpolation_parameters&);



#endif // LIBCPU_SKYTOSKY_SINGLE_H

