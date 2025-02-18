#ifndef LIBCPU_SKYTOSKY_SINGLE_H
#define LIBCPU_SKYTOSKY_SINGLE_H

#ifdef __cplusplus
extern "C" {
#endif

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
    void* grid_channel_idx;
    size_t no_grid_index;
    size_t grid_channel_width;
    size_t no_chan_spw; // Number of nb_channels x nb_spw
    size_t nb_grid_chan;
    size_t polarization_step;
    double cell_size_l;
    double cell_size_m;
    void* u_scale;
    void* v_scale;
    // S2S
    size_t do_s2s;
    void* len_s2s_coo;
    void* a_coo;
    void* u_coo;
    void* v_coo;
    void* ch_coo;
    void* uvach_coo;
    size_t tot_size_coo;
    // Frequencies
    void* chan_wavelength;
    // Grids
    void* input_grid;
    void* output_grid;
    void* psf_grid;
    // Data specs
    void* visibilities;
    void* visibility_weight;
    void* uvw_coordinates;
    // Convolution functions specs
    void* gridding_conv_function;
    size_t filter_size;
    void* filter_AA_2D;
    size_t filter_AA_2D_size;
    size_t filter_choice;
    float max_w;
    void* conv_norm_weight;
    // W-proj Convolution functions specs
} interpolation_parameters;

// Déclarations des fonctions exportées
void degridding_quad_pola(void);
void dgg_init_s2s(void);
void free_params(void);
void gridding_psf(void);
void gridding_quad_pola(void);
void init(void);
void s2s_quad_pola(void);
void s2s_single_pola(void);
void get_sky2sky_matrix_v0( interpolation_parameters* params);
void get_sky2sky_matrix_v1( interpolation_parameters* params);
void get_sky2sky_matrix_v3( interpolation_parameters* params);

#ifdef __cplusplus
}
#endif

#endif // LIBCPU_SKYTOSKY_SINGLE_H

