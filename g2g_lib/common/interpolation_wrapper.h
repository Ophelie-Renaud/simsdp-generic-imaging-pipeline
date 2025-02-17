#pragma once

extern "C"{

    void init(interpolation_parameters &parameters);
    void free_params(interpolation_parameters &parameters);
    void gridding_psf(interpolation_parameters &parameters);
    void gridding_single_pola(interpolation_parameters &parameters);
    void gridding_dual_pola(interpolation_parameters &parameters);
    void gridding_quad_pola(interpolation_parameters &parameters);
    void degridding_single_pola(interpolation_parameters &parameters);
    void degridding_dual_pola(interpolation_parameters &parameters);
    void degridding_quad_pola(interpolation_parameters &parameters);
    void dgg_init_s2s(interpolation_parameters &parameters);
    void s2s_single_pola(interpolation_parameters &parameters);
    void s2s_quad_pola(interpolation_parameters &parameters);
}
