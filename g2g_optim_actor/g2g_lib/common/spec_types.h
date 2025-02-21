
#pragma once

#ifdef SKYTOSKY_SINGLE
    typedef float cell_size_type;
    typedef float cell_scale_type;
    typedef float freq_type;
    typedef float weight_type;
    typedef float uvw_coordinates_type;
    typedef float visibility_type;
    typedef float fft_grid_type;
    typedef float conv_function_type;
    typedef float norm_weight_type;
#endif
#ifdef SKYTOSKY_DOUBLE
    typedef double cell_size_type;
    typedef double cell_scale_type;
    typedef double freq_type;
    typedef double weight_type;
    typedef double uvw_coordinates_type;
    typedef double visibility_type;
    typedef double fft_grid_type;
    typedef double conv_function_type;
    typedef double norm_weight_type;
#endif
