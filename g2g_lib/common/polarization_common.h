#pragma once

#include "spec_types.h"
#include "uvw_structure.h"
#include "complex_structure.h"
#include "polarization_vector.h"



class single_polarization{};
class dual_polarization{};
class quad_polarization{};


template<typename which_polarization>
class polarization{
    public:
        typedef pola4<complex_struct<visibility_type>> vis_pola_type;
        typedef pola1<norm_weight_type> weight_pola_type;
        typedef pola4<complex_struct<fft_grid_type>> pixel_pola_type;
        typedef pola1<complex_struct<conv_function_type>> conv_filter_type;
};


template<>
class polarization<single_polarization>{
    public:
        typedef pola1<complex_struct<visibility_type>> vis_pola_type;
        typedef pola1<norm_weight_type> weight_pola_type;
        typedef pola1<complex_struct<fft_grid_type>> pixel_pola_type;
        typedef pola1<complex_struct<conv_function_type>> conv_filter_type;

};


template<>
class polarization<dual_polarization>{
    public:
        typedef pola2<complex_struct<visibility_type>> vis_pola_type;
        typedef pola1<norm_weight_type> weight_pola_type;
        typedef pola2<complex_struct<fft_grid_type>> pixel_pola_type;
        typedef pola1<complex_struct<conv_function_type>> conv_filter_type;

};


template<>
class polarization<quad_polarization>{
    public:
        typedef pola4<complex_struct<visibility_type>> vis_pola_type;
        typedef pola1<norm_weight_type> weight_pola_type;
        typedef pola4<complex_struct<fft_grid_type>> pixel_pola_type;
        typedef pola1<complex_struct<conv_function_type>> conv_filter_type;
};
