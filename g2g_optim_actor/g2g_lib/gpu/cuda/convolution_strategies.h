#pragma once

#include "interpolation_parameters.h"
#include "../../common/spec_types.h"
#include "../../common/uvw_structure.h"
#include "../../common/complex_structure.h"
#include "../../common/polarization_vector.h"


class conv_AA_2D{};

template<typename polarization_strategy, typename T>
class convolution_gridding_strat{
    public:
        __device__ static void conv_grid(
                    const interpolation_parameters &parameters,
                    size_t thread_convU, 
                    size_t thread_convV,
                    typename polarization_strategy::pola::vis_pola_type visi,
                    float& conv_norm_weight,
                    uvw_coordinates_type u_ch,
                    uvw_coordinates_type v_ch,
                    size_t &thread_gridU, 
                    size_t &thread_gridV,
                    conv_function_type &weight
                );

        __device__ static void conv_s2s(
                    const interpolation_parameters &parameters,
                    typename polarization_strategy::pola::vis_pola_type visi,
                    float& conv_norm_weight,
                    size_t ut,
                    size_t vt,
                    float u_tap, 
                    float v_tap, 
                    float u_frac, 
                    float v_frac,
                    int u_int,
                    int v_int,
                    size_t &thread_gridU, 
                    size_t &thread_gridV,
                    conv_function_type &weight,
                    const conv_function_type *filter
                );

};


template<typename polarization_strategy>
class convolution_gridding_strat<polarization_strategy, conv_AA_2D>{
    public:
        __device__ static void conv_grid(
                    const interpolation_parameters &parameters,
                    size_t thread_convU,
                    size_t thread_convV,
                    typename polarization_strategy::pola::vis_pola_type visi,
                    float& conv_norm_weight,
                    uvw_coordinates_type u_ch,
                    uvw_coordinates_type v_ch,
                    size_t &thread_gridU, 
                    size_t &thread_gridV,
                    conv_function_type &weight
                ){

            int u_int, v_int;
            uvw_coordinates_type u_frac, v_frac;
            size_t v_tap = 0, u_tap = 0;

            u_int = round(u_ch);
            v_int = round(v_ch);
            u_frac = u_int - u_ch;
            v_frac = v_int - v_ch;

            u_tap = (((u_int+parameters.half_support_function+parameters.half_Nx)-threadIdx.x)%parameters.full_support_function) + 1;
            v_tap = (v_int+parameters.half_support_function+parameters.half_Nx-  (size_t)floor((float)(threadIdx.x/parameters.full_support_function)))%parameters.full_support_function + 1;

            thread_convU = (size_t)round((u_tap + u_frac + 1)*parameters.oversampling_factor);
            thread_convV = (size_t)round((v_tap + v_frac + 1)*parameters.oversampling_factor);

            weight = parameters.filter_AA_2D[parameters.filter_AA_2D_size*thread_convV + thread_convU];

            thread_gridU = u_int + u_tap + parameters.half_Nx - parameters.half_support_function;
            thread_gridV = v_int + v_tap + parameters.half_Ny - parameters.half_support_function;

        }
        __device__ static void conv_s2s(
                    const interpolation_parameters &parameters,
                    typename polarization_strategy::pola::vis_pola_type visi,
                    float& conv_norm_weight,
                    size_t ut,
                    size_t vt,
                    float u_tap, 
                    float v_tap, 
                    float u_frac, 
                    float v_frac,
                    int u_int,
                    int v_int,
                    size_t &thread_gridU, 
                    size_t &thread_gridV,
                    conv_function_type &weight,
                    const conv_function_type *filter
                ){
           

            //float u_tap, v_tap;
            //float new_u = ut - float(parameters.half_Nx);
            //float new_v = vt - float(parameters.half_Ny);

            size_t thread_convU, thread_convV;

            //int u_int = rintf(new_u);
            //int v_int = rintf(new_v);

            //float u_frac = float(u_int) - new_u;
            //float v_frac = float(v_int) - new_v;

            //u_tap = (((size_t(u_int)+parameters.half_support_function+parameters.half_Nx)-threadIdx.x)%parameters.full_support_function) + 1;
            //v_tap = (v_int+parameters.half_support_function+parameters.half_Nx-  (size_t)floor((float)(threadIdx.x/parameters.full_support_function)))%parameters.full_support_function + 1;

            thread_gridU = u_int + u_tap + parameters.half_Nx - parameters.half_support_function;
            thread_gridV = v_int + v_tap + parameters.half_Ny - parameters.half_support_function;

            thread_convU = rintf((u_tap + u_frac + 1)*parameters.oversampling_factor);
            thread_convV = rintf((v_tap + v_frac + 1)*parameters.oversampling_factor);

            //weight = parameters.filter_AA_2D[parameters.filter_Prola_2D_size*thread_convV + thread_convU];
            weight = filter[parameters.filter_AA_2D_size*thread_convV + thread_convU];

        }


};

