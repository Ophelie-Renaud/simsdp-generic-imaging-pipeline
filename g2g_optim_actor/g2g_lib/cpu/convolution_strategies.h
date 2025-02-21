#pragma once 

#include "interpolation_parameters.h"
#include "../common/spec_types.h"
#include "../common/uvw_structure.h"
#include "../common/complex_structure.h"
#include "../common/polarization_vector.h"
#include <immintrin.h>

class conv_AA_2D{};

template<typename polarization_strategy, typename T>
class convolution_gridding_strat{
    public:
        static void conv(
                    interpolation_parameters &parameters,
                    uvw_struct<uvw_coordinates_type> uvw,
                    typename polarization_strategy::pola::vis_pola_type visi,
                    size_t ch,
                    typename polarization_strategy::pola::weight_pola_type& conv_norm_weight 
                );
    
        static void conv_degrid(
                    interpolation_parameters &parameters,
                    uvw_struct<uvw_coordinates_type> uvw,
                    size_t ch,
                    size_t row
                );

        static void conv_s2s_step1(
                    interpolation_parameters &parameters,
                    int u_int,
                    float u_frac,
                    int v_int,
                    float v_frac,
                    int ch_idx,
                    typename polarization_strategy::pola::vis_pola_type visi,
                    conv_function_type &conv_norm_weight
                );

         static void conv_s2s_step2(
                    interpolation_parameters &parameters,
                    int u_int,
                    float u_frac,
                    int v_int,
                    float v_frac,
                    int ch_idx,
                    typename polarization_strategy::pola::vis_pola_type visi,
                    conv_function_type &conv_norm_weight
                );

};



/* 
 *
 * Convolution Strategy using 2D Anti Aliasing convolution kernel.
 *
 */
template<typename polarization_strategy>
class convolution_gridding_strat<polarization_strategy, conv_AA_2D>{
    public:
        // Gridding Convolution
        static void conv(
                    interpolation_parameters &parameters,
                    uvw_struct<uvw_coordinates_type> uvw,
                    typename polarization_strategy::pola::vis_pola_type visi,
                    size_t ch,
                    typename polarization_strategy::pola::weight_pola_type& conv_norm_weight
                    ){
                    
            int u_int = std::round(uvw._u);
            int v_int = std::round(uvw._v);
           
            uvw_coordinates_type u_frac = u_int - uvw._u;
            uvw_coordinates_type v_frac = v_int - uvw._v;

            size_t k1=0, k2=0;
            size_t tmp_k1=0, tmp_k2=0;
            size_t pix_id1=0, pix_id2=0;
            
            conv_function_type conv_weight = 0.;
            conv_function_type* conv_function= (conv_function_type*)(parameters.filter_AA_2D); 
            typename polarization_strategy::pola::vis_pola_type conv_vis;
            for(int v_tap=0;  v_tap<parameters.half_support_function+parameters.half_support_function+1; ++v_tap){
                k2      = (size_t)std::round((v_tap + v_frac +1)*parameters.oversampling_factor);
                pix_id2 = v_int + v_tap -parameters.half_support_function + parameters.half_Ny;
                for(int u_tap=0; u_tap<parameters.half_support_function+parameters.half_support_function+1; ++u_tap){
                    k1      = (size_t)std::round((u_tap + u_frac + 1)*parameters.oversampling_factor);
                    pix_id1 = u_int + u_tap - parameters.half_support_function + parameters.half_Nx;
                    conv_weight = ((conv_function_type*)(conv_function + parameters.filter_AA_2D_size*k2 + k1))[0];
                    conv_vis = visi*conv_weight;

                    polarization_strategy::grid_visibility(parameters.Nx, ch, pix_id1, pix_id2, conv_vis, parameters.output_grid);
                    conv_norm_weight += conv_weight;
                }
            }

        }

        // Degridding Convolution
        static void conv_degrid(
                    interpolation_parameters &parameters,
                    uvw_struct<uvw_coordinates_type> uvw,
                    size_t ch,
                    size_t row
                    ){
            int u_int = std::round(uvw._u);
            int v_int = std::round(uvw._v);

            uvw_coordinates_type u_frac = u_int - uvw._u;
            uvw_coordinates_type v_frac = v_int - uvw._v;

            size_t k1=0, k2=0;
            size_t pix_id1=0, pix_id2=0;

            conv_function_type conv_weight = 0;
            conv_function_type cumul_conv_weight = 0;

            conv_function_type* conv_function= (conv_function_type*)(parameters.filter_AA_2D);
            typename polarization_strategy::pola::vis_pola_type conv_pixel;
            typename polarization_strategy::pola::vis_pola_type conv_visi;

            for(int v_tap=1; v_tap<parameters.full_support_function+1; ++v_tap){
                k2      = (size_t)std::round((v_tap + v_frac + 1)*parameters.oversampling_factor);
                pix_id2 = v_int + v_tap -parameters.half_support_function + parameters.half_Ny;
                for(int u_tap=1; u_tap<parameters.full_support_function+1; ++u_tap){
                    k1      = (size_t)std::round((u_tap + u_frac + 1)*parameters.oversampling_factor);
                    pix_id1 = u_int + u_tap - parameters.half_support_function + parameters.half_Nx;
                   
                    //TODO : Change for conjugate kernel
                    conv_weight = ((conv_function_type*)(conv_function + parameters.filter_AA_2D_size*k2 + k1))[0];
                    polarization_strategy::degrid_visibility(parameters.Nx, ch, pix_id1, pix_id2, conv_visi, conv_weight, parameters.input_grid); // Multiplication Pixel * weight 
                    cumul_conv_weight += conv_weight; 
                }
            }
            //conv_visi = conv_visi/cumul_conv_weight;
            //printf("Visibility pre-norm is real = %f, imag = %f\n", conv_visi._one._real, conv_visi._one._imag);
            //printf("Norm is %f\n", cumul_conv_weight);
            polarization_strategy::apply_norm(conv_visi, cumul_conv_weight);
            //printf("Visibility post-norm is real = %f, imag = %f\n", conv_visi._one._real, conv_visi._one._imag);
            polarization_strategy::write_visi(parameters, row, ch, 0, conv_visi);
        }

        static void conv_s2s_step1(
                    interpolation_parameters &parameters,
                    int u_int,
                    float u_frac,
                    int v_int,
                    float v_frac,
                    int ch_idx,
                    typename polarization_strategy::pola::vis_pola_type &visi,
                    conv_function_type &conv_norm_weight
                ){


            size_t k1=0, k2=0;
            size_t pix_id1=0, pix_id2=0;
            conv_function_type conv_weight = 0.;
            conv_function_type* conv_function= (conv_function_type*)(parameters.filter_AA_2D);

            for(int v_tap=0;  v_tap<parameters.half_support_function+parameters.half_support_function+1; ++v_tap){ // Case of AA kernel (2)
                k2      = (size_t)std::round((v_tap + v_frac + 1)*parameters.oversampling_factor); // (2)
                pix_id2 = v_int + v_tap -parameters.half_support_function + parameters.half_Ny; // (2)
                for(int u_tap=0; u_tap<parameters.half_support_function+parameters.half_support_function+1; ++u_tap) { //(2)
                    k1      = (size_t)std::round((u_tap + u_frac + 1)*parameters.oversampling_factor); //(2)
                    pix_id1 = u_int + u_tap -parameters.half_support_function+ parameters.half_Nx; //(2)

                    conv_weight = ((conv_function_type*)(conv_function + parameters.filter_AA_2D_size*k2 + k1))[0];
                    conv_norm_weight += conv_weight;
                    polarization_strategy::degrid_visibility(parameters.Nx, ch_idx, pix_id1, pix_id2, visi, conv_weight, parameters.input_grid);
                }
            }

        }

         static void conv_s2s_step2(
                    interpolation_parameters &parameters,
                    int u_int,
                    float u_frac,
                    int v_int,
                    float v_frac,
                    int ch_idx,
                    typename polarization_strategy::pola::vis_pola_type visi,
                    conv_function_type &conv_norm_weight
                ){

             size_t k1=0, k2=0;
             size_t pix_id1=0, pix_id2=0;
             conv_function_type conv_weight = 0.;
             conv_function_type* conv_function= (conv_function_type*)(parameters.filter_AA_2D);
            
             typename polarization_strategy::pola::vis_pola_type conv_vis;


             for(int v_tap=0;  v_tap<parameters.half_support_function+parameters.half_support_function+1; ++v_tap){ // (2)
                 k2      = (size_t)std::round((v_tap + v_frac + 1)*parameters.oversampling_factor); // (2)
                 pix_id2 = v_int + v_tap - parameters.half_support_function + parameters.half_Ny; // (2)
                 for(int u_tap=0; u_tap<parameters.half_support_function+parameters.half_support_function+1; ++u_tap) { // (2)
                     k1      = (size_t)std::round((u_tap + u_frac + 1)*parameters.oversampling_factor); //(2)
                     pix_id1 = u_int + u_tap -parameters.half_support_function+ parameters.half_Nx; //(2)

                     conv_weight = ((conv_function_type*)(conv_function + parameters.filter_AA_2D_size*k2 + k1))[0];
                     conv_vis = visi*conv_weight;
                     conv_norm_weight += conv_weight;
                     polarization_strategy::grid_visibility(parameters.Nx, ch_idx, pix_id1, pix_id2, conv_vis, parameters.output_grid);
                 }
             }

        }
};

