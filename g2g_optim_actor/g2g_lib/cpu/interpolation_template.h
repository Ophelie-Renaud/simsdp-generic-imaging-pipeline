
#pragma once

#include "interpolation_parameters.h"
#include "../common/complex_structure.h"
#include "../common/spec_types.h"
#include "../common/polarization_vector.h"
#include "../common/uvw_structure.h"


template<typename convolution_strategy,
         typename polarization_strategy>
void grid_template(interpolation_parameters &parameters)
{

    typename polarization_strategy::pola::weight_pola_type conv_norm_weight;// TODO move this declaration outside the for loop and reinit to 0 
    typename polarization_strategy::pola::vis_pola_type visi;
    size_t ch_grid_idx = 0;

    for(int i=0; i<parameters.no_grid_index; i++){
        parameters.conv_norm_weight[i] = 0;
    }

    printf(" nRows = %zu \n", parameters.nrows);
    printf(" Nchan is %zu\n", parameters.Nchan); 
    // #pragma omp parallel for schedule(static) private(visi, conv_norm_weight, ch_idx)
    for(size_t r=0; r<parameters.nrows; r++){

        // TODO : spw
        // TODO : flag
        uvw_struct<uvw_coordinates_type> uvw = ((uvw_struct<uvw_coordinates_type>*) parameters.uvw_coordinates)[r];
        for(size_t ch=0; ch<parameters.Nchan; ch++){
            freq_type wavelength = parameters.chan_wavelength[ch];
            ch_grid_idx = parameters.grid_channel_idx[ch];
            uvw_struct<uvw_coordinates_type> uvw_scaled = uvw;
            uvw_scaled._u /= wavelength;
            uvw_scaled._v /= wavelength;

            uvw_scaled._u *= parameters.u_scale;
            uvw_scaled._v *= parameters.v_scale;
            polarization_strategy::read_visi(parameters, r, ch, parameters.polarization_step, visi);
            // TODO : weight multiplication, For now we assumed unitary weight
            visi *= 1;
            conv_norm_weight = 0;
            convolution_strategy::conv(parameters, uvw_scaled, visi, ch_grid_idx, conv_norm_weight);
            parameters.conv_norm_weight[ch_grid_idx] += conv_norm_weight._one;

        }//Nb of Channels
    }// Nb of rows
}


template<typename convolution_strategy,
         typename polarization_strategy>
void degrid_template(interpolation_parameters &parameters)
{
    typename polarization_strategy::pola::weight_pola_type conv_norm_weight;
    typename polarization_strategy::pola::vis_pola_type visi;

    typename polarization_strategy::pola::pixel_pola_type pixel;
    size_t ch_idx = 0;
    printf("In degrid template !\n");
    //#pragma omp parallel for schedule(static) private(visi, pixel, ch_idx, conv_norm_weight)
    for(size_t r=0; r<parameters.nrows; r++){
        uvw_struct<uvw_coordinates_type> uvw = ((uvw_struct<uvw_coordinates_type>*) parameters.uvw_coordinates)[r];

        for(size_t ch=0; ch<parameters.Nchan; ch++){
            ch_idx = parameters.grid_channel_idx[ch];
            freq_type wavelength = parameters.chan_wavelength[ch];
            uvw_struct<uvw_coordinates_type> uvw_scaled = uvw;
            uvw_scaled._u /= wavelength;
            uvw_scaled._v /= wavelength;

            uvw_scaled._u *= parameters.u_scale;
            uvw_scaled._v *= parameters.v_scale;

            convolution_strategy::conv_degrid(parameters, uvw_scaled, ch_idx, r);
        }
    }
    printf("End degrid template ! \n");
    

}


template<typename convolution_strategy,
         typename polarization_strategy>
void s2s_template(interpolation_parameters &parameters)
{

    conv_function_type conv_norm_weight_1 = 0;
    conv_function_type conv_norm_weight_2 = 0;
    typename polarization_strategy::pola::vis_pola_type visi;

    typename polarization_strategy::pola::pixel_pola_type pixel;

    int a, ch_idx, u_int, v_int;
    float ut, vt, u_frac, v_frac, new_u, new_v;

    float cst_u, cst_v = 0;
    cst_u = parameters.half_Nx;//*parameters.oversampling_factor/parameters.oversampling_factor;
    cst_v = parameters.half_Ny;//*parameters.oversampling_factor/parameters.oversampling_factor;

    parameters.conv_norm_weight[0] = 0;
    #pragma omp parallel for schedule(static) private(conv_norm_weight_1, conv_norm_weight_2, visi, pixel, a, ch_idx, u_int, v_int, ut, vt, u_frac, v_frac, new_u, new_v, cst_u, cst_v)
    for(size_t n=1; n<parameters.tot_size_coo; ++n){

        conv_norm_weight_1 = 0;
        
        //ut = parameters.u_coo[n];
        vt = parameters.v_coo[n];
        a = parameters.a_coo[n];
        ch_idx = parameters.ch_coo[n];

        //new_u = (ut - parameters.half_Nx*parameters.oversampling_factor)/parameters.oversampling_factor;
        new_u = parameters.u_coo[n] - cst_u;
        u_int = int(std::round(new_u));
        u_frac = u_int - new_u;

        //new_v = (vt - parameters.half_Ny*parameters.oversampling_factor)/parameters.oversampling_factor;
        new_v = 0;//vt - cst_v;
        v_int = int(std::round(new_v));
        v_frac = v_int - new_v;

        convolution_strategy::conv_s2s_step1(parameters, u_int, u_frac, v_int, v_frac, ch_idx, visi, conv_norm_weight_1);
        //printf("Visi Pre : r = %f, i = %f\n", visi._one._real, visi._one._imag);

        polarization_strategy::apply_norm(visi, conv_norm_weight_1);
        //printf("Visi Post : r = %f, i = %f\n", visi._one._real, visi._one._imag);
        visi *= a;

        convolution_strategy::conv_s2s_step2(parameters, u_int, u_frac, v_int, v_frac, ch_idx, visi, conv_norm_weight_2);

        parameters.conv_norm_weight[ch_idx] += conv_norm_weight_2*a;

    }

}

/* V2 is using coo v3*/
template<typename convolution_strategy,
         typename polarization_strategy>
void s2s_template_v2(interpolation_parameters &parameters)
{
    printf("In G2G V2 template !\n");
    conv_function_type conv_norm_weight_1 = 0;
    conv_function_type conv_norm_weight_2 = 0;
    typename polarization_strategy::pola::vis_pola_type visi;

    typename polarization_strategy::pola::pixel_pola_type pixel;

    int a, ch_idx, u_int, v_int;
    float ut, vt, u_frac, v_frac, new_u, new_v;

    float cst_u, cst_v = 0;
    cst_u = parameters.half_Nx;//*parameters.oversampling_factor/parameters.oversampling_factor;
    cst_v = parameters.half_Ny;//*parameters.oversampling_factor/parameters.oversampling_factor;

    parameters.conv_norm_weight[0] = 0;
    int sum_a = 0;
    //#pragma omp parallel for schedule(static) private(conv_norm_weight_1, conv_norm_weight_2, visi, pixel, a, ch_idx, u_int, v_int, ut, vt, u_frac, v_frac, new_u, new_v, cst_u, cst_v)
    for(size_t n=1; n<parameters.tot_size_coo; ++n){

        conv_norm_weight_1 = 0;
        
        ut = parameters.uvach_coo[n*4 + 0];
        vt = parameters.uvach_coo[n*4 + 1];
        a = int(parameters.uvach_coo[n*4 + 2]);
        ch_idx = int(parameters.uvach_coo[n*4 + 3]);
        if(a == 0){
            printf("a  = 0 || n = %zu, ut = %d, vt = %d, a = %d, ch_idx = %d\n", n, ut, vt, a, ch_idx);   
        }else if(ut == 0){ 
            printf("vt = 0 || n = %zu, ut = %d, vt = %d, a = %d, ch_idx = %d\n", n, ut, vt, a, ch_idx);   
        }else if(vt == 0){  
            printf("ut = 0 || n = %zu, ut = %d, vt = %d, a = %d, ch_idx = %d\n", n, ut, vt, a, ch_idx);   
        }
        
        //printf("n = %zu, ut = %d, vt = %d, a = %d, ch_idx = %d\n", n, ut, vt, a, ch_idx);

        new_u = ut - cst_u;
        u_int = int(std::round(new_u));
        u_frac = u_int - new_u;

        new_v = vt - cst_v;
        v_int = int(std::round(new_v));
        v_frac = v_int - new_v;

        //printf("ut = %d, vt = %d, a = %d, chidx = %d\n", ut, vt, a, ch_idx);
        // TODO : add local timer
        typename polarization_strategy::pola::vis_pola_type visi;
        convolution_strategy::conv_s2s_step1(parameters, u_int, u_frac, v_int, v_frac, ch_idx, visi, conv_norm_weight_1);

        //printf("Check pre norm vis = %f\n", visi._one._real);
        polarization_strategy::apply_norm(visi, conv_norm_weight_1);

        //printf("Check vis = %f\n", visi._one._real);
        visi *= a;
        sum_a += a;

        // TODO : add local timer
        conv_norm_weight_2 = 0;
        convolution_strategy::conv_s2s_step2(parameters, u_int, u_frac, v_int, v_frac, ch_idx, visi, conv_norm_weight_2);
        
        //printf("conv weight = %f\n", conv_norm_weight_2);
        parameters.conv_norm_weight[ch_idx] += conv_norm_weight_2*a;

    }
    printf("Sum a = %d\n", sum_a);

}


/*
template<typename active_convolution_gridding_strat, typename active_polarization_gridding_strat>
void test(){

    printf("This is a test\n");
}
*/
