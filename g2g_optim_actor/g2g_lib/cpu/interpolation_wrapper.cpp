#include <string>
#include <cstdio>

#include "omp.h"

#include "convolution_strategies.h"
#include "polarization_strategies.h"
#include "interpolation_parameters.h"

#include "interpolation_template.h"
#include "../common/interpolation_wrapper.h"
#include "../common/polarization_common.h"
#include "sky2sky_matrix.h"




extern "C"
{

    void init(interpolation_parameters &parameters)
    {
        //printf("There is %zu rows in the MS file \n", parameters->nrows);
    }

    void free_params(interpolation_parameters &parameters)
    {
    }

    void gridding_psf(interpolation_parameters &parameters){
        printf("In interpolation_wrapper.cpp, gridding_PSF\n");
        printf("PSF : Convolution kernel is 2D AA\n");

        typedef polarization_gridding_strat<quad_polarization> polarization_strategy;
        typedef convolution_gridding_strat<polarization_strategy, conv_AA_2D> convolution_stategy;

        grid_template<convolution_stategy,polarization_strategy>(parameters);

        // Reset weight
        for(int i=0; i<parameters.no_grid_index; i++){
            parameters.conv_norm_weight[i] = 0;
        }
    }


    void gridding_quad_pola(interpolation_parameters &parameters)
    {
        printf("In interpolation_wrapper.cpp, gridding_quad_pola\n");
        printf("Convolution kernel is 2D AA\n");

        typedef polarization_gridding_strat<quad_polarization> polarization_strategy;
        typedef convolution_gridding_strat<polarization_strategy, conv_AA_2D> convolution_stategy;

        grid_template<convolution_stategy,polarization_strategy>(parameters);
    }


    void degridding_quad_pola(interpolation_parameters &parameters)
    {
        printf("In interpolation_wrapper.cpp, Degridding_quad_pola\n");
        printf("Convolution kernel is 2D AA\n");

        typedef polarization_gridding_strat<quad_polarization> polarization_strategy;
        typedef convolution_gridding_strat<polarization_strategy, conv_AA_2D> convolution_stategy;

        degrid_template<convolution_stategy,polarization_strategy>(parameters);
    }


    void dgg_init_s2s(interpolation_parameters &parameters)
    {
        printf("I'm in dgg init\n");
        //get_sky2sky_matrix_v1(parameters);
        get_sky2sky_matrix_v3(parameters);
        printf("I'm in dgg init - Done \n");
    }
    
    void s2s_single_pola(interpolation_parameters &parameters)
    {
    }

    void s2s_quad_pola(interpolation_parameters &parameters)
    {

        printf("I'm in s2s_quad_pola\n");
        printf("Convolution kernel is 2D AA\n");

        typedef polarization_gridding_strat<quad_polarization> polarization_strategy;
        typedef convolution_gridding_strat<polarization_strategy, conv_AA_2D> convolution_stategy;

        s2s_template_v2<convolution_stategy,polarization_strategy>(parameters);
    }
}
