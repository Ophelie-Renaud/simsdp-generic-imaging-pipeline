#pragma once
#include "../../common/polarization_common.h"
#include "../../common/spec_types.h"
#include "../../common/complex_structure.h"
#include "../../common/interpolation_parameters.h"
#include "../../common/polarization_vector.h"


template <typename T>
class polarization_gridding_strat{
    public:
        typedef polarization<single_polarization> pola;
        __device__  static void read_visi(
                                interpolation_parameters &parameters, 
                                size_t it, 
                                size_t ch,
                                size_t ng_pola,
                                pola::vis_pola_type &visi);

        __device__ static void grid_visibility(
                                interpolation_parameters &parameters,
                                size_t ch,
                                size_t ng_pola,
                                size_t thread_gridV,
                                size_t thread_gridU,
                                pola::vis_pola_type adder, 
                                size_t idx);

        __device__ static void cumul_visibilities(
                                interpolation_parameters parameters, 
                                pola::vis_pola_type visi, 
                                float weight, 
                                pola::vis_pola_type &adder);

        __device__ static void raz_adder(
                                pola::vis_pola_type &adder);
};



template <>
class polarization_gridding_strat<quad_polarization>{
    public:
        typedef polarization<quad_polarization> pola;
        __device__ static void read_visi(
                                interpolation_parameters &parameters,
                                size_t it,
                                size_t ch,
                                size_t ng_pola,
                                pola::vis_pola_type &visi){
            visi = ((pola::vis_pola_type*)parameters.visibilities)[(it*parameters.Nchan+ch)*ng_pola/4];
        }



        __device__ static void grid_visibility(
                                interpolation_parameters &parameters, 
                                size_t ch,
                                size_t ng_pola,
                                size_t thread_gridV,
                                size_t thread_gridU,
                                pola::vis_pola_type adder,
                                size_t idx){

            //size_t idx = (ch*parameters.Nx*parameters.Ny + thread_gridV*parameters.Nx + thread_gridU)*ng_pola*2; 
            //idx = (ch*Nx*Ny + thread_gridV*Nx + thread_gridU)*ng_pola*2; 
            
            atomicAdd(&(((float*)(parameters.output_grid+idx+0))[0]),  adder._one._real);
            atomicAdd(&(((float*)(parameters.output_grid+idx+1))[0]),  adder._one._imag);
            atomicAdd(&(((float*)(parameters.output_grid+idx+2))[0]),  adder._two._real);
            atomicAdd(&(((float*)(parameters.output_grid+idx+3))[0]),  adder._two._imag);
            atomicAdd(&(((float*)(parameters.output_grid+idx+4))[0]),  adder._three._real);
            atomicAdd(&(((float*)(parameters.output_grid+idx+5))[0]),  adder._three._imag);
            atomicAdd(&(((float*)(parameters.output_grid+idx+6))[0]),  adder._four._real);
            atomicAdd(&(((float*)(parameters.output_grid+idx+7))[0]),  adder._four._imag);
            

        }


        __device__ static void cumul_visibilities(
                                interpolation_parameters parameters, 
                                pola::vis_pola_type visi, 
                                float weight, 
                                pola::vis_pola_type &adder){

            adder._one._real   += visi._one._real*weight;
            adder._one._imag   += visi._one._imag*weight;
            adder._two._real   += visi._two._real*weight;
            adder._two._imag   += visi._two._imag*weight;
            adder._three._real += visi._three._real*weight;
            adder._three._imag += visi._three._imag*weight;
            adder._four._real  += visi._four._real*weight;
            adder._four._imag  += visi._four._imag*weight;
        }

        __device__ static void raz_adder(pola::vis_pola_type &adder){

                adder = 0.;
                /*
                adder._one._real=0.;
                adder._one._imag=0;
                adder._two._real=0;
                adder._two._imag=0;
                adder._three._real=0;
                adder._three._imag=0;
                adder._four._real=0;
                adder._four._imag=0;
                */
        }
};





