#pragma once 
#include "../common/polarization_common.h"
#include "../common/spec_types.h"
#include "../common/complex_structure.h"
#include "../common/interpolation_parameters.h"
#include "../common/polarization_vector.h"
#include <immintrin.h>


template <typename T>
class polarization_gridding_strat{
    public:
        typedef polarization<single_polarization> pola;
        static void grid_visibility(
                    size_t Nx,
                    size_t ch_index,
                    size_t pix_id1,
                    size_t pix_id2,
                    pola::vis_pola_type conv_vis,
                    fft_grid_type* output_grid
                    );
        static void read_visi(
                    interpolation_parameters &parameters,
                    size_t row,
                    size_t ch,
                    size_t pola_idx,
                    pola::vis_pola_type &vis);

        static void write_visi(
                    interpolation_parameters &parameters,
                    size_t row,
                    size_t ch,
                    size_t pola_idx,
                    pola::vis_pola_type &vis);

        static void degrid_visibility(
                    size_t Nx, 
                    size_t ch_index, 
                    size_t pix_id1, 
                    size_t pix_id2, 
                    pola::vis_pola_type &conv_vis, 
                    conv_function_type conv_weight,
                    fft_grid_type*  input_grid);


        static void apply_norm(
                    pola::vis_pola_type &conv_vis,
                    conv_function_type weight_norm
                );

        static void set_ones(
                    pola::vis_pola_type &vis
                );
        typedef struct avx_type {} avx_type; 
        static void grid_visi_avx(
                     size_t Nx,
                     size_t ch_index,
                     size_t pix_id1,
                     size_t pix_id2,
                     __m256 vis,
                     fft_grid_type* output_grid,
                     size_t differentiate);

        static inline void avx_write_visi(
                    interpolation_parameters &parameters,
                    size_t row, 
                    size_t ch, 
                    size_t pola_idx,
                    __m256 conv_visi);            
};


template<>
class polarization_gridding_strat<quad_polarization>{
    public:
        typedef polarization<quad_polarization> pola;
        static void grid_visibility(
                    size_t Nx,
                    size_t ch_index, // TODO set ch_index instead of ch
                    size_t pix_id1,
                    size_t pix_id2,
                    pola::vis_pola_type conv_vis,
                    fft_grid_type* output_grid
                    ){
            
            fft_grid_type* grid = output_grid + (ch_index*Nx*Nx*4 + pix_id2*Nx*4 + pix_id1*4)*2;
            //printf("pix pix1 = %zu, pix2 = %zu, weight = %f\n", pix_id1, pix_id2, conv_vis._one._real);
            grid[0] += conv_vis._one._real;
            grid[1] += conv_vis._one._imag;
            grid[2] += conv_vis._two._real;
            grid[3] += conv_vis._two._imag;
            grid[4] += conv_vis._three._real;
            grid[5] += conv_vis._three._imag;
            grid[6] += conv_vis._four._real;
            grid[7] += conv_vis._four._imag;
            //printf("Gridded vis1 re = %f, im = %f\n", conv_vis._one._real, conv_vis._one._imag);
        }

        static void degrid_visibility(
                    size_t Nx, 
                    size_t ch, 
                    size_t pix_id1, 
                    size_t pix_id2, 
                    pola::vis_pola_type  &conv_visi, 
                    conv_function_type conv_weight, 
                    fft_grid_type*  input_grid
                    ){
            //fft_grid_type* pixel = input_grid + (ch*Nx*Nx*4 + pix_id2*Nx*4 + pix_id1*4)*2;
            size_t idx = (ch*Nx*Nx*4 + pix_id2*Nx*4 + pix_id1*4)*2;
            
            conv_visi._one._real   += (input_grid + idx)[0]*conv_weight;
            conv_visi._one._imag   += (input_grid + idx)[1]*conv_weight;
            conv_visi._two._real   += (input_grid + idx)[2]*conv_weight;
            conv_visi._two._imag   += (input_grid + idx)[3]*conv_weight;
            conv_visi._three._real += (input_grid + idx)[4]*conv_weight;
            conv_visi._three._imag += (input_grid + idx)[5]*conv_weight;
            conv_visi._four._real  += (input_grid + idx)[6]*conv_weight;
            conv_visi._four._imag  += (input_grid + idx)[7]*conv_weight;



            //conv_visi._one._imag   += pixel[1]*conv_weight;
            //conv_visi._two._real   += pixel[2]*conv_weight;
            //conv_visi._two._imag   += pixel[3]*conv_weight;
            //conv_visi._three._real += pixel[4]*conv_weight;
            //conv_visi._three._imag += pixel[5]*conv_weight;
            //conv_visi._four._real  += pixel[6]*conv_weight;
            //conv_visi._four._imag  += pixel[7]*conv_weight;

        }

        static void read_visi(
                    interpolation_parameters &parameters,
                    size_t row,
                    size_t ch,
                    size_t pola_idx,
                    pola::vis_pola_type &vis){
            size_t idx = (row * parameters.Nchan + ch)*parameters.nb_vis_polarization + pola_idx;
            vis = ((pola::vis_pola_type*)parameters.visibilities)[idx/4];
        }

        static void write_visi(
                    interpolation_parameters &parameters,
                    size_t row,
                    size_t ch,
                    size_t pola_idx,
                    pola::vis_pola_type &vis){
            
            fft_grid_type* visibility = (fft_grid_type*)(parameters.visibilities + (row * parameters.Nchan + ch)*parameters.nb_vis_polarization + pola_idx);
            visibility[0] = vis._one._real;
            visibility[1] = vis._one._imag;
            visibility[2] = vis._two._real;
            visibility[3] = vis._two._imag;
            visibility[4] = vis._three._real;            
            visibility[5] = vis._three._imag;
            visibility[6] = vis._four._real;           
            visibility[7] = vis._four._imag;

        }


        static void apply_norm(
                    pola::vis_pola_type &conv_vis,
                    conv_function_type weight_norm){

            conv_vis._one._real /= (weight_norm+1e-8);
            conv_vis._one._imag /= (weight_norm+1e-8);
            conv_vis._two._real /= (weight_norm+1e-8);
            conv_vis._two._imag /= (weight_norm+1e-8);
            conv_vis._three._real /= (weight_norm+1e-8);
            conv_vis._three._imag /= (weight_norm+1e-8);
            conv_vis._four._real /= (weight_norm+1e-8);
            conv_vis._four._imag /= (weight_norm+1e-8);

        }

        static void set_ones(
                    pola::vis_pola_type &vis
                ){

            vis._one._real = 1.;
            vis._one._imag = 0.;
            vis._two._real = 1.;
            vis._two._imag = 0.;
            vis._three._real = 1.;
            vis._three._imag = 0.;
            vis._four._real = 1.;
            vis._four._imag = 0;
        }

    static inline void grid_visi_avx(
                    size_t Nx,
                    size_t ch_index,
                    size_t pix_id1,
                    size_t pix_id2,
                    __m256 vis,
                    fft_grid_type* output_grid,
                    size_t differentiate){

            fft_grid_type* grid = output_grid + (ch_index*Nx*Nx*4 + pix_id2*Nx*4 + pix_id1*4)*2;
            //float* res = (float*)&vis;
            //printf("vec1 =(%f, %f)\n", vis[0], vis[1]);
            
            grid[0] += ((float*)&vis)[0];//vis[0];
            grid[1] += ((float*)&vis)[1];//vis[1];
            grid[2] += ((float*)&vis)[2];//vis[2];
            grid[3] += ((float*)&vis)[3];//vis[3];
            grid[4] += ((float*)&vis)[4];//vis[4];
            grid[5] += ((float*)&vis)[5];//vis[5];
            grid[6] += ((float*)&vis)[6];//vis[6];
            grid[7] += ((float*)&vis)[7];//vis[7];
            
    }
            
    static inline void avx_write_visi(
                    interpolation_parameters &parameters,
                    size_t row, 
                    size_t ch, 
                    size_t pola_idx,
                    __m256 conv_visi){            

            fft_grid_type* visibility = (fft_grid_type*)(parameters.visibilities + (row * parameters.Nchan + ch)*parameters.nb_vis_polarization);
            visibility[0] = ((float*)&conv_visi)[0];
            visibility[1] = ((float*)&conv_visi)[1];
            visibility[2] = ((float*)&conv_visi)[2];
            visibility[3] = ((float*)&conv_visi)[3];
            visibility[4] = ((float*)&conv_visi)[4];            
            visibility[5] = ((float*)&conv_visi)[5];
            visibility[6] = ((float*)&conv_visi)[6];           
            visibility[7] = ((float*)&conv_visi)[7];
    }

};
