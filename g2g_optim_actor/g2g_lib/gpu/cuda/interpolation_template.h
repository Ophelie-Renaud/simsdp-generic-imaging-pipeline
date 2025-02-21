#pragma once

#include <stdlib.h>
#include <math.h>
#include <iostream>

#include "../../common/interpolation_parameters.h"
#include "../../common/polarization_common.h"
#include "../../common/spec_types.h"
#include "../../common/complex_structure.h"
#include "../../common/polarization_vector.h"

#include "reduction.h"


/*
 * Add Constant Memory for convolution kernel Here
 * Add Shared Memory for Input Grid
 * */


template<typename convolution_strategy,
         typename polarization_strategy>
__global__ void grid_template_shared_memory(interpolation_parameters parameters, size_t no_visi, float* d_conv_weight_norm_array){

    size_t tid = threadIdx.x;
    
    //size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    size_t f_vis = blockIdx.x*no_visi;
    size_t l_vis = f_vis + no_visi ; 
    
    if(f_vis + tid >= parameters.nrows) return;
    
    /* Define local register */
    pola4<complex_struct<visibility_type>> visi;
    pola4<complex_struct<visibility_type>> sum_pix;
    freq_type chan_wavelength; 
    int u_int, v_int;
    size_t thread_gridU=0, thread_gridV=0, prev_thread_gridU=0, prev_thread_gridV=0, thread_convU=0, thread_convV=0, grid_idx=0;
    uvw_coordinates_type u_frac, v_frac;
    conv_function_type weight;
    conv_function_type w_r=1.;
    size_t ng_pola = parameters.nb_grid_polarization; // Number of grid polarization
    
    size_t ch = 0, it = 0;
    size_t ch_idx = 0;
    size_t v_tap = 0, u_tap = 0;

    typename polarization_strategy::pola::vis_pola_type adder;
    polarization_strategy::raz_adder(adder);
    float conv_norm_weight = 0;//TODO

    // Set these values manually for now
    size_t w_step = 24;
    size_t w_idx = 0;

    size_t t = 0;

    for(ch=0; ch<parameters.Nchan; ch++){   
        for(it=f_vis; it<l_vis; it++){
            if(it >= parameters.nrows) return;
            uvw_struct<uvw_coordinates_type> uvw_ch;
            uvw_struct<uvw_coordinates_type> uvw = ((uvw_struct<uvw_coordinates_type>*) parameters.uvw_coordinates)[it];

            chan_wavelength = parameters.chan_wavelength[ch];
            ch_idx = parameters.grid_channel_idx[ch]; // We consider spw_id = 0 --> [spw_id*Nchan + ch]
            uvw_ch._u = uvw._u/chan_wavelength;
            uvw_ch._v = uvw._v/chan_wavelength;
            uvw_ch._u *= parameters.u_scale;
            uvw_ch._v *= parameters.v_scale;

            u_int = round(uvw_ch._u);
            v_int = round(uvw_ch._v);
            u_frac =  u_int - uvw_ch._u;
            v_frac = v_int - uvw_ch._v;

            u_tap = (((u_int+parameters.half_support_function+parameters.half_Nx)-threadIdx.x)%parameters.full_support_function) + 1;
            v_tap = (v_int+parameters.half_support_function+parameters.half_Nx-  (size_t)floor((float)(threadIdx.x/parameters.full_support_function)))%parameters.full_support_function + 1;

            thread_gridU = u_int + u_tap + parameters.half_Nx - parameters.half_support_function;
            thread_gridV = v_int + v_tap + parameters.half_Ny - parameters.half_support_function;

            thread_convU = (size_t)round((u_tap + u_frac + 1)*parameters.oversampling_factor);
            thread_convV = (size_t)round((v_tap + v_frac + 1)*parameters.oversampling_factor);

            if( (prev_thread_gridU != thread_gridU) or (prev_thread_gridV != thread_gridV)){
                // Atomic Add, the two must be the same type
                
                polarization_strategy::grid_visibility(parameters, ch_idx, ng_pola, thread_gridV, thread_gridU, adder, grid_idx);

                grid_idx = (ch_idx*parameters.Nx*parameters.Ny + thread_gridV*parameters.Nx + thread_gridU)*ng_pola*2;
                
                //d_conv_weight_norm_array[tid] += conv_norm_weight;
                conv_norm_weight = 0;

                prev_thread_gridU = thread_gridU;
                prev_thread_gridV = thread_gridV;

                polarization_strategy::raz_adder(adder);
            }
            w_r = parameters.gridding_conv_function[parameters.filter_size*thread_convV + thread_convU];
            
            //conv_norm_weight += w_r;
            atomicAdd(&d_conv_weight_norm_array[tid], w_r);
            //d_conv_weight_norm_array[tid] += w_r;

            adder._one._real   += (((float*)parameters.visibilities)[it*8*parameters.Nchan+ch*8+0]*w_r);
            adder._one._imag   += (((float*)parameters.visibilities)[it*8*parameters.Nchan+ch*8+1]*w_r);
            adder._two._real   += (((float*)parameters.visibilities)[it*8*parameters.Nchan+ch*8+2]*w_r);
            adder._two._imag   += (((float*)parameters.visibilities)[it*8*parameters.Nchan+ch*8+3]*w_r);
            adder._three._real += (((float*)parameters.visibilities)[it*8*parameters.Nchan+ch*8+4]*w_r);
            adder._three._imag += (((float*)parameters.visibilities)[it*8*parameters.Nchan+ch*8+5]*w_r);
            adder._four._real  += (((float*)parameters.visibilities)[it*8*parameters.Nchan+ch*8+6]*w_r);
            adder._four._imag  += (((float*)parameters.visibilities)[it*8*parameters.Nchan+ch*8+7]*w_r);
        }
    }
    polarization_strategy::grid_visibility(parameters, ch_idx, ng_pola, thread_gridV, thread_gridU, adder, grid_idx);
    //d_conv_weight_norm_array[tid] += conv_norm_weight;
}    

template<typename convolution_strategy,
         typename polarization_strategy,
         unsigned int blockSize>
__global__ void degrid_template_shared_memory(interpolation_parameters parameters, size_t no_visi){

    // 64*8 : For 8x8 convolution kernel
    // 256*8: For 16x16 convolution kernel
    // 1024*8 : For 32x32 convolution kernel, and more
    // 81*8 : For 9x9 convolution kernel
    __shared__ float sdata[81*8]; 

    size_t tid = threadIdx.x;

    size_t f_vis = blockIdx.x*no_visi;
    size_t l_vis = f_vis + no_visi ; 
    
    size_t left_id = 8*tid;
    //size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    float test = 0;
    // Init Shared Mem to 0
    for(size_t i=tid; i<81; i+=blockDim.x){
        sdata[i*8 + 0] = 0;
        sdata[i*8 + 1] = 0;
        sdata[i*8 + 2] = 0;
        sdata[i*8 + 3] = 0;
        sdata[i*8 + 4] = 0;
        sdata[i*8 + 5] = 0;
        sdata[i*8 + 6] = 0;
        sdata[i*8 + 7] = 0;
    }
    
    __syncthreads();
    
    if(f_vis + tid >= parameters.nrows) return;


    // Define local register 
    pola4<complex_struct<visibility_type>> conv_visi;
    pola4<complex_struct<visibility_type>> input_pixel;
    size_t thread_gridU=0, thread_gridV=0, prev_thread_gridU=0, prev_thread_gridV=0, thread_convU=0, thread_convV=0, grid_idx=0, prev_grid_idx=0;
    float ut, vt, u_frac, v_frac, new_u, new_v, a;
    int ch_idx, u_int, v_int;

    conv_function_type weight;
    size_t ng_pola = parameters.nb_grid_polarization; // Number of grid polarization

    size_t it = 0;
    size_t v_tap = 0, u_tap = 0;
    size_t ch = 0;

    float cst_u = parameters.half_Nx;
    float cst_v = parameters.half_Ny;

    typename polarization_strategy::pola::vis_pola_type visi;
    typename polarization_strategy::pola::vis_pola_type adder;
    polarization_strategy::raz_adder(adder);
    float conv_norm_weight;//TODO
    conv_function_type w_r;
    conv_function_type w_i;
    freq_type chan_wavelength;
    
    // Set these values manually for now
    size_t w_step = 24;
    size_t w_idx = 0;
    size_t old_w_idx = 1000;

    unsigned int s;
   
if(8*(tid + 256)+7 == 0)  printf("0");

    for(ch=0; ch<parameters.Nchan; ch++){   
        for(it=f_vis; it<l_vis; it++){
            if(it >= parameters.nrows) return;
            uvw_struct<uvw_coordinates_type> uvw_ch;
            uvw_struct<uvw_coordinates_type> uvw = ((uvw_struct<uvw_coordinates_type>*) parameters.uvw_coordinates)[it];

            w_idx = (abs(uvw_ch._w)/w_step)*264*2;

            chan_wavelength = parameters.chan_wavelength[ch];
            ch_idx = parameters.grid_channel_idx[ch]; // We consider spw_id = 0 --> [spw_id*Nchan + ch]
            uvw_ch._u = uvw._u/chan_wavelength;
            uvw_ch._v = uvw._v/chan_wavelength;
            uvw_ch._u *= parameters.u_scale;
            uvw_ch._v *= parameters.v_scale;
       
            u_int = round(uvw_ch._u);
            v_int = round(uvw_ch._v);
            u_frac =  u_int - uvw_ch._u;
            v_frac = v_int - uvw_ch._v;

            u_tap = (((u_int+parameters.half_support_function+parameters.half_Nx)-threadIdx.x)%parameters.full_support_function) + 1;
            v_tap = (v_int+parameters.half_support_function+parameters.half_Nx-  (size_t)floor((float)(threadIdx.x/parameters.full_support_function)))%parameters.full_support_function + 1;


            thread_gridU = u_int + u_tap + parameters.half_Nx - parameters.half_support_function;
            thread_gridV = v_int + v_tap + parameters.half_Ny - parameters.half_support_function;

            thread_convU = (size_t)round((u_tap + u_frac + 1)*parameters.oversampling_factor);
            thread_convV = (size_t)round((v_tap + v_frac + 1)*parameters.oversampling_factor);

            // Mid working utap vtap 
            // >>4 : 8x8; >> 5 : 16x16 
            //u_tap = (((size_t(u_int)+parameters.half_support_function+parameters.half_Nx)-threadIdx.x)>>5) + 1;
            //v_tap = (v_int+parameters.half_support_function+parameters.half_Nx-  threadIdx.x>>5)>>5 + 1;

            // Update output grid for threads that moved outside the convolution function
            if( (prev_thread_gridU != thread_gridU) or (prev_thread_gridV != thread_gridV)){

                // Atomic Add, the two must be the same type
                prev_grid_idx = (ch_idx*parameters.Nx*parameters.Ny + prev_thread_gridV*parameters.Nx + prev_thread_gridU)*ng_pola*2; 

                grid_idx = (ch_idx*parameters.Nx*parameters.Ny + thread_gridV*parameters.Nx + thread_gridU)*ng_pola*2; 


                prev_thread_gridU = thread_gridU;
                prev_thread_gridV = thread_gridV;

                //input_pixel._one._real  = ((float*)(parameters.input_grid+grid_idx))[0];
                input_pixel._one._real    = parameters.input_grid[grid_idx]     ;
                input_pixel._one._imag    = parameters.input_grid[grid_idx+1];
                input_pixel._two._real    = parameters.input_grid[grid_idx+2];
                input_pixel._two._imag    = parameters.input_grid[grid_idx+3];
                input_pixel._three._real  = parameters.input_grid[grid_idx+4];
                input_pixel._three._imag  = parameters.input_grid[grid_idx+5];
                input_pixel._four._real   = parameters.input_grid[grid_idx+6];
                input_pixel._four._imag   = parameters.input_grid[grid_idx+7];
                polarization_strategy::raz_adder(adder);
            }
            //thread_convU = rintf((u_tap + u_frac + 1)*parameters.oversampling_factor);
            //thread_convV = rintf((v_tap + v_frac + 1)*parameters.oversampling_factor);
            
            w_r = parameters.gridding_conv_function[parameters.filter_size*thread_convV + thread_convU];

            // Degrid
            sdata[left_id + 0] = input_pixel._one._real  *w_r;
            sdata[left_id + 1] = input_pixel._one._imag  *w_r;
            sdata[left_id + 2] = input_pixel._two._real  *w_r;
            sdata[left_id + 3] = input_pixel._two._imag  *w_r;
            sdata[left_id + 4] = input_pixel._three._real*w_r;
            sdata[left_id + 5] = input_pixel._three._imag*w_r;
            sdata[left_id + 6] = input_pixel._four._real *w_r;
            sdata[left_id + 7] = input_pixel._four._imag *w_r; 
        
            __syncthreads();
           
            
            // Reduction
            // V6

           if(blockDim.x >= 512){
                if(tid < 256){
                    sdata[left_id+0] += sdata[8*(tid + 256)+0];
                    sdata[left_id+1] += sdata[8*(tid + 256)+1];
                    sdata[left_id+2] += sdata[8*(tid + 256)+2];
                    sdata[left_id+3] += sdata[8*(tid + 256)+3];
                    sdata[left_id+4] += sdata[8*(tid + 256)+4];
                    sdata[left_id+5] += sdata[8*(tid + 256)+5];
                    sdata[left_id+6] += sdata[8*(tid + 256)+6];
                    sdata[left_id+7] += sdata[8*(tid + 256)+7];
                }
           }
           __syncthreads();

           if(blockDim.x >= 256){
                if(tid < 128){
                    sdata[left_id+0] += sdata[8*(tid + 128)+0];
                    sdata[left_id+1] += sdata[8*(tid + 128)+1];
                    sdata[left_id+2] += sdata[8*(tid + 128)+2];
                    sdata[left_id+3] += sdata[8*(tid + 128)+3];
                    sdata[left_id+4] += sdata[8*(tid + 128)+4];
                    sdata[left_id+5] += sdata[8*(tid + 128)+5];
                    sdata[left_id+6] += sdata[8*(tid + 128)+6];
                    sdata[left_id+7] += sdata[8*(tid + 128)+7];
                }
           }
           __syncthreads();
           if(blockDim.x >= 128){
                if(tid < 64){
                    sdata[left_id+0] += sdata[8*(tid + 64)+0];
                    sdata[left_id+1] += sdata[8*(tid + 64)+1];
                    sdata[left_id+2] += sdata[8*(tid + 64)+2];
                    sdata[left_id+3] += sdata[8*(tid + 64)+3];
                    sdata[left_id+4] += sdata[8*(tid + 64)+4];
                    sdata[left_id+5] += sdata[8*(tid + 64)+5];
                    sdata[left_id+6] += sdata[8*(tid + 64)+6];
                    sdata[left_id+7] += sdata[8*(tid + 64)+7];
                }
           }
           __syncthreads();
           if(tid<32) warpReduce_v5<blockSize>(sdata, tid, left_id);
            
           if(tid==0) {
               atomicAdd(&(((float*)(parameters.visibilities+(it*parameters.Nchan*4 + ch*4)*2))[0]), sdata[0]);
               atomicAdd(&(((float*)(parameters.visibilities+(it*parameters.Nchan*4 + ch*4)*2))[1]), sdata[1]);
               atomicAdd(&(((float*)(parameters.visibilities+(it*parameters.Nchan*4 + ch*4)*2))[2]), sdata[2]);
               atomicAdd(&(((float*)(parameters.visibilities+(it*parameters.Nchan*4 + ch*4)*2))[3]), sdata[3]);
               atomicAdd(&(((float*)(parameters.visibilities+(it*parameters.Nchan*4 + ch*4)*2))[4]), sdata[4]);
               atomicAdd(&(((float*)(parameters.visibilities+(it*parameters.Nchan*4 + ch*4)*2))[5]), sdata[5]);
               atomicAdd(&(((float*)(parameters.visibilities+(it*parameters.Nchan*4 + ch*4)*2))[6]), sdata[6]);
               atomicAdd(&(((float*)(parameters.visibilities+(it*parameters.Nchan*4 + ch*4)*2))[7]), sdata[7]);
           }
            
        }
    }
}







// /* Stategy 2 : Romein/Merry for threads and pixels
//  *             With reduction before gridding.
//  * */

// template<typename convolution_strategy,
//          typename polarization_strategy,
//          unsigned int blockSize>
// __global__ void s2s_template_shared_memory(interpolation_parameters parameters, size_t no_visi){

//     // Shared Mem definition
//     // 64*8 : For 8x8 convolution kernel
//     // 256*8: For 16x16 convolution kernel
//     // 1024*1024 : For 32x32 convolution kernel, and more
//     __shared__ float sdata[1024*8]; 

//     size_t tid = threadIdx.x;
//     size_t left_id = 8*tid;
//     //size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
//     float test = 0;
//     // Init Shared Mem to 0
//     for(size_t i=tid; i<64; i+=blockDim.x){
//         sdata[i*8 + 0] = 0;
//         sdata[i*8 + 1] = 0;
//         sdata[i*8 + 2] = 0;
//         sdata[i*8 + 3] = 0;
//         sdata[i*8 + 4] = 0;
//         sdata[i*8 + 5] = 0;
//         sdata[i*8 + 6] = 0;
//         sdata[i*8 + 7] = 0;
//     }
    
//     __syncthreads();

//     /*
//     for(size_t i=tid; i<1024; i+=blockDim.x){
//         if((1024*blockIdx.x + i) < parameters.tot_size_coo){
//             scoo[i*4 + 0] = parameters.uvach_coo[(blockIdx.x*1024 + i)*4 + 0];
//             scoo[i*4 + 1] = parameters.uvach_coo[(blockIdx.x*1024 + i)*4 + 1];
//             scoo[i*4 + 2] = parameters.uvach_coo[(blockIdx.x*1024 + i)*4 + 2];
//             scoo[i*4 + 3] = parameters.uvach_coo[(blockIdx.x*1024 + i)*4 + 3];
//         }
//     }
//     __syncthreads();
//     */

//     size_t f_vis = blockIdx.x*no_visi;
//     if(f_vis == 0) f_vis = 1; // First row of uvach is full of 0
//     size_t l_vis = f_vis + no_visi ; 

//     if(l_vis > parameters.tot_size_coo) l_vis = parameters.tot_size_coo;

//     //if(idx >= parameters.nrows) return;
    
//     /* Define local register */
//     pola4<complex_struct<visibility_type>> conv_visi;
//     pola4<complex_struct<visibility_type>> input_pixel;
//     size_t thread_gridU=0, thread_gridV=0, prev_thread_gridU=0, prev_thread_gridV=0, thread_convU=0, thread_convV=0, grid_idx=0, prev_grid_idx=0;
//     float ut, vt, u_frac, v_frac, new_u, new_v, a;
//     int ch_idx, u_int, v_int;

//     conv_function_type weight;
//     conv_function_type w_r;
//     conv_function_type w_i;
//     conv_function_type wc_r;
//     conv_function_type wc_i;
//     size_t ng_pola = parameters.nb_grid_polarization; // Number of grid polarization

//     size_t it = 0;
//     size_t v_tap = 0, u_tap = 0;

//     float cst_u = parameters.half_Nx;
//     float cst_v = parameters.half_Ny;

//     typename polarization_strategy::pola::vis_pola_type visi;
//     typename polarization_strategy::pola::vis_pola_type adder;
//     polarization_strategy::raz_adder(adder);
//     float conv_norm_weight;//TODO

//     // Set these values manually for now
//     size_t w_idx = 0;
//     size_t old_w_idx = 1000;

//     unsigned int s;
    
//     // Loop over compressed visibilities
//     //for(it=f_vis; it<l_vis; it++){
//     for(it=f_vis; it<l_vis; it++){

//         // Load compressed elements
//         ut     = parameters.uvach_coo[it*4 + 0];
//         vt     = parameters.uvach_coo[it*4 + 1];
//         a      = parameters.uvach_coo[it*4 + 2];
//         ch_idx = parameters.uvach_coo[it*4 + 3];
  
//         w_idx = ch_idx;
//         new_u = ut - cst_u;
//         new_v = vt - cst_v;

//         u_int = rintf(new_u);
//         v_int = rintf(new_v);

//         u_frac = float(u_int) - new_u;
//         v_frac = float(v_int) - new_v;

//         /* Working utap vtap */
//         u_tap = (((u_int+parameters.half_support_function+parameters.half_Nx)-threadIdx.x)%parameters.full_support_function) + 1;
//         v_tap = (v_int+parameters.half_support_function+parameters.half_Nx-  (threadIdx.x/parameters.full_support_function))%parameters.full_support_function + 1;

//         /* Mid working utap vtap */
//         /* >>4 : 8x8; >> 5 : 16x16 */
//         //u_tap = (((size_t(u_int)+parameters.half_support_function+parameters.half_Nx)-threadIdx.x)>>5) + 1;
//         //v_tap = (v_int+parameters.half_support_function+parameters.half_Nx-  threadIdx.x>>5)>>5 + 1;

//         thread_gridU = u_int + u_tap + parameters.half_Nx - parameters.half_support_function;
//         thread_gridV = v_int + v_tap + parameters.half_Ny - parameters.half_support_function;



//         // Update output grid for threads that moved outside the convolution function
//         if( (prev_thread_gridU != thread_gridU) or (prev_thread_gridV != thread_gridV)){

//             // Atomic Add, the two must be the same type
//             prev_grid_idx = (ch_idx*parameters.Nx*parameters.Ny + prev_thread_gridV*parameters.Nx + prev_thread_gridU)*ng_pola*2; 
//             polarization_strategy::grid_visibility(parameters, ch_idx, ng_pola, prev_thread_gridV, prev_thread_gridU, adder, grid_idx);
               

//             grid_idx = (ch_idx*parameters.Nx*parameters.Ny + thread_gridV*parameters.Nx + thread_gridU)*ng_pola*2; 


//             prev_thread_gridU = thread_gridU;
//             prev_thread_gridV = thread_gridV;

//             input_pixel._one._real    = parameters.input_grid[grid_idx]     ;
//             input_pixel._one._imag    = parameters.input_grid[grid_idx+1];
//             input_pixel._two._real    = parameters.input_grid[grid_idx+2];
//             input_pixel._two._imag    = parameters.input_grid[grid_idx+3];
//             input_pixel._three._real  = parameters.input_grid[grid_idx+4];
//             input_pixel._three._imag  = parameters.input_grid[grid_idx+5];
//             input_pixel._four._real   = parameters.input_grid[grid_idx+6];
//             input_pixel._four._imag   = parameters.input_grid[grid_idx+7];

//             polarization_strategy::raz_adder(adder);
//         }
//         thread_convU = rintf((u_tap + u_frac + 1)*parameters.oversampling_factor);
//         thread_convV = rintf((v_tap + v_frac + 1)*parameters.oversampling_factor);

//         w_r = parameters.w_planes[w_idx + thread_convV*2] * parameters.w_planes[w_idx + thread_convU*2];
//         w_i = parameters.w_planes[w_idx + thread_convV*2+1] * parameters.w_planes[w_idx + thread_convU*2+1];
//         wc_r = parameters.wc_planes[w_idx + thread_convV*2] * parameters.wc_planes[w_idx + thread_convU*2];
//         wc_i = parameters.wc_planes[w_idx + thread_convV*2+1] * parameters.wc_planes[w_idx + thread_convU*2+1];

//         // Degrid
//        sdata[left_id + 0] = input_pixel._one._real  *wc_r - input_pixel._one._imag*wc_i;
//        sdata[left_id + 1] = input_pixel._one._imag  *wc_r + input_pixel._one._real*wc_i;
//        sdata[left_id + 2] = input_pixel._two._real  *wc_r - input_pixel._two._imag*wc_i;
//        sdata[left_id + 3] = input_pixel._two._imag  *wc_r + input_pixel._two._real*wc_i;
//        sdata[left_id + 4] = input_pixel._three._real*wc_r - input_pixel._three._imag*wc_i;
//        sdata[left_id + 5] = input_pixel._three._imag*wc_r + input_pixel._three._real*wc_i;
//        sdata[left_id + 6] = input_pixel._four._real *wc_r - input_pixel._four._imag*wc_i;
//        sdata[left_id + 7] = input_pixel._four._imag *wc_r + input_pixel._four._real*wc_i; 
//         __syncthreads();
       
        
//         // Reduction
//         // V1 0.06
//         /*
//         for(unsigned int s=1; s<blockDim.x; s*=2){
//             if(tid %(2*s) == 0){
//                 sdata[8*tid+0] += sdata[8*(tid + s)+0];
//                 sdata[8*tid+1] += sdata[8*(tid + s)+1];
//                 sdata[8*tid+2] += sdata[8*(tid + s)+2];
//                 sdata[8*tid+3] += sdata[8*(tid + s)+3];
//                 sdata[8*tid+4] += sdata[8*(tid + s)+4];
//                 sdata[8*tid+5] += sdata[8*(tid + s)+5];
//                 sdata[8*tid+6] += sdata[8*(tid + s)+6];
//                 sdata[8*tid+7] += sdata[8*(tid + s)+7];

//             }
//             __syncthreads();
//         }
//         */

//         // V2 0.057
//         /*
//         for(s=1; s<blockDim.x; s*=2){
//             index = 2*s*tid;
//             if(index < blockDim.x)
//             {
//                 sdata[8*index+0] += sdata[8*(index+ s)+0];
//                 sdata[8*index+1] += sdata[8*(index+ s)+1];
//                 sdata[8*index+2] += sdata[8*(index+ s)+2];
//                 sdata[8*index+3] += sdata[8*(index+ s)+3];
//                 sdata[8*index+4] += sdata[8*(index+ s)+4];
//                 sdata[8*index+5] += sdata[8*(index+ s)+5];
//                 sdata[8*index+6] += sdata[8*(index+ s)+6];
//                 sdata[8*index+7] += sdata[8*(index+ s)+7];
               
//             }
//             __syncthreads();
//         }
//         */

//         // V3 0.036
//         /* 
//         for(s=64; s>0; s>>=1){
//             if(tid < s){
//                 sdata[left_id+0] += sdata[8*(tid + s)+0];
//                 sdata[left_id+1] += sdata[8*(tid + s)+1];
//                 sdata[left_id+2] += sdata[8*(tid + s)+2];
//                 sdata[left_id+3] += sdata[8*(tid + s)+3];
//                 sdata[left_id+4] += sdata[8*(tid + s)+4];
//                 sdata[left_id+5] += sdata[8*(tid + s)+5];
//                 sdata[left_id+6] += sdata[8*(tid + s)+6];
//                 sdata[left_id+7] += sdata[8*(tid + s)+7];
//            }
//             __syncthreads();
//         }
        
//        __syncthreads();
//        */
    

//         // V5 
//         /*
//         for(s=blockDim.x/2; s>32; s>>=1){
//             if(tid < s){
//                 sdata[left_id+0] += sdata[8*(tid + s)+0];
//                 sdata[left_id+1] += sdata[8*(tid + s)+1];
//                 sdata[left_id+2] += sdata[8*(tid + s)+2];
//                 sdata[left_id+3] += sdata[8*(tid + s)+3];
//                 sdata[left_id+4] += sdata[8*(tid + s)+4];
//                 sdata[left_id+5] += sdata[8*(tid + s)+5];
//                 sdata[left_id+6] += sdata[8*(tid + s)+6];
//                 sdata[left_id+7] += sdata[8*(tid + s)+7];
//             }
//             __syncthreads();
//         }

//         if(tid<32)warpReduce_v4(sdata, tid, left_id);
//         __syncthreads();
//         */
        
//         // V6

//        if(blockDim.x >= 512){
//             if(tid < 256){
//                 sdata[left_id+0] += sdata[8*(tid + 256)+0];
//                 sdata[left_id+1] += sdata[8*(tid + 256)+1];
//                 sdata[left_id+2] += sdata[8*(tid + 256)+2];
//                 sdata[left_id+3] += sdata[8*(tid + 256)+3];
//                 sdata[left_id+4] += sdata[8*(tid + 256)+4];
//                 sdata[left_id+5] += sdata[8*(tid + 256)+5];
//                 sdata[left_id+6] += sdata[8*(tid + 256)+6];
//                 sdata[left_id+7] += sdata[8*(tid + 256)+7];
//             }
//        }
//        __syncthreads();
//        if(blockDim.x >= 256){
//             if(tid < 128){
//                 sdata[left_id+0] += sdata[8*(tid + 128)+0];
//                 sdata[left_id+1] += sdata[8*(tid + 128)+1];
//                 sdata[left_id+2] += sdata[8*(tid + 128)+2];
//                 sdata[left_id+3] += sdata[8*(tid + 128)+3];
//                 sdata[left_id+4] += sdata[8*(tid + 128)+4];
//                 sdata[left_id+5] += sdata[8*(tid + 128)+5];
//                 sdata[left_id+6] += sdata[8*(tid + 128)+6];
//                 sdata[left_id+7] += sdata[8*(tid + 128)+7];
//             }
//        }
//        __syncthreads();
//        if(blockDim.x >= 128){
//             if(tid < 64){
//                 sdata[left_id+0] += sdata[8*(tid + 64)+0];
//                 sdata[left_id+1] += sdata[8*(tid + 64)+1];
//                 sdata[left_id+2] += sdata[8*(tid + 64)+2];
//                 sdata[left_id+3] += sdata[8*(tid + 64)+3];
//                 sdata[left_id+4] += sdata[8*(tid + 64)+4];
//                 sdata[left_id+5] += sdata[8*(tid + 64)+5];
//                 sdata[left_id+6] += sdata[8*(tid + 64)+6];
//                 sdata[left_id+7] += sdata[8*(tid + 64)+7];
//             }
//        }
//        __syncthreads();
//        if(tid<32) warpReduce_v5<blockSize>(sdata, tid, left_id);


//        // Multiplication with "a" coeff.

//         // Grid
//         weight *=a;
//         adder._one._real   +=(  sdata[0]*w_r - sdata[1]*w_i);
//         adder._one._imag   +=(  sdata[1]*w_r + sdata[0]*w_i);
//         adder._two._real   +=(  sdata[2]*w_r - sdata[3]*w_i);
//         adder._two._imag   +=(  sdata[3]*w_r + sdata[2]*w_i);
//         adder._three._real +=(  sdata[4]*w_r - sdata[5]*w_i);
//         adder._three._imag +=(  sdata[5]*w_r + sdata[4]*w_i);
//         adder._four._real  +=(  sdata[6]*w_r - sdata[7]*w_i);
//         adder._four._imag  +=(  sdata[7]*w_r + sdata[6]*w_i);
//     }

//     polarization_strategy::grid_visibility(parameters, ch_idx, ng_pola, thread_gridV, thread_gridU, adder, grid_idx);
 
// }

