#pragma once

#include "interpolation_parameters.h"
#include "../common/complex_structure.h"
#include "../common/spec_types.h"
#include "../common/polarization_vector.h"
#include "../common/uvw_structure.h"


void get_sky2sky_matrix_v0(interpolation_parameters &parameters);
void get_sky2sky_matrix_v1(interpolation_parameters &parameters);


//  V0 : Saving N_hit visibility and channel per slice  as zigzag first = Channel first then row then slice
// No W term
void get_sky2sky_matrix_v0(interpolation_parameters &parameters){
   
    /*
    size_t ch_idx = 0;
    size_t size_coo = 1;
    int slice_grid_size_coo = 1;
    size_t do_write = 0;

    //TODO Change int to uint
    //
    int* u_coo=NULL;
    int* v_coo=NULL;
    int* a_coo=NULL;
    int* ch_coo = NULL;

    int* new_u_coo = NULL;
    int* new_v_coo = NULL;
    int* new_a_coo = NULL;
    int* new_ch_coo = NULL;

    u_coo = (int*)calloc(1, sizeof(int));
    v_coo = (int*)calloc(1, sizeof(int));
    a_coo = (int*)calloc(1, sizeof(int));
    ch_coo = (int*)calloc(1, sizeof(int));

    // Loop over the Channel grid slices
    for(size_t ch_g_idx=0; ch_g_idx<parameters.no_grid_index; ch_g_idx++){
        slice_grid_size_coo = 1;
        
        printf("In nhit v0 building, ch_g_idx=%zu, width=%zu\n", ch_g_idx, parameters.grid_channel_width); 


        // Loop over channels in the grid slice
        for(size_t ch=ch_g_idx*parameters.grid_channel_width; ch<((ch_g_idx+1)*parameters.grid_channel_width); ch++){

            printf("In ch Loop ch=%zu, size_coo=%zu\n", ch, size_coo);
            freq_type wavelength = parameters.chan_wavelength[ch];
            uvw_struct<uvw_coordinates_type> uvw = ((uvw_struct<uvw_coordinates_type>*) parameters.uvw_coordinates)[0];
            uvw._u = (uvw._u*parameters.u_scale)/wavelength;
            uvw._v = (uvw._v*parameters.v_scale)/wavelength;

            int u_int = std::round(uvw._u);
            int v_int = std::round(uvw._v);

            uvw_coordinates_type u_frac = u_int - uvw._u;
            uvw_coordinates_type v_frac = v_int - uvw._v;


            //TODO Check if convolution kernel is out of bound
            for(size_t i=ch_idx; i<size_coo; i++){
                if(u_coo[i] == ((int)(std::round((uvw._u*parameters.oversampling_factor) + (parameters.half_Nx*parameters.oversampling_factor))))){
                    if(v_coo[i] == ((int)(std::round((uvw._v*parameters.oversampling_factor) + (parameters.half_Ny*parameters.oversampling_factor))))){
                        do_write = 0;
                        ch_idx   = i;
                        break;
                    }
                }
            }


            if(do_write == 1){
                new_u_coo = (int*) realloc(u_coo, (size_coo+1)*sizeof(int));
                new_v_coo = (int*) realloc(v_coo, (size_coo+1)*sizeof(int));
                new_a_coo = (int*) realloc(a_coo, (size_coo+1)*sizeof(int));
                new_ch_coo = (int*) realloc(ch_coo, (size_coo+1)*sizeof(int));

                // TODO : Check NULL pointers

                u_coo = new_u_coo;
                v_coo = new_v_coo;
                a_coo = new_a_coo;
                ch_coo = new_ch_coo;

                u_coo[size_coo] =  (int)(std::round((uvw._u*parameters.oversampling_factor))+(parameters.half_Nx*parameters.oversampling_factor));
                v_coo[size_coo] =  (int)(std::round((uvw._v*parameters.oversampling_factor))+(parameters.half_Ny*parameters.oversampling_factor));
                a_coo[size_coo] = 1;
                ch_coo[size_coo] = ch_g_idx;
                size_coo += 1;
                slice_grid_size_coo += 1;

            }    
            else{
                a_coo[ch_idx] += 1;
                do_write = 1;
            }
        }

        for(size_t r=1; r<parameters.nrows; r++){
            for(size_t ch=ch_g_idx*parameters.grid_channel_width; ch<(ch_g_idx+1)*parameters.grid_channel_width; ch++){
                freq_type wavelength = parameters.chan_wavelength[ch];
                uvw_struct<uvw_coordinates_type> uvw = ((uvw_struct<uvw_coordinates_type>*) parameters.uvw_coordinates)[r];
                uvw._u = (uvw._u*parameters.u_scale)/wavelength;
                uvw._v = (uvw._v*parameters.v_scale)/wavelength;
                    

                int u_int = std::round(uvw._u);
                int v_int = std::round(uvw._v);

                uvw_coordinates_type u_frac = u_int - uvw._u;
                uvw_coordinates_type v_frac = v_int - uvw._v;

                //TODO Check if convolution kernel is out of bound
                for(size_t i=ch_idx; i<size_coo; i++){
                    if(u_coo[i] == ((int)(std::round((uvw._u*parameters.oversampling_factor) + (parameters.half_Nx*parameters.oversampling_factor))))){
                        if(v_coo[i] == ((int)(std::round((uvw._v*parameters.oversampling_factor) + (parameters.half_Ny*parameters.oversampling_factor))))){
                            do_write = 0;
                            ch_idx   = i;
                            break;
                        }
                    }
                }

                if(do_write == 1){
                    new_u_coo = (int*) realloc(u_coo, (size_coo+1)*sizeof(int));
                    new_v_coo = (int*) realloc(v_coo, (size_coo+1)*sizeof(int));
                    new_a_coo = (int*) realloc(a_coo, (size_coo+1)*sizeof(int));
                    new_ch_coo = (int*) realloc(ch_coo, (size_coo+1)*sizeof(int));

                    // TODO : Check NULL pointers

                    u_coo = new_u_coo;
                    v_coo = new_v_coo;
                    a_coo = new_a_coo;
                    ch_coo = new_ch_coo;

                    u_coo[size_coo] =  (int)(std::round((uvw._u*parameters.oversampling_factor))+(parameters.half_Nx*parameters.oversampling_factor));
                    v_coo[size_coo] =  (int)(std::round((uvw._v*parameters.oversampling_factor))+(parameters.half_Ny*parameters.oversampling_factor));
                    a_coo[size_coo] = 1;
                    ch_coo[size_coo] = ch_g_idx;
                    size_coo += 1;
                    slice_grid_size_coo += 1;

                }    
                else{
                    a_coo[ch_idx] += 1;
                    do_write = 1;
                }

            }//end ch in slice
        }//end row

        parameters.len_s2s_coo[ch_g_idx] = slice_grid_size_coo;

    }// end slice idx

    parameters.a_coo = a_coo;
    parameters.u_coo = u_coo;
    parameters.v_coo = v_coo;
    */
}



// V1 : Saving N_hit row first then channel then slice
// No W term
void get_sky2sky_matrix_v1(interpolation_parameters &parameters){

    size_t ch_idx = 0;
    size_t size_coo = 1;
    int slice_grid_size_coo = 1;
    int local_coo_size = 0;
    size_t do_write = 0;

    //TODO Change int to uint
    //
    float* u_coo  = NULL;
    float* v_coo  = NULL;
    int* a_coo  = NULL;
    int* ch_coo = NULL;

    float* new_u_coo  = NULL;
    float* new_v_coo  = NULL;
    int* new_a_coo  = NULL;
    int* new_ch_coo = NULL;

    u_coo = (float*)calloc(1, sizeof(float));
    v_coo = (float*)calloc(1, sizeof(float));
    a_coo = (int*)calloc(1, sizeof(int));
    ch_coo = (int*)calloc(1, sizeof(int));


    // Loop over the Channel grid slices
    for(size_t ch_g_idx=0; ch_g_idx<parameters.no_grid_index; ch_g_idx++){
        slice_grid_size_coo = 1;
        
        printf("In nhit v1 building, ch_g_idx=%zu, width=%zu\n", ch_g_idx, parameters.grid_channel_width); 


        // Loop over channels in the grid slice
        for(size_t ch=ch_g_idx*parameters.grid_channel_width; ch<((ch_g_idx+1)*parameters.grid_channel_width); ch++){

            printf("In ch Loop ch=%zu, size_coo=%zu\n", ch, size_coo);
            freq_type wavelength = parameters.chan_wavelength[ch];
            uvw_struct<uvw_coordinates_type> uvw = ((uvw_struct<uvw_coordinates_type>*) parameters.uvw_coordinates)[0];
            uvw._u = (uvw._u*parameters.u_scale)/wavelength;
            uvw._v = (uvw._v*parameters.v_scale)/wavelength;

            int u_int = std::round(uvw._u);
            int v_int = std::round(uvw._v);

            uvw_coordinates_type u_frac = u_int - uvw._u;
            uvw_coordinates_type v_frac = v_int - uvw._v;


            //TODO Check if convolution kernel is out of bound
            for(size_t i=ch_idx; i<size_coo; i++){
                if(u_coo[i] == ((std::round((uvw._u*parameters.oversampling_factor) + (parameters.half_Nx*parameters.oversampling_factor)))/parameters.oversampling_factor)){
                    if(v_coo[i] == ((std::round((uvw._v*parameters.oversampling_factor) + (parameters.half_Ny*parameters.oversampling_factor)))/parameters.oversampling_factor)){
                        do_write = 0;
                        ch_idx   = i;
                        break;
                    }
                }
            }


            if(do_write == 1){
                new_u_coo = (float*) realloc(u_coo, (size_coo+1)*sizeof(float));
                new_v_coo = (float*) realloc(v_coo, (size_coo+1)*sizeof(float));
                new_a_coo = (int*) realloc(a_coo, (size_coo+1)*sizeof(int));
                new_ch_coo = (int*) realloc(ch_coo, (size_coo+1)*sizeof(int));

                // TODO : Check NULL pointers

                u_coo = new_u_coo;
                v_coo = new_v_coo;
                a_coo = new_a_coo;
                ch_coo = new_ch_coo;

                u_coo[size_coo] =  (std::round((uvw._u*parameters.oversampling_factor))+(parameters.half_Nx*parameters.oversampling_factor))/parameters.oversampling_factor;
                v_coo[size_coo] =  (std::round((uvw._v*parameters.oversampling_factor))+(parameters.half_Ny*parameters.oversampling_factor))/parameters.oversampling_factor;
                a_coo[size_coo] = 1;
                ch_coo[size_coo] = ch_g_idx;
                size_coo += 1;
                slice_grid_size_coo += 1;

            }    
            else{
                a_coo[ch_idx] += 1;
                do_write = 1;
            }
        }

        for(size_t ch=ch_g_idx*parameters.grid_channel_width; ch<(ch_g_idx+1)*parameters.grid_channel_width; ch++){
            for(size_t r=1; r<parameters.nrows; r++){
                freq_type wavelength = parameters.chan_wavelength[ch];
                uvw_struct<uvw_coordinates_type> uvw = ((uvw_struct<uvw_coordinates_type>*) parameters.uvw_coordinates)[r];
                uvw._u = (uvw._u*parameters.u_scale)/wavelength;
                uvw._v = (uvw._v*parameters.v_scale)/wavelength;
                    

                int u_int = std::round(uvw._u);
                int v_int = std::round(uvw._v);

                uvw_coordinates_type u_frac = u_int - uvw._u;
                uvw_coordinates_type v_frac = v_int - uvw._v;

                //TODO Check if convolution kernel is out of bound
                for(size_t i=ch_idx; i<size_coo; i++){
                    if(u_coo[i] == ((std::round((uvw._u*parameters.oversampling_factor) + (parameters.half_Nx*parameters.oversampling_factor)))/parameters.oversampling_factor)){
                        if(v_coo[i] == ((std::round((uvw._v*parameters.oversampling_factor) + (parameters.half_Ny*parameters.oversampling_factor)))/parameters.oversampling_factor)){
                            do_write = 0;
                            ch_idx   = i;
                            break;
                        }
                    }
                }

                if(do_write == 1){
                    new_u_coo = (float*) realloc(u_coo, (size_coo+1)*sizeof(float));
                    new_v_coo = (float*) realloc(v_coo, (size_coo+1)*sizeof(float));
                    new_a_coo = (int*) realloc(a_coo, (size_coo+1)*sizeof(int));
                    new_ch_coo = (int*) realloc(ch_coo, (size_coo+1)*sizeof(int));

                    // TODO : Check NULL pointers

                    u_coo = new_u_coo;
                    v_coo = new_v_coo;
                    a_coo = new_a_coo;
                    ch_coo = new_ch_coo;

                    u_coo[size_coo] =  (std::round((uvw._u*parameters.oversampling_factor))+(parameters.half_Nx*parameters.oversampling_factor))/parameters.oversampling_factor;
                    v_coo[size_coo] =  (std::round((uvw._v*parameters.oversampling_factor))+(parameters.half_Ny*parameters.oversampling_factor))/parameters.oversampling_factor;
                    a_coo[size_coo] = 1;
                    ch_coo[size_coo] = ch_g_idx;
                    size_coo += 1;
                    slice_grid_size_coo += 1;

                    local_coo_size += 1;

                }    
                else{
                    a_coo[ch_idx] += 1;
                    do_write = 1;
                }

            }//end ch in slice
            printf("Local coo size = %d\n", local_coo_size);
            local_coo_size = 0;
        }//end row
        
        parameters.len_s2s_coo[ch_g_idx] = slice_grid_size_coo;

    }// end slice idx

    parameters.a_coo = a_coo;
    parameters.u_coo = u_coo;
    parameters.v_coo = v_coo;
    parameters.ch_coo = ch_coo;
    parameters.tot_size_coo = 0;
    for(size_t i=0; i<parameters.no_grid_index; i++){
        parameters.tot_size_coo += parameters.len_s2s_coo[i];
    }
    printf("Total coo size is %zu\n", parameters.tot_size_coo);
}


// TODO
// V2 : Same as V0 but only one Array uvach, instead of 4 array u_coo, v_coo, a_coo, ch_coo
// No W term


//TODO 
// V3 : Same as V1 but only one Array uvach, instead of 4 array u_coo, v_coo, a_coo, ch_coo
// No W term
void get_sky2sky_matrix_v3(interpolation_parameters &parameters){

    size_t ch_idx = 0;
    size_t size_coo = 1;
    int slice_grid_size_coo = 1;
    int local_coo_size = 0;
    size_t do_write = 0;

    //TODO Change int to uint
    //
    float* uvach_coo  = NULL;

    float* new_uvach_coo  = NULL;

    uvach_coo = (float*)calloc(4, sizeof(float));


    // Loop over the Channel grid slices
    for(size_t ch_g_idx=0; ch_g_idx<parameters.no_grid_index; ch_g_idx++){
        slice_grid_size_coo = 1;
        
        printf("In nhit v3 building, ch_g_idx=%zu, width=%zu\n", ch_g_idx, parameters.grid_channel_width); 


        // Loop over channels in the grid slice
        for(size_t ch=ch_g_idx*parameters.grid_channel_width; ch<((ch_g_idx+1)*parameters.grid_channel_width); ch++){

            printf("In ch Loop ch=%zu, size_coo=%zu\n", ch, size_coo);
            freq_type wavelength = parameters.chan_wavelength[ch];
            uvw_struct<uvw_coordinates_type> uvw = ((uvw_struct<uvw_coordinates_type>*) parameters.uvw_coordinates)[0];
            uvw._u = (uvw._u*parameters.u_scale)/wavelength;
            uvw._v = (uvw._v*parameters.v_scale)/wavelength;

            int u_int = std::round(uvw._u);
            int v_int = std::round(uvw._v);

            uvw_coordinates_type u_frac = u_int - uvw._u;
            uvw_coordinates_type v_frac = v_int - uvw._v;


            //TODO Check if convolution kernel is out of bound
            for(size_t i=ch_idx; i<size_coo; i++){
                if(uvach_coo[i*4 + 0] == ((std::round((uvw._u*parameters.oversampling_factor) + (parameters.half_Nx*parameters.oversampling_factor)))/parameters.oversampling_factor)){
                    if(uvach_coo[i*4 + 1] == ((std::round((uvw._v*parameters.oversampling_factor) + (parameters.half_Ny*parameters.oversampling_factor)))/parameters.oversampling_factor)){
                        do_write = 0;
                        ch_idx   = i;
                        break;
                    }
                }
            }


            if(do_write == 1){
                if((std::round((uvw._u*parameters.oversampling_factor))+(parameters.half_Nx*parameters.oversampling_factor))/parameters.oversampling_factor == 0 or 
                   (std::round((uvw._v*parameters.oversampling_factor))+(parameters.half_Ny*parameters.oversampling_factor))/parameters.oversampling_factor == 0) break;

                new_uvach_coo = (float*) realloc(uvach_coo, (size_coo+1)*4*sizeof(float));

                // TODO : Check NULL pointers

                uvach_coo = new_uvach_coo;

                uvach_coo[size_coo*4 + 0] =  (std::round((uvw._u*parameters.oversampling_factor))+(parameters.half_Nx*parameters.oversampling_factor))/parameters.oversampling_factor;
                uvach_coo[size_coo*4 + 1] =  (std::round((uvw._v*parameters.oversampling_factor))+(parameters.half_Ny*parameters.oversampling_factor))/parameters.oversampling_factor;
                uvach_coo[size_coo*4 + 2] = 1;
                uvach_coo[size_coo*4 + 3] = ch_g_idx;
                size_coo += 1;
                slice_grid_size_coo += 1;

            }    
            else{
                uvach_coo[ch_idx*4 + 2] += 1;
                do_write = 1;
            }
        }

        for(size_t ch=ch_g_idx*parameters.grid_channel_width; ch<(ch_g_idx+1)*parameters.grid_channel_width; ch++){
            printf("In ch Loop ch=%zu\n", ch);
            for(size_t r=1; r<parameters.nrows; r++){
                freq_type wavelength = parameters.chan_wavelength[ch];
                uvw_struct<uvw_coordinates_type> uvw = ((uvw_struct<uvw_coordinates_type>*) parameters.uvw_coordinates)[r];
                uvw._u = (uvw._u*parameters.u_scale)/wavelength;
                uvw._v = (uvw._v*parameters.v_scale)/wavelength;
                    

                int u_int = std::round(uvw._u);
                int v_int = std::round(uvw._v);

                uvw_coordinates_type u_frac = u_int - uvw._u;
                uvw_coordinates_type v_frac = v_int - uvw._v;

                //TODO Check if convolution kernel is out of bound
                for(size_t i=ch_idx; i<size_coo; i++){
                    if(uvach_coo[i*4 + 0] == ((std::round((uvw._u*parameters.oversampling_factor) + (parameters.half_Nx*parameters.oversampling_factor)))/parameters.oversampling_factor)){
                        if(uvach_coo[i*4 + 1] == ((std::round((uvw._v*parameters.oversampling_factor) + (parameters.half_Ny*parameters.oversampling_factor)))/parameters.oversampling_factor)){
                            do_write = 0;
                            ch_idx   = i;
                            break;
                        }
                    }
                }

                if(do_write == 1){
                    if((std::round((uvw._u*parameters.oversampling_factor))+(parameters.half_Nx*parameters.oversampling_factor))/parameters.oversampling_factor == 0 or 
                       (std::round((uvw._v*parameters.oversampling_factor))+(parameters.half_Ny*parameters.oversampling_factor))/parameters.oversampling_factor == 0) break;
                    new_uvach_coo = (float*) realloc(uvach_coo, (size_coo+1)*4*sizeof(float));

                    // TODO : Check NULL pointers

                    uvach_coo = new_uvach_coo;

                    uvach_coo[size_coo*4 + 0] =  (std::round((uvw._u*parameters.oversampling_factor))+(parameters.half_Nx*parameters.oversampling_factor))/parameters.oversampling_factor;
                    uvach_coo[size_coo*4 + 1] =  (std::round((uvw._v*parameters.oversampling_factor))+(parameters.half_Ny*parameters.oversampling_factor))/parameters.oversampling_factor;
                    uvach_coo[size_coo*4 + 2] = 1;
                    uvach_coo[size_coo*4 + 3] = ch_g_idx;
                    size_coo += 1;
                    slice_grid_size_coo += 1;

                    local_coo_size += 1;

                    // TODELTE
                    //printf("size_coo = %d, uv = %f, ut = %f\n", size_coo, uvach_coo[size_coo*4 + 0], uvach_coo[size_coo*4 + 1]);
                }    
                else{
                    uvach_coo[ch_idx*4 + 2] += 1;
                    do_write = 1;
                }

            }//end ch in slice
            printf("Local coo size = %d\n", local_coo_size);
            local_coo_size = 0;
        }//end row
        
        parameters.len_s2s_coo[ch_g_idx] = slice_grid_size_coo;

    }// end slice idx

    parameters.uvach_coo = uvach_coo;
    parameters.tot_size_coo = 0;
    //for(size_t i=0; i<parameters.no_grid_index; i++){
    //    parameters.tot_size_coo += parameters.len_s2s_coo[i];
    //}
    parameters.tot_size_coo = size_coo-1;
    printf("Total coo size is %zu\n", parameters.tot_size_coo);
}


//TODO
// V4
// Add W term

