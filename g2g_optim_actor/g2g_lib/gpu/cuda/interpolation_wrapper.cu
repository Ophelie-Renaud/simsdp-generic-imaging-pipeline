#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
//#include <helper_cuda.h>


#include "interpolation_template.h"

#include "polarization_strategies.h"
#include "convolution_strategies.h"
#include "../../common/interpolation_wrapper.h"
#include "../../common/polarization_common.h"
#include "cuda_error.h"
#include "sky2sky_matrix.h"


#define BLOCK_SIZE 1024 // 64 : 8x8;  256 : 16x16; 1024 : 32x32

extern "C"
{
    interpolation_parameters d_parameters;
    int test = 0;
    size_t Memory_mod = 0; // raw=0,cst=1,txture=2
    size_t S2S_mod = 0; //no_S2S=0, S2S=1

    void init(interpolation_parameters &parameters)
    {
        printf("#################################################################\n");
        printf("################### In GPU Init #################################\n");
        printf("#################################################################\n");

        cudaDeviceReset();



        // Count an print all the devices on the node
        int deviceCount = 0;
        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
        if (error_id != cudaSuccess){
            printf("cudaGetDecivecCount return %d -> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
            printf("Result Fail\n");
            exit(EXIT_FAILURE);
        }

        if(deviceCount == 0){
            printf("There is no GPU that support CUDA\n");
        }
        else{
            printf("There is %d GPU(s) available that support CUDA\n", deviceCount);
        }


        // Cuda GPU information
        /*
         * Get All GPU Hardware information.
         *   
        */
        int dev, driverVersion = 0, runtimeVersion = 0;
        for (dev = 0; dev < deviceCount; ++dev) {
            cudaSetDevice(dev);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);

            printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

            // Console logs
            cudaDriverGetVersion(&driverVersion);
            cudaRuntimeGetVersion(&runtimeVersion);
            printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
                    driverVersion / 1000, (driverVersion % 100) / 10,
                    runtimeVersion / 1000, (runtimeVersion % 100) / 10);
            printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
                    deviceProp.major, deviceProp.minor);

            char msg[256];
            snprintf(msg, sizeof(msg),
                    "  Total amount of global memory:                 %.0f MBytes "
                    "(%llu bytes)\n",
                    static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                    (unsigned long long)deviceProp.totalGlobalMem);
            printf("%s", msg);
            printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
            printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);

        }
        printf("Size of float is %d\n", sizeof(float));
        printf("Nchan is %d\n", parameters.Nchan);
        printf("chan_wavelength Size is         %d MB\n", (sizeof(freq_type)*parameters.Nchan)/(1024*1024));
        printf("gridding_conv_function Size is  %d MB\n", (sizeof(conv_function_type)*parameters.filter_size*parameters.filter_size)/(1024*1024));
        printf("output_grid Size is             %d MB\n", (sizeof(fft_grid_type)*parameters.Nx*parameters.Ny*parameters.no_grid_index*parameters.nb_grid_polarization*2)/(1024*1024));
        printf("grid_channel_idx, Size is       %d MB\n", (sizeof(size_t)*parameters.no_chan_spw)/(1024*1024));
        printf("input_grid Size is              %d MB\n", (sizeof(fft_grid_type)*parameters.Nx*parameters.Ny*parameters.no_grid_index*parameters.nb_grid_polarization*2)/(1024*1024));
        printf("uvach_coo Size is               %d MB\n", (sizeof(fft_grid_type)*parameters.tot_size_coo*4)/(1024*1024));
        printf("2D convlution function size     %d bytes\n", sizeof(fft_grid_type)*parameters.filter_AA_2D_size*parameters.filter_AA_2D_size);
        printf("Size of conv function is %d\n", parameters.filter_size*parameters.filter_size);



        /*
         * Allocate memory and copy on GPU such as :
         *      - chan_wavelength        : Array of wavelength 
         *      - gridding_conv_function : Convolution kernel (outdated TOREMOVE)
         *      - output_grid            : output Fourier grid
         *      - grid_channel_idx       : array to map specific wavelength to the right grid slice
         *      - uvw_coordinates        : (STD) uvw coordinates
         *      - visibility_weight      : (STD) Data weight
         *      - visibilities           : (STD) Data
         *      - input_grid             : (S2S) Input Fourier grid
         *      - uvach_coo              : (S2S) nhit data
         *   
        */
        printf("Init Polarization = %zu\n", parameters.nb_grid_polarization);
        checkCudaErrors(cudaMalloc((void**)&d_parameters.chan_wavelength, sizeof(freq_type)*parameters.Nchan));// chan_wavelength
        checkCudaErrors(cudaMalloc((void**)&d_parameters.gridding_conv_function, sizeof(conv_function_type)*parameters.filter_size*parameters.filter_size));// AA conv kernel (No W, only float)
        checkCudaErrors(cudaMalloc((void**)&d_parameters.filter_AA_2D, sizeof(conv_function_type)*parameters.filter_AA_2D_size*parameters.filter_AA_2D_size));// AA conv kernel (No W, only float)

        checkCudaErrors(cudaMalloc((void**)&d_parameters.output_grid, sizeof(fft_grid_type)*parameters.Nx*parameters.Ny*parameters.no_grid_index*parameters.nb_grid_polarization*2)); // Outputgrid Nx*Ny*Nchan*Pola*complex
        checkCudaErrors(cudaMalloc((void**)&d_parameters.grid_channel_idx, sizeof(size_t)*parameters.no_chan_spw));



        /* Maloc regarding gridding mod */
        if(parameters.do_s2s==0){

            // Not S2S Mode
            checkCudaErrors(cudaMalloc((void**)&d_parameters.uvw_coordinates, sizeof(uvw_coordinates_type)*3*parameters.nrows));   // uvw coordinates
            checkCudaErrors(cudaMalloc((void**)&d_parameters.visibility_weight, sizeof(weight_type)*parameters.nb_vis_polarization*parameters.nrows)); // visiblity weights
            checkCudaErrors(cudaMalloc((void**)&d_parameters.visibilities, sizeof(complex_struct<visibility_type>)*parameters.Nchan*parameters.nb_vis_polarization*parameters.nrows));// visibilities
            


        }
        else if(parameters.do_s2s==1){
            // S2S Mode
            printf("Do S2S - Cuda Malloc.\n");
            // TODO : Take W proj into account (*5) not *4
            checkCudaErrors(cudaMalloc((void**)&d_parameters.input_grid, sizeof(fft_grid_type)*parameters.Nx*parameters.Ny*parameters.no_grid_index*parameters.nb_grid_polarization*2)); // Inputgrid Nx*Ny*Nchan*Pola*complex
            checkCudaErrors(cudaMalloc((void**)&d_parameters.uvach_coo, sizeof(float)*parameters.tot_size_coo*4));
        }
        else if(parameters.do_s2s==2){
            // Degridding
            checkCudaErrors(cudaMalloc((void**)&d_parameters.input_grid, sizeof(fft_grid_type)*parameters.Nx*parameters.Ny*parameters.no_grid_index*parameters.nb_grid_polarization*2)); // Inputgrid Nx*Ny*Nchan*Pola*complex
            checkCudaErrors(cudaMalloc((void**)&d_parameters.uvw_coordinates, sizeof(uvw_coordinates_type)*3*parameters.nrows));   // uvw coordinates
            checkCudaErrors(cudaMalloc((void**)&d_parameters.visibility_weight, sizeof(weight_type)*parameters.nb_vis_polarization*parameters.nrows)); // visiblity weights
            checkCudaErrors(cudaMalloc((void**)&d_parameters.visibilities, sizeof(complex_struct<visibility_type>)*parameters.Nchan*parameters.nb_vis_polarization*parameters.nrows));// visibilities
        }



        if(parameters.do_s2s==0 || parameters.do_s2s==2){
            checkCudaErrors(cudaHostRegister(parameters.uvw_coordinates, sizeof(uvw_coordinates_type)*3*parameters.nrows, 0));
            checkCudaErrors(cudaHostRegister(parameters.visibility_weight, sizeof(weight_type)*parameters.nb_vis_polarization*parameters.nrows, 0));
        }
        checkCudaErrors(cudaHostRegister(parameters.chan_wavelength, sizeof(freq_type)*parameters.Nchan, 0));
        checkCudaErrors(cudaHostRegister(parameters.gridding_conv_function, sizeof(conv_function_type)*parameters.filter_size*parameters.filter_size, 0));

        checkCudaErrors(cudaMemcpyAsync(d_parameters.chan_wavelength, parameters.chan_wavelength, sizeof(freq_type)*parameters.Nchan, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_parameters.grid_channel_idx, parameters.grid_channel_idx, sizeof(size_t)*parameters.no_chan_spw, cudaMemcpyHostToDevice));


        // Choose how to copy convolution function : to constant mem or global mem
        if(Memory_mod==0){
            // Raw Memory Mode
            checkCudaErrors(cudaMemcpyAsync(d_parameters.gridding_conv_function, parameters.gridding_conv_function, sizeof(conv_function_type)*parameters.filter_size*parameters.filter_size, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpyAsync(d_parameters.filter_AA_2D, parameters.filter_AA_2D, sizeof(conv_function_type)*parameters.filter_AA_2D_size*parameters.filter_AA_2D_size,                     cudaMemcpyHostToDevice));
        }

        if(parameters.do_s2s==0){
            // Not S2S Mode
            checkCudaErrors(cudaMemcpyAsync(d_parameters.uvw_coordinates, parameters.uvw_coordinates, sizeof(uvw_coordinates_type)*3*parameters.nrows, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpyAsync(d_parameters.visibility_weight, parameters.visibility_weight, sizeof(weight_type)*parameters.nb_vis_polarization*parameters.nrows, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpyAsync(d_parameters.visibilities, parameters.visibilities, sizeof(complex_struct<visibility_type>)*parameters.Nchan*parameters.nb_vis_polarization*parameters.nrows, cudaMemcpyHostToDevice));
        }
        else if (parameters.do_s2s==2){//Degridding
            checkCudaErrors(cudaMemcpyAsync(d_parameters.uvw_coordinates, parameters.uvw_coordinates, sizeof(uvw_coordinates_type)*3*parameters.nrows, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpyAsync(d_parameters.visibility_weight, parameters.visibility_weight, sizeof(weight_type)*parameters.nb_vis_polarization*parameters.nrows, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpyAsync(d_parameters.input_grid, parameters.input_grid, sizeof(fft_grid_type)*parameters.Nx*parameters.Ny*parameters.no_grid_index*parameters.nb_grid_polarization*2, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpyAsync(d_parameters.visibilities, parameters.visibilities, sizeof(complex_struct<visibility_type>)*parameters.Nchan*parameters.nb_vis_polarization*parameters.nrows, cudaMemcpyHostToDevice));
        }
        else if (parameters.do_s2s==1){
            // S2S Mode
            printf("Do S2S - Cuda Copy.\n");
            checkCudaErrors(cudaMemcpyAsync(d_parameters.uvach_coo, parameters.uvach_coo, sizeof(float)*parameters.tot_size_coo*4, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpyAsync(d_parameters.input_grid, parameters.input_grid, sizeof(fft_grid_type)*parameters.Nx*parameters.Ny*parameters.no_grid_index*parameters.nb_grid_polarization*2, cudaMemcpyHostToDevice));
        }


        if(parameters.do_s2s==0 || parameters.do_s2s==2){
            checkCudaErrors(cudaHostUnregister(parameters.uvw_coordinates));
            checkCudaErrors(cudaHostUnregister(parameters.visibility_weight));
            //checkCudaErrors(cudaHostUnregister(parameters.visibilities));
        }

        checkCudaErrors(cudaHostUnregister(parameters.chan_wavelength));
        checkCudaErrors(cudaHostUnregister(parameters.gridding_conv_function));

        cudaDeviceSynchronize();
    }

    void free_parameters(interpolation_parameters &parameters)
    {
        cudaDeviceSynchronize();
        printf("### Free GPU Memory ###\n");
        checkCudaErrors(cudaFree(d_parameters.chan_wavelength));
        checkCudaErrors(cudaFree(d_parameters.gridding_conv_function));
        checkCudaErrors(cudaFree(d_parameters.filter_AA_2D));
        checkCudaErrors(cudaFree(d_parameters.output_grid));
        checkCudaErrors(cudaFree(d_parameters.grid_channel_idx));
        if(parameters.do_s2s==0){
            checkCudaErrors(cudaFree(d_parameters.uvw_coordinates));
            checkCudaErrors(cudaFree(d_parameters.visibility_weight));
            checkCudaErrors(cudaFree(d_parameters.visibilities));
        }
        else if(parameters.do_s2s==1){
            printf("Do S2S - Cuda Free.\n");
            checkCudaErrors(cudaFree(d_parameters.uvach_coo));
            checkCudaErrors(cudaFree(d_parameters.input_grid));
        }
        else if(parameters.do_s2s==1){
            checkCudaErrors(cudaFree(d_parameters.input_grid));
            checkCudaErrors(cudaFree(d_parameters.uvw_coordinates));
            checkCudaErrors(cudaFree(d_parameters.visibility_weight));
            checkCudaErrors(cudaFree(d_parameters.visibilities));
        }

    }

    void gridding_psf(interpolation_parameters &parameters)
    {
    }


/*************************************************************
 *************************************************************
 *********************  GRIDDING *****************************
 ************************************************************* 
 *************************************************************/
    void gridding_quad_pola(interpolation_parameters &parameters)
    {

        printf("#################################################################\n");
        printf("#### In GPU interpolation_wrapper.cpp, gridding_quad_pola #####\n");
        printf("#################################################################\n");


        typedef polarization_gridding_strat<quad_polarization> polarization_strategy;
	    printf("Use conv_AA_2D filter\n");
        typedef convolution_gridding_strat<polarization_strategy, conv_AA_2D> convolution_stategy;
     

        clock_t  start, end;

        /* As [Merry,2016], Visiblities are sorted by baselines.  
           Each block compute the same number of visibilities to avoid unbalanced workload dur to compression.
         */
        size_t visi_per_block = 1024; // Let's set at 1024 for now
        size_t npix_convolution_filter = 81;

        // Create array of conv_norm_weight (Size of "visi_per_block")
        float* conv_weight_norm_array = (float*)malloc(sizeof(float)*npix_convolution_filter);
        for(int i = 0; i< npix_convolution_filter; i++){
            conv_weight_norm_array[i] = 0;
        }
        float conv_weight_norm=0;
        float* d_conv_weight_norm_array;
        checkCudaErrors(cudaMalloc((void**)&d_conv_weight_norm_array, sizeof(float)*npix_convolution_filter));
        checkCudaErrors(cudaMemcpy(d_conv_weight_norm_array, conv_weight_norm_array, sizeof(float)*npix_convolution_filter, cudaMemcpyHostToDevice));

        //parameters.nrows = 1;

        /* 
           V1
           The number of threads is defined by the size of the GCF.
        */
        //const dim3 blockSize(parameters.full_support_function*parameters.full_support_function, 1);

        
        const dim3 blockSize(npix_convolution_filter, 1);

        size_t nb_block = ceil((parameters.nrows)/visi_per_block);
        const dim3 gridSize(nb_block,1);


        printf("There is %d Threads per block\n", parameters.full_support_function*parameters.full_support_function);
        printf("There is %d Blocks\n", nb_block);

        /*
           Set meta data to the device. 
        */
        

        d_parameters.nrows = parameters.nrows;
        d_parameters.Nchan = parameters.Nchan;
        d_parameters.half_support_function = parameters.half_support_function;
        d_parameters.full_support_function = parameters.full_support_function;
        d_parameters.half_Nx = parameters.half_Nx;
        d_parameters.half_Ny = parameters.half_Ny;
        d_parameters.u_scale = parameters.u_scale;
        d_parameters.v_scale = parameters.v_scale;
        d_parameters.max_w   = parameters.max_w;
        d_parameters.Nx = parameters.Nx;
        d_parameters.Ny = parameters.Ny;
        d_parameters.filter_size = parameters.filter_size;
        d_parameters.filter_AA_2D_size = parameters.filter_AA_2D_size;
        d_parameters.oversampling_factor = parameters.oversampling_factor;
        d_parameters.nb_grid_polarization = parameters.nb_grid_polarization;
       

        start = clock();

        printf("Small kernel \n");
        grid_template_shared_memory<convolution_stategy,polarization_strategy><<<gridSize, blockSize>>>(d_parameters, visi_per_block, d_conv_weight_norm_array);
        cudaDeviceSynchronize();
        end = clock();

        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(parameters.output_grid, d_parameters.output_grid, sizeof(fft_grid_type)*parameters.Nx*parameters.Ny*parameters.no_grid_index*parameters.nb_grid_polarization*2, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(conv_weight_norm_array, d_conv_weight_norm_array, sizeof(float)*npix_convolution_filter, cudaMemcpyDeviceToHost));
        
        cudaDeviceSynchronize();
        for(int i=0; i<npix_convolution_filter; i++){
            conv_weight_norm += conv_weight_norm_array[i];
        }
        printf("!!!Total conv weight = %f \n", conv_weight_norm);

        // Transfert Norm weight to data structure
        for(size_t ch_grid_idx=0; ch_grid_idx<parameters.no_grid_index; ch_grid_idx++){
            parameters.conv_norm_weight[ch_grid_idx] = conv_weight_norm;
        }

        checkCudaErrors(cudaFree(d_conv_weight_norm_array));

        /* DEBUG PRINT*/
        printf("###################################\n");
        printf("Total visibilities   : %d (x4 because quad polarization) \n", parameters.nrows*parameters.Nchan);
        printf("Gridding kernel time    : %0.4f seconds\n", ((float) end - start)/CLOCKS_PER_SEC);
        printf("Visibility throughput   : %.4e MVisi/s\n", (4*parameters.nrows*parameters.Nchan)/(((float) end - start)/CLOCKS_PER_SEC));
        printf("Number of threads created = %d threads\n", nb_block*parameters.full_support_function*parameters.full_support_function);
        printf("###################################\n");
    }
   



// /*************************************************************
//  *************************************************************
//  *********************  DEGRIDDING ***************************
//  ************************************************************* 
//  *************************************************************/
    void degridding_quad_pola(interpolation_parameters &parameters)
    {
        printf("#################################################################\n");
        printf("#### In GPU interpolation_wrapper.cpp, Degridding_quad_pola #####\n");
        printf("#################################################################\n");

        typedef polarization_gridding_strat<quad_polarization> polarization_strategy;
        typedef convolution_gridding_strat<polarization_strategy, conv_AA_2D> convolution_stategy;
        typedef convolution_gridding_strat<polarization_strategy, conv_AA_2D> convolution_stategy;

        clock_t  start, end;

        /* As [Merry,2016], Visiblities are sorted by baselines.  
           Each block compute the same number of visibilities to avoid unbalanced workload dur to compression.
         */
        size_t visi_per_block = 1024; // Let's set at 1024 for now

        //parameters.nrows = 1;

        /* 
           V1
           The number of threads is defined by the size of the GCF.
        */
        const dim3 blockSize(BLOCK_SIZE, 1); // conv function : 8x8

        size_t nb_block = ceil((parameters.nrows)/visi_per_block);
        const dim3 gridSize(nb_block,1);


        printf("There is %d Threads per block\n", parameters.full_support_function*parameters.full_support_function);
        printf("There is %d Blocks\n", nb_block);

        /*
           Set meta data to the device. 
        */
        

        d_parameters.nrows = parameters.nrows;
        d_parameters.Nchan = parameters.Nchan;
        d_parameters.half_support_function = parameters.half_support_function;
        d_parameters.full_support_function = parameters.full_support_function;
        d_parameters.half_Nx = parameters.half_Nx;
        d_parameters.half_Ny = parameters.half_Ny;
        d_parameters.u_scale = parameters.u_scale;
        d_parameters.v_scale = parameters.v_scale;
        d_parameters.max_w   = parameters.max_w;
        d_parameters.Nx = parameters.Nx;
        d_parameters.Ny = parameters.Ny;
        d_parameters.filter_size = parameters.filter_size;
        //d_parameters.filter_AA_1D_size = parameters.filter_AA_1D_size;
        d_parameters.filter_AA_2D_size = parameters.filter_AA_2D_size;
        //d_parameters.filter_Prola_2D_size = parameters.filter_Prola_2D_size;
        d_parameters.oversampling_factor = parameters.oversampling_factor;
        d_parameters.nb_grid_polarization = parameters.nb_grid_polarization;
        //d_parameters.w_planes_tot_size = parameters.w_planes_tot_size;
       

        start = clock();
        //grid_template<convolution_stategy,polarization_strategy><<<gridSize, blockSize>>>(d_parameters, visi_per_block);
        printf("Small kernel\n");
        degrid_template_shared_memory<convolution_stategy,polarization_strategy, BLOCK_SIZE><<<gridSize, blockSize>>>(d_parameters, visi_per_block);

        cudaDeviceSynchronize();
        end = clock();

        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(parameters.visibilities, d_parameters.visibilities, sizeof(complex_struct<visibility_type>)*parameters.Nchan*parameters.nb_vis_polarization*parameters.nrows, cudaMemcpyDeviceToHost));
        
        cudaDeviceSynchronize();



        /* DEBUG PRINT*/
        printf("###################################\n");
        printf("Total visibilities   : %d \n", 4*parameters.nrows*parameters.Nchan);
        printf("Degridding kernel time    : %0.4f seconds\n", ((float) end - start)/CLOCKS_PER_SEC);
        printf("Visibility throughput   : %.4e MVisi/s\n", (4*parameters.nrows*parameters.Nchan)/(((float) end - start)/CLOCKS_PER_SEC));
        printf("Number of threads created = %d threads\n", nb_block*parameters.full_support_function*parameters.full_support_function);
        printf("###################################\n");

    }




// /*************************************************************
//  *************************************************************
//  *********************  G2G **********************************
//  ************************************************************* 
//  *************************************************************/
//     void dgg_init_s2s(interpolation_parameters &parameters)
//     {
//         printf("I'm in dgg init\n");
//         //get_sky2sky_matrix_v1(parameters);
//         //get_sky2sky_matrix_v3(parameters);
//         get_sky2sky_matrix_v35(parameters);
//         printf("I'm in dgg init - Done \n");
//     }
     

//     void s2s_quad_pola(interpolation_parameters &parameters)
//     {

//         printf("#################################################################\n");
//         printf("#### In GPU interpolation_wrapper.cpp, S2S_Quad_pola #####\n");
//         printf("#################################################################\n");

//         typedef polarization_gridding_strat<quad_polarization> polarization_strategy;
//         typedef convolution_gridding_strat<polarization_strategy, conv_AA_2D> convolution_stategy;
//         printf("Choose conv_AA_2D convolution strategy \n");
//         typedef convolution_gridding_strat<polarization_strategy, conv_AA_2D> convolution_stategy;

//         clock_t  start, end;

//         /* As [Merry,2016], Visiblities are sorted by baselines.  
//            Each block compute the same number of visibilities to avoid unbalanced workload dur to compression.
//          */
//         size_t visi_per_block = 1024; // Let's set at 1024 for now


//         /* 
//            V1
//            The number of threads is defined by the size of the GCF.
//         */
//         //const dim3 blockSize(parameters.full_support_function*parameters.full_support_function, 1);
//         const dim3 blockSize(BLOCK_SIZE, 1); // conv function : 8x8

//         size_t nb_block = ceil((parameters.tot_size_coo)/visi_per_block);
//         const dim3 gridSize(nb_block,1);


//         printf("There is %d Threads per block\n", parameters.full_support_function*parameters.full_support_function);
//         printf("There is %d Blocks\n", nb_block);


//         /*
//            Set meta data to the device. sizeof(fft_grid_type)*parameters.Nx*parameters.Ny*parameters.no_grid_index*4*2
//         */
        

//         d_parameters.nrows = parameters.nrows;
//         d_parameters.Nchan = parameters.Nchan;
//         d_parameters.half_support_function = parameters.half_support_function;
//         d_parameters.full_support_function = parameters.full_support_function;
//         d_parameters.half_Nx = parameters.half_Nx;
//         d_parameters.half_Ny = parameters.half_Ny;
//         d_parameters.u_scale = parameters.u_scale;
//         d_parameters.v_scale = parameters.v_scale;
//         d_parameters.max_w   = parameters.max_w;
//         d_parameters.Nx = parameters.Nx;
//         d_parameters.Ny = parameters.Ny;
//         d_parameters.filter_size = parameters.filter_size;
//         //d_parameters.filter_AA_1D_size = parameters.filter_AA_1D_size;
//         d_parameters.filter_AA_2D_size = parameters.filter_AA_2D_size;
//         //d_parameters.filter_Prola_2D_size = parameters.filter_Prola_2D_size;
//         d_parameters.oversampling_factor = parameters.oversampling_factor;
//         d_parameters.tot_size_coo = parameters.tot_size_coo;
//         d_parameters.nb_grid_polarization = parameters.nb_grid_polarization;
//         //d_parameters.w_planes_tot_size = parameters.w_planes_tot_size;

//         for(size_t i=0; i<parameters.tot_size_coo; i++){
//             if(parameters.uvach_coo[i*4] > 1280){
//             printf("Ch idx = %f\n", parameters.uvach_coo[i*4]);}
//         }

//         // 
//         start = clock();
//         printf("Launch small kernel function\n");
//         s2s_template_shared_memory<convolution_stategy,polarization_strategy, BLOCK_SIZE><<<gridSize, blockSize>>>(d_parameters, visi_per_block);
//         cudaDeviceSynchronize();
//         end = clock();
        
//         checkCudaErrors(cudaGetLastError());

//         checkCudaErrors(cudaMemcpy(parameters.output_grid, d_parameters.output_grid, sizeof(fft_grid_type)*parameters.Nx*parameters.Ny*parameters.no_grid_index*parameters.nb_grid_polarization*2, cudaMemcpyDeviceToHost));
//         //checkCudaErrors(cudaMemcpy(parameters.gridding_conv_function, d_parameters.gridding_conv_function, sizeof(conv_function_type)*parameters.filter_size*parameters.filter_size, cudaMemcpyDeviceToHost));


//         cudaDeviceSynchronize();


//         /* DEBUG PRINT*/
//         printf("###################################\n");
//         printf("Total compressed visi   : %d \n", 4*parameters.tot_size_coo);
//         printf("Gridding kernel time    : %0.4f seconds\n", ((float) end - start)/CLOCKS_PER_SEC);
//         printf("Visibility throughput   : %.4e MVisi/s\n", 4*parameters.tot_size_coo/(((float) end - start)/CLOCKS_PER_SEC));
//         printf("Number of threads created = %d threads\n", nb_block*parameters.full_support_function*parameters.full_support_function);
//         printf("###################################\n");

//     }
}
