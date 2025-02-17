#include <cuda.h>
#include <cuda_runtime.h>
//#include <helper_cuda.h>


#include "interpolation_template.h"
#include "cuda_error.h"
#include "../../common/interpolation_wrapper.h"
#include "../../common/polarization_common.h"


extern "C"
{
    interpolation_parameters d_parameters;
    int test = 0;

    void init(interpolation_parameters &parameters)
    {
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

        }


        // Alloc all the necessary data to GPU
        checkCudaErrors(cudaMalloc((void**)&d_parameters.uvw_coordinates, sizeof(uvw_coordinates_type)*3*parameters.nrows));   // uvw coordinates
        checkCudaErrors(cudaMalloc((void**)&d_parameters.visibility_weight, sizeof(weight_type)*parameters.nb_vis_polarization*parameters.nrows)); // visiblity weights
        checkCudaErrors(cudaMalloc((void**)&d_parameters.visibilities, sizeof(complex_struct<visibility_type>)*parameters.Nchan*parameters.nb_vis_polarization*parameters.nrows));// visibilities
        checkCudaErrors(cudaMalloc((void**)&d_parameters.chan_wavelength, sizeof(freq_type)*parameters.Nchan));// chan_wavelength
        checkCudaErrors(cudaMalloc((void**)&d_parameters.gridding_conv_function, sizeof(conv_function_type)*parameters.filter_size));// AA conv kernel (No W, only float)
        //checkCudaErrors(cudaMalloc(// Oversampling factor
        // Size conv kernel
        // conv_norm_weight


        checkCudaErrors(cudaHostRegister(parameters.uvw_coordinates, sizeof(uvw_coordinates_type)*3*parameters.nrows, 0));
        checkCudaErrors(cudaHostRegister(parameters.visibility_weight, sizeof(weight_type)*parameters.nb_vis_polarization*parameters.nrows, 0));
        checkCudaErrors(cudaHostRegister(parameters.visibilities, sizeof(complex_struct<visibility_type>)*parameters.Nchan*parameters.nb_vis_polarization*parameters.nrows, 0));
        checkCudaErrors(cudaHostRegister(parameters.chan_wavelength, sizeof(freq_type)*parameters.Nchan, 0));
        checkCudaErrors(cudaHostRegister(parameters.gridding_conv_function, sizeof(conv_function_type)*parameters.filter_size, 0));


        checkCudaErrors(cudaMemcpyAsync(d_parameters.uvw_coordinates, parameters.uvw_coordinates, sizeof(uvw_coordinates_type)*3*parameters.nrows, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_parameters.visibility_weight, parameters.visibility_weight, sizeof(weight_type)*parameters.nb_vis_polarization*parameters.nrows, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_parameters.visibilities, parameters.visibilities, sizeof(complex_struct<visibility_type>)*parameters.Nchan*parameters.nb_vis_polarization*parameters.nrows, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_parameters.chan_wavelength, parameters.chan_wavelength, sizeof(freq_type)*parameters.Nchan, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpyAsync(d_parameters.gridding_conv_function, parameters.gridding_conv_function, sizeof(conv_function_type)*parameters.filter_size, cudaMemcpyHostToDevice));


        checkCudaErrors(cudaHostUnregister(parameters.uvw_coordinates));
        checkCudaErrors(cudaHostUnregister(parameters.visibility_weight));
        checkCudaErrors(cudaHostUnregister(parameters.visibilities));
        checkCudaErrors(cudaHostUnregister(parameters.chan_wavelength));
        checkCudaErrors(cudaHostUnregister(parameters.gridding_conv_function));

        cudaDeviceSynchronize();
        test = 5;

    }

    void free_params(interpolation_parameters &parameters)
    {
        cudaDeviceSynchronize();
        checkCudaErrors(cudaFree(d_parameters.uvw_coordinates));
        checkCudaErrors(cudaFree(d_parameters.visibility_weight));
        checkCudaErrors(cudaFree(d_parameters.visibilities));
        checkCudaErrors(cudaFree(d_parameters.chan_wavelength));
        checkCudaErrors(cudaFree(d_parameters.gridding_conv_function));
        printf("Test is %d \n", test);
    }

    void gridding_psf(interpolation_parameters &parameters)
    {
    }
    
    void gridding_single_pola(interpolation_parameters &parameters)
    {
    }
    
    void gridding_dual_pola(interpolation_parameters &parameters)
    {
    }
    
    void gridding_quad_pola(interpolation_parameters &parameters)
    {
    }


}
