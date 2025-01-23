
// Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "common.h"

void consume_loop_token(__attribute__((unused)) char *loop_token)
{
    return;
}

void loop_iterator(int NB_ITERATION, int *cycle_count)
{
    for (int i = 0; i < NB_ITERATION; i++)
        cycle_count[i] = i;
}

//void image_free(Device_Mem_Handles *device)
//{
//    printf("UPDATE >>> Freeing Image Memory memory...\n\n");
//    if(device->d_image != NULL)
//        CUDA_CHECK_RETURN(cudaFree(device->d_image));
//    device->d_image = NULL;
//}
//
//int exit_and_clean(const char *message, Host_Mem_Handles *host_mem, Device_Mem_Handles *device_mem)
//{
//    printf("ERROR >>> %s , exiting ... \n", message);
//    clean_up(host_mem, device_mem);
//    return EXIT_FAILURE;
//}
//
//void allocate_device_measured_vis(Config *config, Device_Mem_Handles *device)
//{
//    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_measured_vis), sizeof(PRECISION2) * config->num_visibilities));
//}
//
//void allocate_device_vis_coords(Config *config, Device_Mem_Handles *device)
//{
//    CUDA_CHECK_RETURN(cudaMalloc(&(device->d_vis_uvw_coords), sizeof(PRECISION3) * config->num_visibilities));
//}
//
//void free_device_vis_coords(Device_Mem_Handles *device)
//{
//    if((*device).d_vis_uvw_coords)
//        CUDA_CHECK_RETURN(cudaFree(device->d_vis_uvw_coords));
//    device->d_vis_uvw_coords = NULL;
//}
//void free_device_measured_vis(Device_Mem_Handles *device)
//{
//    if((*device).d_measured_vis)
//        CUDA_CHECK_RETURN(cudaFree(device->d_measured_vis));
//    device->d_measured_vis = NULL;
//}
//void free_device_predicted_vis(Device_Mem_Handles *device)
//{
//    if((*device).d_visibilities)
//        CUDA_CHECK_RETURN(cudaFree(device->d_visibilities));
//    device->d_visibilities = NULL;
//}



//void copy_measured_vis_to_device(Config *config, Host_Mem_Handles *host, Device_Mem_Handles *device)
//{
//    printf("UPDATE >>> Copying measured visibilities to device, number of visibilities: %d...\n\n",config->num_visibilities);
//    CUDA_CHECK_RETURN(cudaMemcpy(device->d_measured_vis, host->measured_vis, sizeof(PRECISION2) * config->num_visibilities,
//        cudaMemcpyHostToDevice));
//    cudaDeviceSynchronize();
//}

//void free_device_sources(Device_Mem_Handles *device)
//{
//    if(device->d_sources != NULL)
//        CUDA_CHECK_RETURN(cudaFree(device->d_sources));
//    device->d_sources           = NULL;
//}

//void clean_up(Host_Mem_Handles *host, Device_Mem_Handles *device)
//{
//    printf("UPDATE >>> Cleaning up all allocated host memory...\n");
//    if((*host).kernels)         free((*host).kernels);
//    if((*host).kernel_supports) free((*host).kernel_supports);
//    if((*host).dirty_image)     free((*host).dirty_image);
//    if((*host).residual_image)  free((*host).residual_image);
//    if((*host).vis_uvw_coords)  free((*host).vis_uvw_coords);
//    if((*host).visibilities)    free((*host).visibilities);
//    if((*host).h_psf)           free((*host).h_psf);
//    if((*host).h_sources)       free((*host).h_sources);
//    if((*host).measured_vis)    free((*host).measured_vis);
//    if((*host).prolate)         free((*host).prolate);
//    if((*host).receiver_pairs)  free((*host).receiver_pairs);
//    if((*host).h_gains)         free((*host).h_gains);
//
//    printf("UPDATE >>> DEALLOCATED THE MEMORY...\n");
//    (*host).kernels         = NULL;
//    (*host).kernel_supports = NULL;
//    (*host).dirty_image     = NULL;
//    (*host).residual_image  = NULL;
//    (*host).vis_uvw_coords  = NULL;
//    (*host).visibilities    = NULL;
//    (*host).h_psf           = NULL;
//    (*host).h_sources       = NULL;
//    (*host).measured_vis    = NULL;
//    (*host).prolate         = NULL;
//    (*host).receiver_pairs  = NULL;
//    (*host).h_gains         = NULL;
//
//    printf("UPDATE >>> Cleaning up all allocated device memory...\n");
//    if((*device).d_gains)           CUDA_CHECK_RETURN(cudaFree((*device).d_gains));
//    if((*device).d_kernels)         CUDA_CHECK_RETURN(cudaFree((*device).d_kernels));
//    if((*device).d_kernel_supports) CUDA_CHECK_RETURN(cudaFree((*device).d_kernel_supports));
//    if((*device).d_image)           CUDA_CHECK_RETURN(cudaFree((*device).d_image));
//    if((*device).d_uv_grid)         CUDA_CHECK_RETURN(cudaFree((*device).d_uv_grid));
//    if((*device).d_vis_uvw_coords)  CUDA_CHECK_RETURN(cudaFree((*device).d_vis_uvw_coords));
//    if((*device).d_visibilities)    CUDA_CHECK_RETURN(cudaFree((*device).d_visibilities));
//    if((*device).d_prolate)         CUDA_CHECK_RETURN(cudaFree((*device).d_prolate));
//    if((*device).d_sources)         CUDA_CHECK_RETURN(cudaFree((*device).d_sources));
//    if((*device).d_psf)             CUDA_CHECK_RETURN(cudaFree((*device).d_psf));
//    if((*device).d_max_locals)      CUDA_CHECK_RETURN(cudaFree((*device).d_max_locals));
//    if((*device).d_measured_vis)    CUDA_CHECK_RETURN(cudaFree((*device).d_measured_vis));
//    if((*device).d_receiver_pairs)  CUDA_CHECK_RETURN(cudaFree((*device).d_receiver_pairs));
//    if((*device).fft_plan)          free((*device).fft_plan);
//
//    (*device).d_gains           = NULL;
//    (*device).d_kernels         = NULL;
//    (*device).d_kernel_supports = NULL;
//    (*device).d_image           = NULL;
//    (*device).d_uv_grid         = NULL;
//    (*device).d_vis_uvw_coords  = NULL;
//    (*device).d_visibilities    = NULL;
//    (*device).d_prolate         = NULL;
//    (*device).d_sources         = NULL;
//    (*device).d_psf             = NULL;
//    (*device).d_max_locals      = NULL;
//    (*device).d_measured_vis    = NULL;
//    (*device).d_receiver_pairs  = NULL;
//    (*device).fft_plan          = NULL;
//
//    printf("UPDATE >>> Clean up Done\n");
//}

//void init_mem(Host_Mem_Handles *host, Device_Mem_Handles *device)
//{
//    (*host).kernels             = NULL;
//    (*host).kernel_supports     = NULL;
//    (*host).dirty_image         = NULL;
//    (*host).residual_image      = NULL;
//    (*host).vis_uvw_coords      = NULL;
//    (*host).visibilities        = NULL;
//    (*host).h_psf               = NULL;
//    (*host).h_sources           = NULL;
//    (*host).measured_vis        = NULL;
//    (*host).prolate             = NULL;
//    (*host).receiver_pairs      = NULL;
//    (*host).h_gains             = NULL;
//
//    (*device).d_gains           = NULL;
//    (*device).d_kernels         = NULL;
//    (*device).d_kernel_supports = NULL;
//    (*device).d_image           = NULL;
//    (*device).d_uv_grid         = NULL;
//    (*device).d_vis_uvw_coords  = NULL;
//    (*device).d_visibilities    = NULL;
//    (*device).d_prolate         = NULL;
//    (*device).d_sources         = NULL;
//    (*device).d_psf             = NULL;
//    (*device).d_max_locals      = NULL;
//    (*device).d_measured_vis    = NULL;
//    (*device).d_receiver_pairs  = NULL;
//    (*device).fft_plan          = NULL;
//}

void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;

	printf(">>> CUDA ERROR: %s returned %s at %s : %u ",statement, file, cudaGetErrorString(err), line);

	exit(EXIT_FAILURE);
}

void cufft_safe_call(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
		printf("CUFFT error in file '%s', line %d\nerror %d: %s\nterminating!\n",
			file, line, err, cuda_get_error_enum(err));
		cudaDeviceReset();
    }
}

const char* cuda_get_error_enum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
        	return "CUFFT_INCOMPLETE_PARAMETER_LIST";

    	case CUFFT_INVALID_DEVICE:
    		return "CUFFT_INVALID_DEVICE";

		case CUFFT_PARSE_ERROR:
			return "CUFFT_PARSE_ERROR";

		case CUFFT_NO_WORKSPACE:
			return "CUFFT_NO_WORKSPACE";

		case CUFFT_NOT_IMPLEMENTED:
			return "CUFFT_NOT_IMPLEMENTED";

		case CUFFT_LICENSE_ERROR:
			return "CUFFT_LICENSE_ERROR";

		case CUFFT_NOT_SUPPORTED:
			return "CUFFT_NOT_SUPPORTED";
    }

    return "<unknown>";
}