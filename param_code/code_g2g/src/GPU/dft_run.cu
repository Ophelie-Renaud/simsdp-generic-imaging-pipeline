
#include "dft_run.h"



void dft_actor(int NUM_VISIBILITIES, int NUM_MAX_SOURCES,
			PRECISION3 *sources, PRECISION3 *vis_uvw_coords, int *num_sources, Config *config, PRECISION2 *visibilities)
{

	PRECISION3 *d_vis_uvw_coords = NULL;
	PRECISION3 *d_sources = NULL;
	PRECISION2 *d_visibilities = NULL;

#ifdef VERBOSE_MD5
	printf("sources MD5 \t\t: ");
	MD5_Update(sizeof(PRECISION3) * *num_sources, sources);

	printf("vis_uvw_coords MD5 \t\t: ");
	MD5_Update(sizeof(PRECISION3) * NUM_VISIBILITIES, vis_uvw_coords);

	printf("num_sources : \t\t\t%d\n", *num_sources);
#endif


	CUDA_CHECK_RETURN(cudaMalloc(&d_vis_uvw_coords, sizeof(PRECISION3) * NUM_VISIBILITIES));
	CUDA_CHECK_RETURN(cudaMemcpy(d_vis_uvw_coords, vis_uvw_coords, sizeof(PRECISION3) * NUM_VISIBILITIES, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_sources, sizeof(PRECISION3) * NUM_MAX_SOURCES));
	CUDA_CHECK_RETURN(cudaMemset(d_sources, 0, sizeof(PRECISION3) * NUM_MAX_SOURCES));
	// occurs only if has sources from previous major cycle
	if(*num_sources > 0)	
		CUDA_CHECK_RETURN(cudaMemcpy(d_sources, sources, sizeof(PRECISION3) * *num_sources, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_visibilities, sizeof(PRECISION2) * NUM_VISIBILITIES));
	CUDA_CHECK_RETURN(cudaMemset(d_visibilities, 0, sizeof(PRECISION2) * NUM_VISIBILITIES));
	cudaDeviceSynchronize();


	int max_threads_per_block = min(config->gpu_max_threads_per_block, NUM_VISIBILITIES);
	int num_blocks = (int) ceil((double) NUM_VISIBILITIES / max_threads_per_block);
	dim3 kernel_blocks(num_blocks, 1, 1);
	dim3 kernel_threads(max_threads_per_block, 1, 1);

	printf("UPDATE >>> Executing the Direct Fourier Transform algorithm...\n\n");
	printf("UPDATE >>> DFT distributed over %d blocks, consisting of %d threads...\n\n", num_blocks, max_threads_per_block);

	printf("num_sources : \t\t%d\n", *num_sources);

	direct_fourier_transform<<<kernel_blocks, kernel_threads>>>
	(
		d_vis_uvw_coords,
		d_visibilities,
		NUM_VISIBILITIES,
		d_sources,
		*num_sources
	);
	cudaDeviceSynchronize();



	CUDA_CHECK_RETURN(cudaMemcpy(visibilities, d_visibilities, sizeof(PRECISION2) * NUM_VISIBILITIES, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

#ifdef VERBOSE_MD5
    printf("d_visibilities MD5 \t\t: ");
    MD5_Update(sizeof(PRECISION2) * NUM_VISIBILITIES, visibilities);
#endif

	CUDA_CHECK_RETURN(cudaFree(d_visibilities));
	CUDA_CHECK_RETURN(cudaFree(d_sources));
	CUDA_CHECK_RETURN(cudaFree(d_vis_uvw_coords));
}




//execute direct fourier transform on GPU
__global__ void direct_fourier_transform(const PRECISION3 *vis_uvw, PRECISION2 *predicted_vis,
	const int vis_count, const PRECISION3 *sources, const int source_count)
{
	const int vis_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(vis_index >= vis_count)
		return;

	const PRECISION two_PI = PI + PI;
	const PRECISION3 vis = vis_uvw[vis_index];
	PRECISION3 src;
	PRECISION2 theta_complex = MAKE_PRECISION2(0.0, 0.0);
	PRECISION2 source_sum = MAKE_PRECISION2(0.0, 0.0);

	// For all sources
	for(int src_indx = 0; src_indx < source_count; ++src_indx)
	{	
		src = sources[src_indx];
		//Two formula below. uncomment if needed

		// square root formula (most accurate method)
		// 	PRECISION term = SQRT(1.0 - (src.x * src.x) - (src.y * src.y));
		// 	PRECISION image_correction = term;
		// 	PRECISION w_correction = term - 1.0;

		// approximation formula - faster but less accurate
		PRECISION term = 0.5 * ((src.x * src.x) + (src.y * src.y));
		PRECISION w_correction = -term;
		PRECISION image_correction = 1.0 - term;

		PRECISION src_correction = src.z / image_correction;
		PRECISION theta = (vis.x * src.x + vis.y * src.y + vis.z * w_correction) * two_PI;
		SINCOS(theta, &(theta_complex.y), &(theta_complex.x));
		source_sum.x += theta_complex.x * src_correction;
		source_sum.y += -theta_complex.y * src_correction;
	}
	predicted_vis[vis_index] = MAKE_PRECISION2(source_sum.x, source_sum.y);
}