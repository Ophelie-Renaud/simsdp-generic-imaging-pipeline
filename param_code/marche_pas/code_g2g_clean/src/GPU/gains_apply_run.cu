
#include "gains_apply_run.h"

void reciprocal_transform_actor(int NUM_RECEIVERS,
			PRECISION2* gains_in, Config *config, PRECISION2* gains_out)
{

	PRECISION2* d_gains_in = NULL;
	PRECISION2* d_gains_out = NULL;

	// Allocating device input gains buffer
	CUDA_CHECK_RETURN(cudaMalloc(&d_gains_in, sizeof(PRECISION2) * NUM_RECEIVERS));
	CUDA_CHECK_RETURN(cudaMemcpy(d_gains_in, gains_in, sizeof(PRECISION2) * NUM_RECEIVERS, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// Allocating device output gains buffer
	CUDA_CHECK_RETURN(cudaMalloc(&d_gains_out, sizeof(PRECISION2) * NUM_RECEIVERS));
	CUDA_CHECK_RETURN(cudaMemset(d_gains_out, 0, sizeof(PRECISION2) * NUM_RECEIVERS));
	cudaDeviceSynchronize();


	int max_threads_per_block = min(config->gpu_max_threads_per_block, NUM_RECEIVERS);
	int num_blocks = (int) ceil((double) NUM_RECEIVERS / max_threads_per_block);
	dim3 kernel_blocks(num_blocks, 1, 1);
	dim3 kernel_threads(max_threads_per_block, 1, 1);

	//performs a reciprocal transform on the gains before applying to save on divisions
	reciprocal_transform<<<kernel_blocks, kernel_threads>>>(d_gains_out, d_gains_in, NUM_RECEIVERS);
	cudaDeviceSynchronize();


	CUDA_CHECK_RETURN(cudaMemcpy(gains_out, d_gains_out, sizeof(PRECISION2) * NUM_RECEIVERS, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaFree(d_gains_out));
	CUDA_CHECK_RETURN(cudaFree(d_gains_in));
}

void apply_gains_actor(int CALIBRATION, int NUM_RECEIVERS, int NUM_BASELINES, int NUM_VISIBILITIES,
			PRECISION2* measured_vis, PRECISION2* visibilities_in, PRECISION2* gains, int2* receiver_pairs, Config *config, PRECISION2* visibilities_out)
{
	printf("UPDATE >>> Applying gains... \n\n");

	PRECISION2* d_measured_vis = NULL;
	PRECISION2* d_visibilities = NULL;
	PRECISION2* d_gains = NULL;
	int2* d_receiver_pairs = NULL;

    printf("measured_vis\n");
    MD5_Update(sizeof(PRECISION2) * NUM_VISIBILITIES, measured_vis);

    printf("gains\n");
    MD5_Update(sizeof(PRECISION2) * NUM_RECEIVERS, gains);

	// Allocating device measured_vis buffer
	CUDA_CHECK_RETURN(cudaMalloc(&d_measured_vis, sizeof(PRECISION2) * NUM_VISIBILITIES));
	CUDA_CHECK_RETURN(cudaMemcpy(d_measured_vis, measured_vis, sizeof(PRECISION2) * NUM_VISIBILITIES, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// Allocating device input visibilities buffer
	CUDA_CHECK_RETURN(cudaMalloc(&d_visibilities, sizeof(PRECISION2) * NUM_VISIBILITIES));
	CUDA_CHECK_RETURN(cudaMemcpy(d_visibilities, visibilities_in, sizeof(PRECISION2) * NUM_VISIBILITIES, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// Allocating device gains buffer
	CUDA_CHECK_RETURN(cudaMalloc(&d_gains, sizeof(PRECISION2) * NUM_RECEIVERS));
	CUDA_CHECK_RETURN(cudaMemcpy(d_gains, gains, sizeof(PRECISION2) * NUM_RECEIVERS, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// Allocating device gains buffer
	CUDA_CHECK_RETURN(cudaMalloc(&d_receiver_pairs, sizeof(PRECISION2) * NUM_BASELINES));
	CUDA_CHECK_RETURN(cudaMemcpy(d_receiver_pairs, receiver_pairs, sizeof(int2) * NUM_BASELINES, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();



	int max_threads_per_block = min(config->gpu_max_threads_per_block, NUM_VISIBILITIES);
	int num_blocks = (int) ceil((double) NUM_VISIBILITIES / max_threads_per_block);
	dim3 gains_blocks(num_blocks, 1, 1);
	dim3 gains_threads(max_threads_per_block, 1, 1);

	if(!CALIBRATION)
	{
		//apply gain calibration to update gains between measured and predicted visibilities
		apply_gains_subtraction<<<gains_blocks, gains_threads>>>(
				d_measured_vis,
				d_visibilities,
				NUM_VISIBILITIES,
				d_gains,
				d_receiver_pairs,
				NUM_RECEIVERS,
				NUM_BASELINES
		);
	}
	else
	{
		apply_gains<<<gains_blocks, gains_threads>>>(
				d_measured_vis,
				d_visibilities,
				NUM_VISIBILITIES,
				d_gains,
				d_receiver_pairs,
				NUM_RECEIVERS,
				NUM_BASELINES
		);
	}
	cudaDeviceSynchronize();



	CUDA_CHECK_RETURN(cudaMemcpy(visibilities_out, d_visibilities, sizeof(PRECISION2) * NUM_VISIBILITIES, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

#ifdef VERBOSE_MD5
    printf("Sum d_visibilities\n");
    MD5_Update(sizeof(PRECISION2) * NUM_VISIBILITIES, visibilities_out);
#endif

	CUDA_CHECK_RETURN(cudaFree(d_receiver_pairs));
	CUDA_CHECK_RETURN(cudaFree(d_gains));
	CUDA_CHECK_RETURN(cudaFree(d_visibilities));
	CUDA_CHECK_RETURN(cudaFree(d_measured_vis));

}


__global__ void reciprocal_transform(PRECISION2 *gains_out, PRECISION2 *gains_in, const int num_recievers)
{
	const unsigned int receiver = blockIdx.x * blockDim.x + threadIdx.x;
	if(receiver >= num_recievers)
		return;

	gains_out[receiver] = complex_reciprocal_apply(gains_in[receiver]);
}

__global__ void apply_gains(PRECISION2 *measured_vis, PRECISION2 *predicted_vis, const int num_vis,
	const PRECISION2 *gains_recip, const int2 *receiver_pairs, const int num_recievers, const int num_baselines)
{
	const unsigned int vis_index = blockIdx.x * blockDim.x + threadIdx.x;
	if(vis_index >= num_vis)
		return;

	int baselineNumber =  vis_index % num_baselines;

	//THIS ASSUMES THAT WE HAVE GAINS AS THE RECIPROCAL
	PRECISION2 gains_a_recip = gains_recip[receiver_pairs[baselineNumber].x];
	PRECISION2 gains_b_recip_conj = complex_conjugate_apply(gains_recip[receiver_pairs[baselineNumber].y]);


	PRECISION2 gains_product_recip = complex_multiply_apply(gains_a_recip, gains_b_recip_conj);
	PRECISION2 measured_with_gains = complex_multiply_apply(measured_vis[vis_index], gains_product_recip);
	predicted_vis[vis_index] = measured_with_gains; 
}


//Apply gains on GPU between predicted and measured visibility values - USED FOR IMAGING
__global__ void apply_gains_subtraction(PRECISION2 *measured_vis, PRECISION2 *predicted_vis, const int num_vis,
	const PRECISION2 *gains_recip, const int2 *receiver_pairs, const int num_recievers, const int num_baselines)
{
	const unsigned int vis_index = blockIdx.x * blockDim.x + threadIdx.x;
	if(vis_index >= num_vis)
		return;

	int baselineNumber =  vis_index % num_baselines;

	//THIS ASSUMES THAT WE HAVE GAINS AS THE RECIPROCAL
	PRECISION2 gains_a_recip = gains_recip[receiver_pairs[baselineNumber].x];
	PRECISION2 gains_b_recip_conj = complex_conjugate_apply(gains_recip[receiver_pairs[baselineNumber].y]);


	PRECISION2 gains_product_recip = complex_multiply_apply(gains_a_recip, gains_b_recip_conj);
	PRECISION2 measured_with_gains = complex_multiply_apply(measured_vis[vis_index], gains_product_recip);
	predicted_vis[vis_index] = complex_subtract_apply(measured_with_gains, predicted_vis[vis_index]);
}

__device__ PRECISION2 complex_reciprocal_apply(const PRECISION2 z)
{   
	PRECISION real = z.x / (z.x * z.x + z.y * z.y); 
	PRECISION imag = z.y / (z.x * z.x + z.y * z.y); 
	return MAKE_PRECISION2(real, -imag); 
}

__device__ PRECISION2 complex_multiply_apply(const PRECISION2 z1, const PRECISION2 z2)
{
	return MAKE_PRECISION2(z1.x * z2.x - z1.y * z2.y, z1.x * z2.y + z1.y * z2.x);
}

__device__ PRECISION2 complex_conjugate_apply(const PRECISION2 z1)
{
	return MAKE_PRECISION2(z1.x, -z1.y);
}

__device__ PRECISION2 complex_subtract_apply(const PRECISION2 z1, const PRECISION2 z2)
{
	return MAKE_PRECISION2(z1.x - z2.x, z1.y - z2.y);
}