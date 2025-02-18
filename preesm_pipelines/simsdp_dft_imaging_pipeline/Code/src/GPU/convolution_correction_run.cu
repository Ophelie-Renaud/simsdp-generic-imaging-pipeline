
#include "convolution_correction_run.h"

void execute_convolution_correction_actor(int GRID_SIZE, PRECISION* dirty_image_in, PRECISION* prolate, Config *config, PRECISION* dirty_image_out)
{
    int grid_square = GRID_SIZE * GRID_SIZE;

#ifdef VERBOSE_MD5
    printf("prolate\n");
    MD5_Update(sizeof(PRECISION) * GRID_SIZE / 2, prolate);

    printf("dirty_image_in\n");
    MD5_Update(sizeof(PRECISION) * grid_square, dirty_image_in);
#endif

	PRECISION *d_image = NULL;
	PRECISION *d_prolate = NULL;


	// Bind prolate spheroidal to gpu
    CUDA_CHECK_RETURN(cudaMalloc(&d_prolate, sizeof(PRECISION) * GRID_SIZE / 2));
    CUDA_CHECK_RETURN(cudaMemcpy(d_prolate, prolate, sizeof(PRECISION) * GRID_SIZE / 2, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();


	CUDA_CHECK_RETURN(cudaMalloc(&d_image, sizeof(PRECISION) * grid_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_image, dirty_image_in, sizeof(PRECISION) * grid_square, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, GRID_SIZE);
	int num_blocks_per_dimension = (int) ceil((double) GRID_SIZE / max_threads_per_block_dimension);
	dim3 cc_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 cc_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

    printf("grid_size: %d\n", GRID_SIZE);

	execute_convolution_correction<<<cc_blocks, cc_threads>>>(d_image, d_prolate, GRID_SIZE);
	cudaDeviceSynchronize();



	CUDA_CHECK_RETURN(cudaMemcpy(dirty_image_out, d_image, sizeof(PRECISION) * grid_square, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

#ifdef VERBOSE_MD5
    printf("convolution_correction MD5\t: ");
    MD5_Update(sizeof(PRECISION) * grid_square, dirty_image_out);
#endif

//    for (int i = 0; i < 10; i++)
//    	printf("%lf\n", dirty_image_out[i]);

	CUDA_CHECK_RETURN(cudaFree(d_image));
	CUDA_CHECK_RETURN(cudaFree(d_prolate));

}

__global__ void execute_convolution_correction(PRECISION *image, const PRECISION *prolate, const int image_size)
{
	const int row_index = threadIdx.y + blockDim.y * blockIdx.y;
	const int col_index = threadIdx.x + blockDim.x * blockIdx.x;

	if(row_index >= image_size || col_index >= image_size)
		return;

	const int image_index = row_index * image_size + col_index;
	const int half_image_size = image_size / 2;

//	if (image_index == 0)
//		printf("prolate[%d]: %.15lf * prolate[%d]: %.15lf\n", abs(col_index - half_image_size), prolate[abs(col_index - half_image_size)], abs(row_index - half_image_size), prolate[abs(row_index - half_image_size)]);
//
//	if(abs(col_index - half_image_size) == 0 || abs(row_index - half_image_size) == 0)
//		printf("prolate[%d]: %.15lf * prolate[%d]: %.15lf\n", abs(col_index - half_image_size), prolate[abs(col_index - half_image_size)], abs(row_index - half_image_size), prolate[abs(row_index - half_image_size)]);

//	const PRECISION taper = prolate[abs(col_index - half_image_size)] * prolate[abs(row_index - half_image_size)];
	PRECISION taper = prolate[abs(col_index - half_image_size)] * prolate[abs(row_index - half_image_size)];

	if(abs(col_index - half_image_size) == 1229 || abs(row_index - half_image_size) == 1229)
		taper = 0.0;

//	if (image_index == 0) printf("%.15lf\n", taper);
//
//	if (image_index == 0) printf("image[%d]: %lf\n", image_index, image[image_index]);

	image[image_index] = (ABS(taper) > (1E-10)) ? image[image_index] / taper  : 0.0;

//	if (image_index == 0) printf("image[%d]: %lf\n", image_index, image[image_index]);
}