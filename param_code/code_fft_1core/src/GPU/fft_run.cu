
#include "fft_run.h"

void fft_shift_complex_to_complex_actor(int GRID_SIZE, PRECISION2* uv_grid_in, Config *config, PRECISION2* uv_grid_out)
{
	MD5_Update(sizeof(PRECISION2) * GRID_SIZE * GRID_SIZE, uv_grid_in);

	PRECISION2 *d_uv_grid = NULL;

	int grid_square = GRID_SIZE * GRID_SIZE;
	CUDA_CHECK_RETURN(cudaMalloc(&d_uv_grid, sizeof(PRECISION2) * grid_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_uv_grid, uv_grid_in, sizeof(PRECISION2) * grid_square, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, GRID_SIZE);
	int num_blocks_per_dimension = (int) ceil((double) GRID_SIZE / max_threads_per_block_dimension);
	dim3 shift_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 shift_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

	printf("UPDATE >>> Shifting grid data for FFT...\n\n");
	// Perform 2D FFT shift
	fft_shift_complex_to_complex<<<shift_blocks, shift_threads>>>(d_uv_grid, GRID_SIZE);
	cudaDeviceSynchronize();


	CUDA_CHECK_RETURN(cudaMemcpy(uv_grid_out, d_uv_grid, sizeof(PRECISION2) * grid_square, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaFree(d_uv_grid));

    MD5_Update(sizeof(PRECISION2) * grid_square, uv_grid_out);
}


void CUFFT_EXECUTE_C2C_actor(int GRID_SIZE, PRECISION2* uv_grid_in, PRECISION2* uv_grid_out)
{
	PRECISION2 *d_uv_grid = NULL;
	cufftHandle *fft_plan;


	fft_plan = (cufftHandle*) calloc(1, sizeof(cufftHandle));
	CUFFT_SAFE_CALL(cufftPlan2d(fft_plan, GRID_SIZE, GRID_SIZE, CUFFT_C2C_PLAN));

	int grid_square = GRID_SIZE * GRID_SIZE;
	CUDA_CHECK_RETURN(cudaMalloc(&d_uv_grid, sizeof(PRECISION2) * grid_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_uv_grid, uv_grid_in, sizeof(PRECISION2) * grid_square, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();


	printf("UPDATE >>> Performing iFFT...\n\n");
	CUFFT_SAFE_CALL(CUFFT_EXECUTE_C2C(*fft_plan, d_uv_grid, d_uv_grid, CUFFT_INVERSE));
	cudaDeviceSynchronize();


	CUDA_CHECK_RETURN(cudaMemcpy(uv_grid_out, d_uv_grid, sizeof(PRECISION2) * grid_square, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();


	CUFFT_SAFE_CALL(cufftDestroy(*fft_plan));
	free(fft_plan);
	CUDA_CHECK_RETURN(cudaFree(d_uv_grid));

    MD5_Update(sizeof(PRECISION2) * grid_square, uv_grid_out);
}



void fft_shift_complex_to_real_actor(int GRID_SIZE, PRECISION2* uv_grid, Config *config, PRECISION* dirty_image)
{

	PRECISION2 *d_uv_grid = NULL;
	PRECISION *d_image = NULL;

	int grid_square = GRID_SIZE * GRID_SIZE;
    CUDA_CHECK_RETURN(cudaMalloc(&d_image, sizeof(PRECISION) * grid_square));
    CUDA_CHECK_RETURN(cudaMemset(d_image, 0, grid_square * sizeof(PRECISION)));
    cudaDeviceSynchronize();

	// int grid_square = GRID_SIZE * GRID_SIZE;
	CUDA_CHECK_RETURN(cudaMalloc(&d_uv_grid, sizeof(PRECISION2) * grid_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_uv_grid, uv_grid, sizeof(PRECISION2) * grid_square, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();


	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, GRID_SIZE);
	int num_blocks_per_dimension = (int) ceil((double) GRID_SIZE / max_threads_per_block_dimension);
	dim3 shift_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 shift_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

	printf("UPDATE >>> Shifting grid data back and converting to output image...\n\n");
	// Perform 2D FFT shift back
	fft_shift_complex_to_real<<<shift_blocks, shift_threads>>>(d_uv_grid, d_image, GRID_SIZE);
	cudaDeviceSynchronize();



	CUDA_CHECK_RETURN(cudaMemcpy(dirty_image, d_image, grid_square * sizeof(PRECISION), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaFree(d_uv_grid));
	CUDA_CHECK_RETURN(cudaFree(d_image));

	//printf("fft_shift_complex_to_real");
	MD5_Update(sizeof(PRECISION) * grid_square, dirty_image);
}

__global__ void fft_shift_complex_to_complex(PRECISION2 *grid, const int width)
{
    int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    int col_index = threadIdx.x + blockDim.x * blockIdx.x;
 
    if(row_index >= width || col_index >= width)
		return;
 
    int a = 1 - 2 * ((row_index + col_index) & 1);
    grid[row_index * width + col_index].x *= a;
    grid[row_index * width + col_index].y *= a;
}
//Note passing in output image - Future work do Complex to Real transform
__global__ void fft_shift_complex_to_real(PRECISION2 *grid, PRECISION *image, const int width)
{
	int row_index = threadIdx.y + blockDim.y * blockIdx.y;
	int col_index = threadIdx.x + blockDim.x * blockIdx.x;
 
	if(row_index >= width || col_index >= width)
		return;
		int index = row_index * width + col_index;

	int a = 1 - 2 * ((row_index + col_index) & 1);
	image[index] = grid[index].x * a;
}