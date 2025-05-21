
#include "deconvolution_run.h"

__device__ bool d_exit_early = false;
__device__ unsigned int d_source_counter = 0;

void deconvolution_run(int GRID_SIZE, int CALIBRATION, int NUMBER_MINOR_CYCLES_CAL, int NUMBER_MINOR_CYCLES_IMG, int NUM_MAX_SOURCES, int NUM_MAJOR_CYCLES,
			PRECISION* dirty_image_in, PRECISION* psf_in, PRECISION3* sources_in, int* num_sources_in, Config* config, int* num_sources_out, PRECISION3* sources_out, PRECISION* residual_image)
{
	PRECISION* d_image = NULL;
	PRECISION* d_psf = NULL;
	PRECISION3* d_sources = NULL;

	PRECISION3* h_max_locals = NULL;
	//PRECISION* h_temp_image = NULL;

	int num_sources = *num_sources_in;

	// int number_minor_cycles = (CALIBRATION) ? NUMBER_MINOR_CYCLES_CAL : NUMBER_MINOR_CYCLES_IMG;

	int grid_square = GRID_SIZE * GRID_SIZE;
	CUDA_CHECK_RETURN(cudaMalloc(&d_psf, sizeof(PRECISION) * grid_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_psf, psf_in, sizeof(PRECISION) * grid_square, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_image, sizeof(PRECISION) * grid_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_image, dirty_image_in, sizeof(PRECISION) * grid_square,cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	PRECISION3 *max_locals = NULL;
	CUDA_CHECK_RETURN(cudaMalloc(&max_locals, sizeof(PRECISION3) * GRID_SIZE));
	CUDA_CHECK_RETURN(cudaMemset(max_locals, 0, sizeof(PRECISION3) * GRID_SIZE));
	cudaDeviceSynchronize();


	printf("UPDATE >>> Allocating Device Source Buffer of size %d...\n\n", NUM_MAX_SOURCES);
	CUDA_CHECK_RETURN(cudaMalloc(&d_sources, sizeof(PRECISION3) * NUM_MAX_SOURCES));
	CUDA_CHECK_RETURN(cudaMemset(d_sources, 0, sizeof(PRECISION3) * NUM_MAX_SOURCES));
	if(num_sources > 0) // occurs only if has sources from previous major cycle
	{
		CUDA_CHECK_RETURN(cudaMemcpy(d_sources, sources_in, sizeof(PRECISION3) * num_sources, cudaMemcpyHostToDevice));
	}
	cudaDeviceSynchronize();

	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, GRID_SIZE);
	int num_blocks_per_dimension = (int) ceil((double) GRID_SIZE / max_threads_per_block_dimension);
	dim3 scale_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 scale_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);
	//Scale dirty image by psf use reciprical next time???
	scale_dirty_image_by_psf<<<scale_blocks, scale_threads>>>(d_image, d_psf, config->psf_max_value, GRID_SIZE);
	cudaDeviceSynchronize();

	PRECISION* d_image_scale_dirty = (PRECISION*) malloc(sizeof(PRECISION) * GRID_SIZE * GRID_SIZE);
	cudaMemcpy(d_image_scale_dirty, d_image, sizeof(PRECISION) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("scale_dirty_image_by_psf MD5\t: ");
	MD5_Update(sizeof(PRECISION) * GRID_SIZE * GRID_SIZE, d_image_scale_dirty);
	free(d_image_scale_dirty);




	// row reduction configuration
	int max_threads_per_block = min(config->gpu_max_threads_per_block, GRID_SIZE);
	int num_blocks = (int) ceil((double) GRID_SIZE / max_threads_per_block);
	dim3 reduction_blocks(num_blocks, 1, 1);
	dim3 reduction_threads(config->gpu_max_threads_per_block, 1, 1);

	// PSF subtraction configuration
	int max_psf_threads_per_block_dim = min(config->gpu_max_threads_per_block_dimension, GRID_SIZE);
	int num_blocks_psf = (int) ceil((double) GRID_SIZE / max_psf_threads_per_block_dim);
	dim3 psf_blocks(num_blocks_psf, num_blocks_psf, 1);
	dim3 psf_threads(max_psf_threads_per_block_dim, max_psf_threads_per_block_dim, 1);

	int cycle_number = 0;
	bool exit_early = false;

	if(CALIBRATION)
		num_sources = 0;

	
	// Reset exit early clause in case of multiple major cycles
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_exit_early, &exit_early, sizeof(bool), 0, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_source_counter, &(num_sources), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	//convert existing sources to grid coords
	if(num_sources > 0)
	{	
		printf("UPDATE >>> Performing grid conversion on previously found Source coordinates...\n\n");

		int max_threads_per_block_conversion = min(config->gpu_max_threads_per_block, num_sources);
		int num_blocks_conversion = (int) ceil((double) num_sources / max_threads_per_block_conversion);
		dim3 conversion_blocks(num_blocks_conversion, 1, 1);
		dim3 conversion_threads(config->gpu_max_threads_per_block, 1, 1);

		image_to_grid_coords_conversion<<<conversion_blocks, conversion_threads>>>(d_sources, config->cell_size, GRID_SIZE / 2, num_sources);
		cudaDeviceSynchronize();

		PRECISION3* h_sources = (PRECISION3*) malloc(sizeof(PRECISION3) * num_sources);
		cudaMemcpy(h_sources, d_sources, sizeof(PRECISION3) * num_sources, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		printf("image_to_grid_coords_conversion MD5\t: ");
		MD5_Update(sizeof(PRECISION3) * num_sources, h_sources);
		free(h_sources);
	}

	int number_minor_cycles = (CALIBRATION) ? NUMBER_MINOR_CYCLES_CAL : NUMBER_MINOR_CYCLES_IMG;
	printf("UPDATE >>> Performing deconvolution, up to %d minor cycles...\n\n",number_minor_cycles);

	double weak_source_percent = (CALIBRATION) ? config->weak_source_percent_gc : config->weak_source_percent_img;
	while(cycle_number < number_minor_cycles)
	{
		if(cycle_number % 10 == 0)
			printf("UPDATE >>> Performing minor cycle number: %u...\n\n", cycle_number);

		// Find local row maximum via reduction
		find_max_source_row_reduction<<<reduction_blocks, reduction_threads>>>
			(d_image, max_locals, GRID_SIZE);
		cudaDeviceSynchronize();

//		if (!CALIBRATION)
//		{
//			h_max_locals = (PRECISION3 *) malloc(sizeof(PRECISION3) * GRID_SIZE);
//			cudaMemcpy(h_max_locals, max_locals, sizeof(PRECISION3) * GRID_SIZE, cudaMemcpyDeviceToHost);
//			cudaDeviceSynchronize();
//			printf("find_max_source_row_reduction MD5\t: ");
//			MD5_Update(sizeof(PRECISION3) * GRID_SIZE, h_max_locals);
//			free(h_max_locals);
//		}

		// Find final image maximum via column reduction (local maximums array)
		find_max_source_col_reduction<<<1, 1>>>
			(d_sources, max_locals, cycle_number, GRID_SIZE, config->loop_gain, 
		 		weak_source_percent, config->noise_factor);
		cudaDeviceSynchronize();

//		if (!CALIBRATION)
//		{
//			cudaMemcpyFromSymbol(&num_sources, d_source_counter, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
//			cudaDeviceSynchronize();
//
//			printf("num_sources: %d\n", num_sources);
//
//			h_max_locals = NULL;
//			h_max_locals = (PRECISION3 *) malloc(sizeof(PRECISION3) * num_sources);
//			if (h_max_locals == NULL)
//				printf("ALLOC FAILED\n");
//
//			cudaMemcpy(h_max_locals, max_locals, sizeof(PRECISION3) * num_sources, cudaMemcpyDeviceToHost);
//			cudaDeviceSynchronize();
//			printf("source MD5\t\t\t: ");
//			MD5_Update(sizeof(PRECISION3) * num_sources, h_max_locals);
//			free(h_max_locals);
//		}

		subtract_psf_from_image<<<psf_blocks, psf_threads>>>
				(d_image, d_sources, d_psf, cycle_number, GRID_SIZE, config->loop_gain);
		cudaDeviceSynchronize();


//		if (!CALIBRATION)
//		{
//			h_temp_image = (PRECISION *) malloc(sizeof(PRECISION) * GRID_SIZE * GRID_SIZE);
//			cudaMemcpy(h_temp_image, d_image, sizeof(PRECISION) * GRID_SIZE * GRID_SIZE, cudaMemcpyDeviceToHost);
//			cudaDeviceSynchronize();
//			printf("scale_dirty_image_by_psf MD5\t: ");
//			MD5_Update(sizeof(PRECISION) * GRID_SIZE * GRID_SIZE, h_temp_image);
//			free(h_temp_image);
//		}


		compress_sources<<<1, 1>>>(d_sources);
		cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&exit_early, d_exit_early, sizeof(bool), 0, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

		if(exit_early)
		{
			printf(">>> UPDATE: Terminating minor cycles as now just cleaning noise, cycle number %u...\n\n", cycle_number);
			break;
		}

		cycle_number++;
	}

	// Determine how many compressed sources were found
	cudaMemcpyFromSymbol(&num_sources, d_source_counter, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	if(num_sources > 0)
	{
		printf("UPDATE >>> Performing conversion on Source coordinates...\n\n");

		int max_threads_per_block_conversion = min(config->gpu_max_threads_per_block, num_sources);
		int num_blocks_conversion = (int) ceil((double) num_sources / max_threads_per_block_conversion);
		dim3 conversion_blocks(num_blocks_conversion, 1, 1);
		dim3 conversion_threads(config->gpu_max_threads_per_block, 1, 1);

		grid_to_image_coords_conversion<<<conversion_blocks, conversion_threads>>>(d_sources, config->cell_size,
			GRID_SIZE / 2, num_sources);
		cudaDeviceSynchronize();
	}

    if(max_locals != NULL) 
    	CUDA_CHECK_RETURN(cudaFree(max_locals));
    max_locals = NULL;

    *num_sources_out = num_sources;

	CUDA_CHECK_RETURN(cudaMemcpy(residual_image, d_image, grid_square * sizeof(PRECISION), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy(sources_out, d_sources, sizeof(PRECISION3) * num_sources, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

#ifdef VERBOSE_MD5
	printf("residual_image\t\t: ");
	MD5_Update(sizeof(PRECISION) * grid_square, residual_image);

	printf("sources_out\t\t: ");
	MD5_Update(sizeof(PRECISION3) * num_sources, sources_out);
#endif

	free(h_max_locals);

	CUDA_CHECK_RETURN(cudaFree(d_psf));
	CUDA_CHECK_RETURN(cudaFree(d_image));
	CUDA_CHECK_RETURN(cudaFree(d_sources));


}

void scale_dirty_image_by_psf_actor(int GRID_SIZE,
			PRECISION* dirty_image_in, PRECISION* psf, Config *config, PRECISION* dirty_image_out)
{
	PRECISION* d_image = NULL;
	PRECISION* d_psf = NULL;

	int grid_square = GRID_SIZE * GRID_SIZE;

	CUDA_CHECK_RETURN(cudaMalloc(&d_psf, sizeof(PRECISION) * grid_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_psf, psf, sizeof(PRECISION) * grid_square, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_image, sizeof(PRECISION) * grid_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_image, dirty_image_in, sizeof(PRECISION) * grid_square,cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();


	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, GRID_SIZE);
	int num_blocks_per_dimension = (int) ceil((double) GRID_SIZE / max_threads_per_block_dimension);
	dim3 scale_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 scale_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);
	//Scale dirty image by psf use reciprical next time???
	scale_dirty_image_by_psf<<<scale_blocks, scale_threads>>>(d_image, d_psf, config->psf_max_value, GRID_SIZE);
	cudaDeviceSynchronize();


	CUDA_CHECK_RETURN(cudaMemcpy(dirty_image_out, d_image, grid_square * sizeof(PRECISION), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaFree(d_psf));
	CUDA_CHECK_RETURN(cudaFree(d_image));
}


void image_to_grid_coords_conversion_actor(int GRID_SIZE, int NUM_MAX_SOURCES,
			PRECISION3 *sources_in, int *num_sources, Config *config, PRECISION3 *sources_out)
{

	if(*num_sources > 0)
	{	

		PRECISION3* d_sources = NULL;

		CUDA_CHECK_RETURN(cudaMalloc(&d_sources, sizeof(PRECISION3) * NUM_MAX_SOURCES));
		CUDA_CHECK_RETURN(cudaMemset(d_sources, 0, sizeof(PRECISION3) * NUM_MAX_SOURCES));
		// occurs only if has sources from previous major cycle
		if(*num_sources > 0)	
			CUDA_CHECK_RETURN(cudaMemcpy(d_sources, sources_in, sizeof(PRECISION3) * *num_sources, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();



		printf("UPDATE >>> Performing grid conversion on previously found Source coordinates...\n\n");

		int max_threads_per_block_conversion = min(config->gpu_max_threads_per_block, *num_sources);
		int num_blocks_conversion = (int) ceil((double) *num_sources / max_threads_per_block_conversion);
		dim3 conversion_blocks(num_blocks_conversion, 1, 1);
		dim3 conversion_threads(config->gpu_max_threads_per_block, 1, 1);

		image_to_grid_coords_conversion<<<conversion_blocks, conversion_threads>>>(d_sources, config->cell_size, GRID_SIZE / 2, *num_sources);
		cudaDeviceSynchronize();


		CUDA_CHECK_RETURN(cudaMemcpy(sources_out, d_sources, sizeof(PRECISION3) * *num_sources, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

		CUDA_CHECK_RETURN(cudaFree(d_sources));
	}
	else
	{
		memcpy(sources_out, sources_in, sizeof(PRECISION3) * NUM_MAX_SOURCES);
	}
}


void reset_num_sources_actor(int CALIBRATION, IN int* num_sources_in, OUT int* num_sources_out)
{
	if (CALIBRATION)
		*num_sources_out = 0;
	else
		*num_sources_out = *num_sources_in;

	// Reset exit early clause in case of multiple major cycles
	// CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_exit_early, &exit_early, sizeof(bool), 0, cudaMemcpyHostToDevice));
	// CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_source_counter, &(config->num_sources), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
	// cudaDeviceSynchronize();
}


void find_max_source_row_reduction_actor(int GRID_SIZE,
			PRECISION *dirty_image_in, int *loop_token, Config *config, PRECISION3 *max_locals)
{
	PRECISION* d_dirty_image_in = NULL;
	PRECISION3* d_max_locals = NULL;


	int grid_square = GRID_SIZE * GRID_SIZE;
	CUDA_CHECK_RETURN(cudaMalloc(&d_dirty_image_in, sizeof(PRECISION) * grid_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_dirty_image_in, dirty_image_in, grid_square * sizeof(PRECISION), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_max_locals, sizeof(PRECISION3) * GRID_SIZE));
	CUDA_CHECK_RETURN(cudaMemset(d_max_locals, 0, sizeof(PRECISION3) * GRID_SIZE));
	cudaDeviceSynchronize();


	// row reduction configuration
	int max_threads_per_block = min(config->gpu_max_threads_per_block, GRID_SIZE);
	int num_blocks = (int) ceil((double) GRID_SIZE / max_threads_per_block);
	dim3 reduction_blocks(num_blocks, 1, 1);
	dim3 reduction_threads(config->gpu_max_threads_per_block, 1, 1);



	find_max_source_row_reduction<<<reduction_blocks, reduction_threads>>>
			(d_dirty_image_in, d_max_locals, GRID_SIZE);
		cudaDeviceSynchronize();


	CUDA_CHECK_RETURN(cudaMemcpy(max_locals, d_max_locals, GRID_SIZE * sizeof(PRECISION3), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaFree(d_max_locals));
	CUDA_CHECK_RETURN(cudaFree(d_dirty_image_in));
}

void find_max_source_col_reduction_actor(int GRID_SIZE, int NUM_MAX_SOURCES, int CALIBRATION,
			PRECISION3 *sources_in, int *num_sources_in, PRECISION3 *max_locals, int *cycle_number, Config *config, PRECISION3 *sources_out, int *num_sources_out)
{
	PRECISION3* d_max_locals = NULL;
	PRECISION3* d_sources = NULL;

	int num_sources = *num_sources_in;

	printf("UPDATE >>> Allocating Device Source Buffer of size %d...\n\n", NUM_MAX_SOURCES);
	CUDA_CHECK_RETURN(cudaMalloc(&d_sources, sizeof(PRECISION3) * NUM_MAX_SOURCES));
	CUDA_CHECK_RETURN(cudaMemset(d_sources, 0, sizeof(PRECISION3) * NUM_MAX_SOURCES));

	if(num_sources > 0) // occurs only if has sources from previous major cycle
	{
		CUDA_CHECK_RETURN(cudaMemcpy(d_sources, sources_in, sizeof(PRECISION3) * num_sources, cudaMemcpyHostToDevice));
	}
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_max_locals, sizeof(PRECISION3) * GRID_SIZE));
	CUDA_CHECK_RETURN(cudaMemcpy(d_max_locals, max_locals, GRID_SIZE * sizeof(PRECISION3), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_source_counter, &num_sources, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();


	double weak_source_percent = (CALIBRATION) ? config->weak_source_percent_gc : config->weak_source_percent_img;


	find_max_source_col_reduction<<<1, 1>>>
		(d_sources, max_locals, *cycle_number, GRID_SIZE, config->loop_gain, weak_source_percent, config->noise_factor);
	cudaDeviceSynchronize();


	cudaMemcpyFromSymbol(&num_sources, d_source_counter, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy(sources_out, d_sources, sizeof(PRECISION3) * num_sources, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	*num_sources_out = num_sources;

	CUDA_CHECK_RETURN(cudaFree(d_sources));
	CUDA_CHECK_RETURN(cudaFree(d_max_locals));
}

void grid_to_image_coords_conversion_actor(int GRID_SIZE, int NUM_MAX_SOURCES,
			PRECISION3 *sources_in, int *num_sources, Config *config, PRECISION3 *sources_out)
{

	if(*num_sources > 0)
	{

		PRECISION3* d_sources = NULL;

		printf("UPDATE >>> Allocating Device Source Buffer of size %d...\n\n", NUM_MAX_SOURCES);
		CUDA_CHECK_RETURN(cudaMalloc(&d_sources, sizeof(PRECISION3) * NUM_MAX_SOURCES));
		CUDA_CHECK_RETURN(cudaMemset(d_sources, 0, sizeof(PRECISION3) * NUM_MAX_SOURCES));
		CUDA_CHECK_RETURN(cudaMemcpy(d_sources, sources_in, sizeof(PRECISION3) * *num_sources, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();



		printf("UPDATE >>> Performing conversion on Source coordinates...\n\n");

		int max_threads_per_block_conversion = min(config->gpu_max_threads_per_block, *num_sources);
		int num_blocks_conversion = (int) ceil((double) *num_sources / max_threads_per_block_conversion);
		dim3 conversion_blocks(num_blocks_conversion, 1, 1);
		dim3 conversion_threads(config->gpu_max_threads_per_block, 1, 1);

		grid_to_image_coords_conversion<<<conversion_blocks, conversion_threads>>>(d_sources, config->cell_size,
			GRID_SIZE / 2, *num_sources);
		cudaDeviceSynchronize();



		CUDA_CHECK_RETURN(cudaMemcpy(sources_out, d_sources, sizeof(PRECISION3) * *num_sources, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
	}
	else
	{
		memcpy(sources_out, sources_in, sizeof(PRECISION3) * NUM_MAX_SOURCES);
	}

}

void subtract_psf_from_image_actor(int GRID_SIZE, int NUM_MAX_SOURCES,
			PRECISION *dirty_image_in, PRECISION3 *sources, PRECISION* psf, int *cycle_number, int *num_sources, Config *config, PRECISION *dirty_image_out)
{
	int grid_square = GRID_SIZE * GRID_SIZE;

	if (*num_sources > 0)
	{
		PRECISION* d_image = NULL;
		PRECISION* d_psf = NULL;
		PRECISION3* d_sources = NULL;


		CUDA_CHECK_RETURN(cudaMalloc(&d_psf, sizeof(PRECISION) * grid_square));
		CUDA_CHECK_RETURN(cudaMemcpy(d_psf, psf, sizeof(PRECISION) * grid_square, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		CUDA_CHECK_RETURN(cudaMalloc(&d_image, sizeof(PRECISION) * grid_square));
		CUDA_CHECK_RETURN(cudaMemcpy(d_image, dirty_image_in, sizeof(PRECISION) * grid_square, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		printf("UPDATE >>> Allocating Device Source Buffer of size %d...\n\n", NUM_MAX_SOURCES);
		CUDA_CHECK_RETURN(cudaMalloc(&d_sources, sizeof(PRECISION3) * *num_sources));
		CUDA_CHECK_RETURN(cudaMemcpy(d_sources, sources, sizeof(PRECISION3) * *num_sources, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();


				// PSF subtraction configuration
		int max_psf_threads_per_block_dim = min(config->gpu_max_threads_per_block_dimension, GRID_SIZE);
		int num_blocks_psf = (int) ceil((double) GRID_SIZE / max_psf_threads_per_block_dim);
		dim3 psf_blocks(num_blocks_psf, num_blocks_psf, 1);
		dim3 psf_threads(max_psf_threads_per_block_dim, max_psf_threads_per_block_dim, 1);


		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_source_counter, num_sources, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		subtract_psf_from_image<<<psf_blocks, psf_threads>>>
				(d_image, d_sources, d_psf, *cycle_number, GRID_SIZE, config->loop_gain);
		cudaDeviceSynchronize();



		CUDA_CHECK_RETURN(cudaMemcpy(dirty_image_out, d_image, sizeof(PRECISION) * grid_square, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();


	}
	else
	{
		memcpy(dirty_image_out, dirty_image_in, sizeof(PRECISION) * grid_square);
	}
}

void compress_sources_actor(int NUM_MAX_SOURCES,
			IN PRECISION3 *sources_in, IN int *num_sources_in, OUT PRECISION3 *sources_out, OUT int *num_sources_out)
{
	int num_sources = *num_sources_in;

	if (num_sources > 0)
	{
		PRECISION3* d_sources = NULL;

		printf("UPDATE >>> Allocating Device Source Buffer of size %d...\n\n", NUM_MAX_SOURCES);
		CUDA_CHECK_RETURN(cudaMalloc(&d_sources, sizeof(PRECISION3) * num_sources));
		CUDA_CHECK_RETURN(cudaMemcpy(d_sources, sources_in, sizeof(PRECISION3) * num_sources, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		CUDA_CHECK_RETURN(cudaMemcpyToSymbol(d_source_counter, &num_sources, sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();


		compress_sources<<<1, 1>>>(d_sources);
		cudaDeviceSynchronize();


		CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(&num_sources, d_source_counter, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

		CUDA_CHECK_RETURN(cudaMemcpy(sources_out, d_sources, sizeof(PRECISION3) * num_sources, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

	}
	else
	{	
		// Probably useless
		memcpy(sources_out, sources_in, sizeof(PRECISION3) * NUM_MAX_SOURCES);
	}

	*num_sources_out = num_sources;
}




__global__ void scale_dirty_image_by_psf(PRECISION *image, PRECISION *psf, PRECISION psf_max, const int grid_size)
{
	int col_index = blockIdx.x*blockDim.x + threadIdx.x;
	int row_index = blockIdx.y*blockDim.y + threadIdx.y;

	if(col_index >= grid_size || row_index >= grid_size)
		return;

	image[row_index * grid_size + col_index] /= psf_max;
}


__global__ void image_to_grid_coords_conversion(PRECISION3 *sources, const PRECISION cell_size, const int half_grid_size, const int source_count)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	if(index >= source_count)
		return;

	sources[index].x = ROUND((sources[index].x / cell_size) + half_grid_size); 
	sources[index].y = ROUND((sources[index].y / cell_size) + half_grid_size);
}


__global__ void find_max_source_row_reduction(const PRECISION *image, PRECISION3 *local_max, const int grid_size)
{
	unsigned int row_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(row_index >= grid_size)
		return;

	// l, m, intensity 
	// just going to borrow the "m" or y coordinate and use to find the average in this row.
	//PRECISION3 max = MAKE_PRECISION3(0.0, (double) row_index, image[row_index * grid_size]);
	PRECISION3 max = MAKE_PRECISION3(0.0, ABS(image[row_index * grid_size]), image[row_index * grid_size]);

	for(int col_index = 1; col_index < grid_size; ++col_index)
	{
		PRECISION current = image[row_index * grid_size + col_index];
		max.y += ABS(current);
		if(ABS(current) > ABS(max.z))
		{
			// update m and intensity
			max.x = (PRECISION) col_index;
			max.z = current;
		}
	}
	
	local_max[row_index] = max;
}

__global__ void find_max_source_col_reduction(PRECISION3 *sources, const PRECISION3 *local_max, const int cycle_number,
		const int grid_size, const double loop_gain, const double weak_source_percent, const double noise_factor)
{
	unsigned int col_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(col_index >= 1) // only single threaded
		return;

	//obtain max from row and col and clear the y (row) coordinate.
	PRECISION3 max = local_max[0];
	PRECISION running_avg = local_max[0].y;
	max.y = 0.0;

	PRECISION3 current;
	
	for(int index = 1; index < grid_size; ++index)
	{
		current = local_max[index];
		running_avg += current.y;		
		current.y = index;

		if(ABS(current.z) > ABS(max.z))
			max = current;
	}

	running_avg /= (grid_size * grid_size);
	max.z *= loop_gain;
	
	// determine whether we drop out and ignore this source
	
	bool extracting_noise = ABS(max.z) < noise_factor * running_avg * loop_gain;
	bool weak_source = ABS(max.z) < (ABS(sources[0].z) * weak_source_percent);
	d_exit_early = extracting_noise || weak_source;

	if(d_exit_early)	
		return;	

	// source was reasonable, so we keep it
	sources[d_source_counter] = max;
	++d_source_counter;
}

__global__ void grid_to_image_coords_conversion(PRECISION3 *sources, const PRECISION cell_size, const int half_grid_size, const int source_count)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	if(index >= source_count)
		return;

	sources[index].x = (sources[index].x - half_grid_size) * cell_size;
	sources[index].y = (sources[index].y - half_grid_size) * cell_size;
}

#if 1
__global__ void subtract_psf_from_image(PRECISION *image, PRECISION3 *sources, const PRECISION *psf, 
		const int cycle_number, const int grid_size, const PRECISION loop_gain)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	// thread out of bounds
	if(idx >= grid_size || idy >= grid_size || d_source_counter == 0)
		return;

	const int half_grid_size = grid_size / 2;

	// Determine image coordinates relative to source location
	int2 image_coord = make_int2(
		sources[d_source_counter-1].x - half_grid_size + idx,
		sources[d_source_counter-1].y - half_grid_size + idy
	);
	
	// image coordinates fall out of bounds
	if(image_coord.x < 0 || image_coord.x >= grid_size || image_coord.y < 0 || image_coord.y >= grid_size)
		return;

	// Get required psf sample for subtraction
	const PRECISION psf_weight = psf[idy * grid_size + idx];

	// Subtract shifted psf sample from image
	image[image_coord.y * grid_size + image_coord.x] -= psf_weight  * sources[d_source_counter-1].z;
}
#else

__global__ void subtract_psf_from_image(PRECISION *image, PRECISION3 *sources, const PRECISION *psf,
										const int cycle_number, const int grid_size, const PRECISION loop_gain) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	// thread out of bounds
	if (idx >= grid_size || idy >= grid_size || d_source_counter == 0)
		return;

	const int half_grid_size = grid_size / 2;

	const int half_psf_grid_size = 2048 / 2;

	if (abs(idx - sources[d_source_counter - 1].x) >= half_psf_grid_size ||
		abs(idy - sources[d_source_counter - 1].y) >= half_psf_grid_size)
		return;

	const PRECISION intensity = sources[d_source_counter - 1].z;

	// Determine image coordinates relative to source location
	int2 psf_coord = make_int2(
			(half_psf_grid_size - 1) - (sources[d_source_counter - 1].x - idx) + 1,
			(half_psf_grid_size - 1) - (sources[d_source_counter - 1].y - idy) + 1
	);
	// Adding a 1 offset as 1st row and col of sources psf file is full of 0

	// image coordinates fall out of bounds
	//	if(image_coord.x < 0 || image_coord.x >= grid_size || image_coord.y < 0 || image_coord.y >= grid_size)
	//		return;

	// Get required psf sample for subtraction
	const PRECISION psf_weight = psf[psf_coord.y * half_grid_size + psf_coord.x];


	// Subtract shifted psf sample from image
	image[idy * grid_size + idx] -= psf_weight * sources[d_source_counter - 1].z;
}
#endif

__global__ void compress_sources(PRECISION3 *sources)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index >= 1  || d_source_counter == 0) // only single threaded
		return;

	PRECISION3 last_source = sources[d_source_counter - 1];
	for(int i = d_source_counter - 2; i >= 0; --i)
	{
		if((int)last_source.x == (int)sources[i].x && (int)last_source.y == (int)sources[i].y)
		{
			sources[i].z += last_source.z;
			--d_source_counter;
			break;
		}
	}
}