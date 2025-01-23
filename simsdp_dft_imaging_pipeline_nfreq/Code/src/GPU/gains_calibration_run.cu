
#include "gains_calibration_run.h"

#if 1
void update_gain_calibration_actor(int NUM_BASELINES, int NUM_RECEIVERS, int NUM_VISIBILITIES,
			int *loop_cycle_token, PRECISION2 *measured_vis, PRECISION2 *visibilities, PRECISION2 *gains, int2 *receiver_pairs, Config *config, PRECISION *A_array, PRECISION *Q_array)
{

//    printf("gains MD5 \t: ");
//    MD5_Update(sizeof(PRECISION2) * NUM_RECEIVERS, gains);

//    printf("measured_vis MD5 \t: ");
//    MD5_Update(sizeof(PRECISION2) * NUM_VISIBILITIES, measured_vis);
//
//    printf("visibilities MD5 \t: ");
//    MD5_Update(sizeof(PRECISION2) * NUM_VISIBILITIES, visibilities);
//
//    printf("receiver_pairs MD5 \t: ");
//    MD5_Update(sizeof(int2) * NUM_BASELINES, receiver_pairs);

    memset(Q_array, 0, sizeof(PRECISION) * 2 * NUM_RECEIVERS);
    memset(A_array, 0, sizeof(PRECISION) * 2 * NUM_RECEIVERS * 2 * NUM_RECEIVERS);

	printf("UPDATE >>> Performing Gain Calibration cycle: %d ...\n", *loop_cycle_token);
		//EXECUTE CUDA KERNEL UPDATING Q AND A array (NEED ATOMIC ACCESS!)
	update_gain_calibration_CPU(
			measured_vis,
			visibilities,
			gains,
			receiver_pairs,
			A_array,
			Q_array,
			NUM_RECEIVERS,
			NUM_BASELINES
	);
	//cudaDeviceSynchronize();

//    printf("A_array MD5 \t: ");
//    MD5_Update(sizeof(PRECISION) * 2 * NUM_RECEIVERS * 2 * NUM_RECEIVERS, A_array);
//    printf("Q_array MD5 \t: ");
//    MD5_Update(sizeof(PRECISION) * 2 * NUM_RECEIVERS, Q_array);

//	for (int i = 0; i < NUM_RECEIVERS*2; i++)
//	{
//		printf("[%d] %lf\n", i, Q_array[i]);
//	}

//	PRECISION sum = 0.0;
//	for (int i = 0; i < 2 * NUM_RECEIVERS * 2 * NUM_RECEIVERS; i++)
//	{
//		sum += A_array[i];
//	}
//	printf("A %lf\n", sum);

}

#else

void update_gain_calibration_actor(int NUM_BASELINES, int NUM_RECEIVERS, int NUM_VISIBILITIES,
			int *loop_cycle_token, PRECISION2 *measured_vis, PRECISION2 *visibilities, PRECISION2 *gains, int2 *receiver_pairs, Config *config, PRECISION *A_array, PRECISION *Q_array)
{
//
//    printf("gains MD5 \t: ");
//    MD5_Update(sizeof(PRECISION2) * NUM_RECEIVERS, gains);
//
//    printf("measured_vis MD5 \t: ");
//    MD5_Update(sizeof(PRECISION2) * NUM_VISIBILITIES, measured_vis);
//
//    printf("visibilities MD5 \t: ");
//    MD5_Update(sizeof(PRECISION2) * NUM_VISIBILITIES, visibilities);
//
//    printf("receiver_pairs MD5 \t: ");
//    MD5_Update(sizeof(int2) * NUM_BASELINES, receiver_pairs);

	PRECISION2 *d_measured_vis = NULL;
	PRECISION2 *d_visibilities = NULL;
	PRECISION2 *d_gains = NULL;
	int2 *d_receiver_pairs = NULL;
	PRECISION *d_A_array = NULL;
	PRECISION *d_Q_array = NULL;

	// Allocating device measured_vis buffer
	CUDA_CHECK_RETURN(cudaMalloc(&d_measured_vis, sizeof(PRECISION2) * NUM_VISIBILITIES));
	CUDA_CHECK_RETURN(cudaMemcpy(d_measured_vis, measured_vis, sizeof(PRECISION2) * NUM_VISIBILITIES, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

    // Allocating device input visibilities buffer
	CUDA_CHECK_RETURN(cudaMalloc(&d_visibilities, sizeof(PRECISION2) * NUM_VISIBILITIES));
	CUDA_CHECK_RETURN(cudaMemcpy(d_visibilities, visibilities, sizeof(PRECISION2) * NUM_VISIBILITIES, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// Allocating device gains buffer
	CUDA_CHECK_RETURN(cudaMalloc(&d_gains, sizeof(PRECISION2) * NUM_RECEIVERS));
	CUDA_CHECK_RETURN(cudaMemcpy(d_gains, gains, sizeof(PRECISION2) * NUM_RECEIVERS, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// Allocating device gains buffer
	CUDA_CHECK_RETURN(cudaMalloc(&d_receiver_pairs, sizeof(PRECISION2) * NUM_BASELINES));
	CUDA_CHECK_RETURN(cudaMemcpy(d_receiver_pairs, receiver_pairs, sizeof(int2) * NUM_BASELINES, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_A_array,  sizeof(PRECISION) * 2 * NUM_RECEIVERS * 2 * NUM_RECEIVERS));
	CUDA_CHECK_RETURN(cudaMemset(d_A_array, 0, sizeof(PRECISION) * 2 * NUM_RECEIVERS * 2 * NUM_RECEIVERS));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_Q_array,  sizeof(PRECISION) * 2 * NUM_RECEIVERS));
	CUDA_CHECK_RETURN(cudaMemset(d_Q_array, 0, sizeof(PRECISION) * 2 * NUM_RECEIVERS));
	cudaDeviceSynchronize();

	//SET CUDA WORK PLAN:
 	int max_threads_per_block = min(config->gpu_max_threads_per_block, NUM_BASELINES);
	int num_blocks = (int) ceil((double) NUM_BASELINES / max_threads_per_block);
	dim3 kernel_blocks(num_blocks, 1, 1);
	dim3 kernel_threads(max_threads_per_block, 1, 1);


	printf("UPDATE >>> Performing Gain Calibration cycle: %d ...\n", *loop_cycle_token);
		//EXECUTE CUDA KERNEL UPDATING Q AND A array (NEED ATOMIC ACCESS!)
	update_gain_calibration<<<kernel_blocks, kernel_threads>>>(
			d_measured_vis,
			d_visibilities,
			d_gains,
			d_receiver_pairs,
			d_A_array,
			d_Q_array,
			NUM_RECEIVERS,
			NUM_BASELINES
	);
	cudaDeviceSynchronize();


	CUDA_CHECK_RETURN(cudaMemcpy(A_array, d_A_array, sizeof(PRECISION) * 2 * NUM_RECEIVERS * 2 * NUM_RECEIVERS, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy(Q_array, d_Q_array, sizeof(PRECISION) * 2 * NUM_RECEIVERS, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

//    printf("A_array MD5 \t: ");
//    MD5_Update(sizeof(PRECISION) * 2 * NUM_RECEIVERS * 2 * NUM_RECEIVERS, A_array);
//    printf("Q_array MD5 \t: ");
//    MD5_Update(sizeof(PRECISION) * 2 * NUM_RECEIVERS, Q_array);

	CUDA_CHECK_RETURN(cudaFree(d_receiver_pairs));
	CUDA_CHECK_RETURN(cudaFree(d_gains));
	CUDA_CHECK_RETURN(cudaFree(d_visibilities));
	CUDA_CHECK_RETURN(cudaFree(d_measured_vis));
	CUDA_CHECK_RETURN(cudaFree(d_Q_array));
	CUDA_CHECK_RETURN(cudaFree(d_A_array));
}

#endif

void execute_calibration_SVD_actor(int NUM_COLS, int NUM_ROWS,
			PRECISION *A_array, PRECISION *S_array, PRECISION *U_array, PRECISION *V_array)
{
	PRECISION *d_A_array = NULL;
	PRECISION *d_U_array = NULL;
	PRECISION *d_V_array = NULL;
	PRECISION *d_S_array = NULL;

	PRECISION *work;

	cusolverDnHandle_t solver_handle;
	cusolverDnCreate(&solver_handle);

	int work_size = 0;
	CUDA_SOLVER_CHECK_RETURN(SVD_BUFFER_SIZE(solver_handle, NUM_ROWS, NUM_COLS, &work_size));


	CUDA_CHECK_RETURN(cudaMalloc(&work, work_size * sizeof(PRECISION)));
	CUDA_CHECK_RETURN(cudaMemset(work, 0, sizeof(PRECISION) * work_size));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_A_array,  sizeof(PRECISION) * NUM_ROWS * NUM_COLS));
	CUDA_CHECK_RETURN(cudaMemcpy(d_A_array, A_array, sizeof(PRECISION) * NUM_ROWS * NUM_COLS, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_S_array, sizeof(PRECISION) * NUM_ROWS));
	CUDA_CHECK_RETURN(cudaMemset(d_S_array, 0, sizeof(PRECISION) * NUM_ROWS));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_U_array, sizeof(PRECISION) * NUM_ROWS * NUM_COLS));
	CUDA_CHECK_RETURN(cudaMemset(d_U_array, 0, sizeof(PRECISION) * NUM_ROWS * NUM_COLS));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_V_array, sizeof(PRECISION) * NUM_ROWS * NUM_COLS));
	CUDA_CHECK_RETURN(cudaMemset(d_V_array, 0, sizeof(PRECISION) * NUM_ROWS * NUM_COLS));
	cudaDeviceSynchronize();


	int *devInfo;
	CUDA_CHECK_RETURN(cudaMalloc(&devInfo, sizeof(int)));

	//ALL OUR MATRICES ARE "ROW" MAJOR HOWEVER AS A IS SYMMETRIC DOES NOT NEED TO BE TRANSPOSED FOR FOR SVD ROUTINE
	//SO NEED TO NOT TRANSPOSE U AND TRANSPOSE VSTAR
	CUDA_SOLVER_CHECK_RETURN(SVD(solver_handle, 'A', 'A', NUM_ROWS, NUM_COLS, d_A_array,
		 NUM_ROWS, d_S_array, d_U_array, NUM_ROWS, d_V_array, NUM_COLS, work, work_size, NULL, devInfo));
	cudaDeviceSynchronize();

	int devInfo_h = 0;	
	CUDA_CHECK_RETURN(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();



	CUDA_CHECK_RETURN(cudaMemcpy(S_array, d_S_array, sizeof(PRECISION) * NUM_ROWS, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy(U_array, d_U_array, sizeof(PRECISION) * NUM_ROWS * NUM_COLS, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy(V_array, d_V_array, sizeof(PRECISION) * NUM_ROWS * NUM_COLS, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();


	CUDA_CHECK_RETURN(cudaFree(d_S_array));
	CUDA_CHECK_RETURN(cudaFree(d_U_array));
	CUDA_CHECK_RETURN(cudaFree(d_V_array));
	CUDA_CHECK_RETURN(cudaFree(d_A_array));
	CUDA_CHECK_RETURN(cudaFree(work));


//    printf("h_V_array MD5 \t: ");
//    MD5_Update(sizeof(PRECISION) * NUM_ROWS * NUM_COLS, V_array);
//
//    printf("h_U_array MD5 \t: ");
//    MD5_Update(sizeof(PRECISION) * NUM_ROWS * NUM_COLS, U_array);
//
//    printf("h_S_array MD5 \t: ");
//    MD5_Update(sizeof(PRECISION) * NUM_ROWS, S_array);


//	for (int i = 0; i < NUM_ROWS; i++)
//	{
//		printf("[%d] %lf\n", i, S_array[i]);
//	}


	//CHECKING S PRODUCT!!
    //bool success = (devInfo_h == 0);
	//printf("UPDATE >>> SVD complete...\n\n");
	if (devInfo) CUDA_CHECK_RETURN(cudaFree(devInfo));
	//return success;
}


void calculate_SUQ_product_actor(int NUM_COLS, int NUM_ROWS,
			PRECISION *S_array, PRECISION *U_array, PRECISION *Q_array, Config *config, PRECISION *SUQ_array)
{

	PRECISION *d_S_array = NULL;
	PRECISION *d_U_array = NULL;
	PRECISION *d_Q_array = NULL;
	PRECISION *d_SUQ_array = NULL;


	CUDA_CHECK_RETURN(cudaMalloc(&d_S_array, sizeof(PRECISION) * NUM_ROWS));
	CUDA_CHECK_RETURN(cudaMemcpy(d_S_array, S_array, sizeof(PRECISION) * NUM_ROWS, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_U_array, sizeof(PRECISION) * NUM_ROWS * NUM_COLS));
	CUDA_CHECK_RETURN(cudaMemcpy(d_U_array, U_array, sizeof(PRECISION) * NUM_ROWS * NUM_COLS, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_Q_array, sizeof(PRECISION) * NUM_ROWS));
	CUDA_CHECK_RETURN(cudaMemcpy(d_Q_array, Q_array, sizeof(PRECISION) * NUM_ROWS, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_SUQ_array, sizeof(PRECISION) * NUM_COLS));
	CUDA_CHECK_RETURN(cudaMemset(d_SUQ_array, 0, sizeof(PRECISION) * NUM_COLS));
	cudaDeviceSynchronize();


	int max_threads_per_block = min(config->gpu_max_threads_per_block, NUM_COLS);
	int num_blocks = (int) ceil((double)NUM_COLS / max_threads_per_block);
	dim3 kernel_blocks(num_blocks, 1, 1);
	dim3 kernel_threads(max_threads_per_block, 1, 1);
	//Calculate product of S inverse,U and Q
	calculate_suq_product<<<kernel_blocks, kernel_threads>>>(
		d_S_array,
		d_U_array,
		d_Q_array, 
		d_SUQ_array, 
		NUM_COLS);
	cudaDeviceSynchronize();


	CUDA_CHECK_RETURN(cudaMemcpy(SUQ_array, d_SUQ_array, sizeof(PRECISION) * NUM_COLS, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaFree(d_S_array));
	CUDA_CHECK_RETURN(cudaFree(d_U_array));
	CUDA_CHECK_RETURN(cudaFree(d_Q_array));
	CUDA_CHECK_RETURN(cudaFree(d_SUQ_array));

//    printf("h_SUQ_array MD5\t: ");
//    MD5_Update(sizeof(PRECISION) * NUM_COLS, SUQ_array);

}

void calculate_delta_update_gains_actor(int NUM_COLS, int NUM_RECEIVERS,
			PRECISION *SUQ_array, PRECISION *V_array, Config *config, PRECISION2 *gains_in, PRECISION2* gains_out)
{
//    printf("h_V_array MD5 \t: ");
//    MD5_Update(sizeof(PRECISION) * NUM_COLS * NUM_COLS, V_array);

//    printf("h_SUQ_array MD5\t: ");
//    MD5_Update(sizeof(PRECISION) * NUM_COLS, SUQ_array);

	printf("gains_in  %lf %lf\n", gains_in[2].x, gains_in[2].y);
	printf("\n");

	//problème avec V_array, différence de signe entre FP32 et FP64
	//problème avec SUQ_array, différence de signe entre FP32 et FP64
	//notamment SUQ_array[1023] valant 0 en fp64 et 11.59 en fp32

	for (int i = 0; i < NUM_COLS; i++)
	{
//		printf("[%d] %lf %lf\n", i, V_array[2*2*NUM_COLS +i], V_array[(2*2+1)*NUM_COLS +i]);
//		printf("[%d] %lf\n", i, SUQ_array[i]);
		__asm__("nop");
	}

	//vindex*num_cols + i

	PRECISION *d_V_array = NULL;
	PRECISION *d_SUQ_array = NULL;
	PRECISION2 *d_gains = NULL;


	CUDA_CHECK_RETURN(cudaMalloc(&d_V_array, sizeof(PRECISION) * NUM_COLS * NUM_COLS));
	CUDA_CHECK_RETURN(cudaMemcpy(d_V_array, V_array, sizeof(PRECISION) * NUM_COLS * NUM_COLS, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_SUQ_array, sizeof(PRECISION) * NUM_COLS));
	CUDA_CHECK_RETURN(cudaMemcpy(d_SUQ_array, SUQ_array, sizeof(PRECISION) * NUM_COLS, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// Allocating device gains buffer
	CUDA_CHECK_RETURN(cudaMalloc(&d_gains, sizeof(PRECISION2) * NUM_RECEIVERS));
	CUDA_CHECK_RETURN(cudaMemcpy(d_gains, gains_in, sizeof(PRECISION2) * NUM_RECEIVERS, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();


	int max_threads_per_block = min(config->gpu_max_threads_per_block, NUM_RECEIVERS);
	int num_blocks = (int) ceil((double) NUM_RECEIVERS / max_threads_per_block);
	dim3 blocks(num_blocks, 1, 1);
	dim3 threads(max_threads_per_block, 1, 1);
	//Caluclate product of V and SUQ from above, and update gains array
	calculate_delta_update_gains<<<blocks, threads>>>(
			d_V_array,
			d_SUQ_array,
			d_gains,
			NUM_RECEIVERS,
			NUM_COLS
		);
	cudaDeviceSynchronize();


	CUDA_CHECK_RETURN(cudaMemcpy(gains_out, d_gains, sizeof(PRECISION2) * NUM_RECEIVERS, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaFree(d_V_array));
	CUDA_CHECK_RETURN(cudaFree(d_SUQ_array));
	CUDA_CHECK_RETURN(cudaFree(d_gains));

	printf("gains_out %lf %lf\n", gains_out[2].x, gains_out[2].y);
}



//delta+= transpose(V)*product of S U and Q (calculated in previous kernel)
__global__ void calculate_delta_update_gains(const PRECISION *d_V, const PRECISION *d_SUQ, PRECISION2 *d_gains, 
													const int num_recievers, const int num_cols)
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x ;

	if(index >= num_recievers)
		return;

	PRECISION delta_top = 0;
	PRECISION delta_bottom = 0;

	int vindex = index * 2;
	for(int i=0;i<num_cols; ++i)
	{
		delta_top += d_SUQ[i] * d_V[vindex*num_cols + i];//[i*num_cols + vindex];
		delta_bottom += d_SUQ[i] * d_V[(vindex+1)*num_cols + i];//[i*num_cols + vindex+1];
	}

	d_gains[index].x += delta_top;
	d_gains[index].y += delta_bottom; 
}

__global__ void calculate_suq_product(const PRECISION *d_S, const PRECISION *d_U, const PRECISION *d_Q, 
	PRECISION *d_SUQ, const int num_entries)
{
	//qus 2Nx1, q = 2Nx1 , s=2Nx1, u=2N*2N
	const int index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index >= num_entries)
		return;

	PRECISION sinv = (ABS(d_S[index]) > 1E-6) ? 1.0/d_S[index] : 0.0;
//	PRECISION sinv = (ABS(d_S[index]) > 1E-4) ? 1.0/d_S[index] : 0.0;

	PRECISION product = 0; 
	for(int i=0;i<num_entries;++i)
	{
		product += d_Q[i] * d_U[index*num_entries + i];
	}

	d_SUQ[index] = product * sinv;

	if(index == 1023)
	{
		printf("S[1023]: %lf\n", d_S[1023]);
		printf("SUQ[1023]: %lf * %lf\n", product, sinv);
	}
}

__global__ void update_gain_calibration(const PRECISION2 *vis_measured_array, const PRECISION2 *vis_predicted_array, 
	const PRECISION2 *gains_array, const int2 *receiver_pairs, PRECISION *A_array, PRECISION *Q_array, 
	const int num_recievers, const int num_baselines)
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x;
	if(index >= num_baselines)
		return;

	PRECISION2 vis_measured = vis_measured_array[index];
	PRECISION2 vis_predicted = vis_predicted_array[index];

	int2 antennae = receiver_pairs[index];

	PRECISION2 gainA = gains_array[antennae.x];
	//only need gainB as conjugate??
	PRECISION2 gainB_conjugate = complex_conjugate(gains_array[antennae.y]);

	//NOTE do not treat residual as a COMPLEX!!!!! (2 reals)
	PRECISION2 residual = complex_subtract(vis_measured, complex_multiply(vis_predicted,complex_multiply(gainA, gainB_conjugate)));


	//CALCULATE Partial Derivatives

	PRECISION2 part_respect_to_real_gain_a = complex_multiply(vis_predicted, gainB_conjugate);

	PRECISION2 part_respect_to_imag_gain_a = flip_for_i(complex_multiply(vis_predicted, gainB_conjugate));

	PRECISION2 part_respect_to_real_gain_b = complex_multiply(vis_predicted,gainA);

	PRECISION2 part_respect_to_imag_gain_b = flip_for_neg_i(complex_multiply(vis_predicted, gainA));

	//Calculate Q[2a],Q[2a+1],Q[2b],Q[2b+1] arrays - In this order... NEED ATOMIC UPDATE 
	double qValue = part_respect_to_real_gain_a.x * residual.x 
					+ part_respect_to_real_gain_a.y * residual.y;
	atomicAdd(&(Q_array[2*antennae.x]), qValue);

	qValue = part_respect_to_imag_gain_a.x * residual.x 
					+ part_respect_to_imag_gain_a.y * residual.y;
	atomicAdd(&(Q_array[2*antennae.x+1]), qValue);

	qValue = part_respect_to_real_gain_b.x * residual.x 
					+ part_respect_to_real_gain_b.y * residual.y;
	atomicAdd(&(Q_array[2*antennae.y]), qValue);

	qValue = part_respect_to_imag_gain_b.x * residual.x 
					+ part_respect_to_imag_gain_b.y * residual.y;
	atomicAdd(&(Q_array[2*antennae.y+1]), qValue);


	int num_cols = 2 * num_recievers;
	//CALCULATE JAcobian product on A matrix... 2a2a, 2a2a+1, 2a2b, 2a2b+1
	//2a2a
	double aValue = part_respect_to_real_gain_a.x * part_respect_to_real_gain_a.x + 
					part_respect_to_real_gain_a.y * part_respect_to_real_gain_a.y; 
	
	int aIndex = (2 *  antennae.x * num_cols) + (2 * antennae.x);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a2a+1,
	aValue = part_respect_to_real_gain_a.x * part_respect_to_imag_gain_a.x + 
					part_respect_to_real_gain_a.y * part_respect_to_imag_gain_a.y; 

	aIndex = (2 *  antennae.x * num_cols) + (2 * antennae.x + 1);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a2b
	aValue = part_respect_to_real_gain_a.x * part_respect_to_real_gain_b.x + 
					part_respect_to_real_gain_a.y * part_respect_to_real_gain_b.y;

	aIndex = (2 *  antennae.x * num_cols) + (2 * antennae.y);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a2b+1
	aValue = part_respect_to_real_gain_a.x * part_respect_to_imag_gain_b.x + 
					part_respect_to_real_gain_a.y * part_respect_to_imag_gain_b.y;

	aIndex = (2 *  antennae.x * num_cols) + (2 * antennae.y+1);

	atomicAdd(&(A_array[aIndex]), aValue);
	//CACLUATE JAcobian product on A matrix... [2a+1,2a], [2a+1,2a+1], [2a+1,2b], [2a+1,2b+1]
	//2a+1,2a
	aValue = part_respect_to_imag_gain_a.x * part_respect_to_real_gain_a.x + 
					part_respect_to_imag_gain_a.y * part_respect_to_real_gain_a.y; 

	aIndex = ((2 *  antennae.x+1) * num_cols) + (2 * antennae.x);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a+1,2a+1
	aValue = part_respect_to_imag_gain_a.x * part_respect_to_imag_gain_a.x + 
					part_respect_to_imag_gain_a.y * part_respect_to_imag_gain_a.y; 

	aIndex = ((2 *  antennae.x+1) * num_cols) + (2 * antennae.x+1);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a+1,2b
	aValue = part_respect_to_imag_gain_a.x * part_respect_to_real_gain_b.x + 
					part_respect_to_imag_gain_a.y * part_respect_to_real_gain_b.y;

	aIndex = ((2 *  antennae.x+1) * num_cols) + (2 * antennae.y);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2a+1,2b+1
	aValue = part_respect_to_imag_gain_a.x * part_respect_to_imag_gain_b.x + 
					part_respect_to_imag_gain_a.y * part_respect_to_imag_gain_b.y;

	aIndex = ((2 *  antennae.x+1) * num_cols) + (2 * antennae.y+1);

	atomicAdd(&(A_array[aIndex]), aValue);


	//CACLUATE JAcobian product on A matrix... [2b,2a], [2b,2a+1], [2b,2b], [2b,2b+1]
	//2b,2a
	aValue = part_respect_to_real_gain_b.x * part_respect_to_real_gain_a.x + 
					part_respect_to_real_gain_b.y * part_respect_to_real_gain_a.y; 
	
	aIndex = (2 *  antennae.y * num_cols) + (2 * antennae.x);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2b,2a+1
	aValue = part_respect_to_real_gain_b.x * part_respect_to_imag_gain_a.x + 
			 		part_respect_to_real_gain_b.y * part_respect_to_imag_gain_a.y;

	aIndex = (2 *  antennae.y * num_cols) + (2 * antennae.x+1);

	atomicAdd(&(A_array[aIndex]), aValue);
	
	//2b,2b
	aValue = part_respect_to_real_gain_b.x * part_respect_to_real_gain_b.x + 
					part_respect_to_real_gain_b.y * part_respect_to_real_gain_b.y;
	
	aIndex = (2 *  antennae.y * num_cols) + (2 * antennae.y);

	atomicAdd(&(A_array[aIndex]), aValue);


	//2b, 2b+1
	aValue = part_respect_to_real_gain_b.x * part_respect_to_imag_gain_b.x + 
					part_respect_to_real_gain_b.y * part_respect_to_imag_gain_b.y;

	aIndex = (2 *  antennae.y * num_cols) + (2 * antennae.y + 1);

	atomicAdd(&(A_array[aIndex]), aValue);

	//CALCULATE JAcobian product on A matrix... [2b+1,2a], [2b+1,2a+1], [2b+1,2b], [2b+1,2b+1]
	//2b+1,2a
	aValue =  part_respect_to_imag_gain_b.x * part_respect_to_real_gain_a.x + 
					part_respect_to_imag_gain_b.y * part_respect_to_real_gain_a.y; 

	aIndex = ((2 *  antennae.y+1) * num_cols) + (2 * antennae.x);

	atomicAdd(&(A_array[aIndex]), aValue);


	//2b+1,2a+1
	aValue =  part_respect_to_imag_gain_b.x * part_respect_to_imag_gain_a.x+ 
					 part_respect_to_imag_gain_b.y * part_respect_to_imag_gain_a.y;

	aIndex = ((2 *  antennae.y+1) * num_cols) + (2 * antennae.x+1);

	atomicAdd(&(A_array[aIndex]), aValue);


	//2b+1,2b
	aValue =  part_respect_to_imag_gain_b.x * part_respect_to_real_gain_b.x+ 
					part_respect_to_imag_gain_b.y * part_respect_to_real_gain_b.y;

	aIndex = ((2 *  antennae.y+1) * num_cols) + (2 * antennae.y);

	atomicAdd(&(A_array[aIndex]), aValue);

	//2b+1, 2b+1
	aValue = part_respect_to_imag_gain_b.x * part_respect_to_imag_gain_b.x + 
					part_respect_to_imag_gain_b.y * part_respect_to_imag_gain_b.y; 

	aIndex = ((2 *  antennae.y+1) * num_cols) + (2 * antennae.y+1);

	atomicAdd(&(A_array[aIndex]), aValue);
}


__device__ PRECISION2 flip_for_i(const PRECISION2 z)
{
	return MAKE_PRECISION2(-z.y, z.x);
}

__device__ PRECISION2 flip_for_neg_i(const PRECISION2 z)
{
	return MAKE_PRECISION2(z.y, -z.x);
}

__device__ PRECISION2 complex_multiply(const PRECISION2 z1, const PRECISION2 z2)
{	
	return MAKE_PRECISION2(z1.x * z2.x - z1.y * z2.y, z1.x * z2.y + z1.y * z2.x);
}

__device__ PRECISION2 complex_conjugate(const PRECISION2 z1)
{
	return MAKE_PRECISION2(z1.x, -z1.y);
}

__device__ PRECISION2 complex_subtract(const PRECISION2 z1, const PRECISION2 z2)
{
	return MAKE_PRECISION2(z1.x - z2.x, z1.y - z2.y);
}

__device__ PRECISION2 complex_reciprocal(const PRECISION2 z)
{
	PRECISION real = z.x / (z.x * z.x + z.y * z.y); 
	PRECISION imag = z.y / (z.x * z.x + z.y * z.y); 
	return MAKE_PRECISION2(real, -imag); 
}

void check_cuda_solver_error_aux(const char *file, unsigned line, const char *statement, cusolverStatus_t err)
{
	if (err == cudaSuccess)
		return;

	printf(">>> CUDA ERROR: %s returned %s: %u ", statement, file, line);
	exit(EXIT_FAILURE);
}

void update_gain_calibration_CPU(const PRECISION2 *vis_measured_array, const PRECISION2 *vis_predicted_array,
                                 const PRECISION2 *gains_array, const int2 *receiver_pairs, PRECISION *A_array, PRECISION *Q_array,
                                 const int num_recievers, const int num_baselines)
{
    // const int index = threadIdx.x + blockDim.x * blockIdx.x;
    // if(index >= num_baselines)
    // 	return;

    for (int index = 0; index < num_baselines; index++)
    {

        PRECISION2 vis_measured = vis_measured_array[index];
        PRECISION2 vis_predicted = vis_predicted_array[index];

        int2 antennae = receiver_pairs[index];

        PRECISION2 gainA = gains_array[antennae.x];
        //only need gainB as conjugate??
        PRECISION2 gainB_conjugate = complex_conjugate_CPU(gains_array[antennae.y]);

        //NOTE do not treat residual as a COMPLEX!!!!! (2 reals)
        PRECISION2 residual = complex_subtract_CPU(vis_measured, complex_multiply_CPU(vis_predicted,complex_multiply_CPU(gainA, gainB_conjugate)));


        //CALCULATE Partial Derivatives

        PRECISION2 part_respect_to_real_gain_a = complex_multiply_CPU(vis_predicted, gainB_conjugate);

        PRECISION2 part_respect_to_imag_gain_a = flip_for_i_CPU(complex_multiply_CPU(vis_predicted, gainB_conjugate));

        PRECISION2 part_respect_to_real_gain_b = complex_multiply_CPU(vis_predicted,gainA);

        PRECISION2 part_respect_to_imag_gain_b = flip_for_neg_i_CPU(complex_multiply_CPU(vis_predicted, gainA));

        //Calculate Q[2a],Q[2a+1],Q[2b],Q[2b+1] arrays - In this order... NEED ATOMIC UPDATE
        double qValue = part_respect_to_real_gain_a.x * residual.x
                        + part_respect_to_real_gain_a.y * residual.y;
        // atomicAdd(&(Q_array[2*antennae.x]), qValue);
        Q_array[2*antennae.x] += qValue;

        qValue = part_respect_to_imag_gain_a.x * residual.x
                 + part_respect_to_imag_gain_a.y * residual.y;
        // atomicAdd(&(Q_array[2*antennae.x+1]), qValue);
        Q_array[2*antennae.x+1] += qValue;

        qValue = part_respect_to_real_gain_b.x * residual.x
                 + part_respect_to_real_gain_b.y * residual.y;
        // atomicAdd(&(Q_array[2*antennae.y]), qValue);
        Q_array[2*antennae.y] += qValue;

        qValue = part_respect_to_imag_gain_b.x * residual.x
                 + part_respect_to_imag_gain_b.y * residual.y;
        // atomicAdd(&(Q_array[2*antennae.y+1]), qValue);
        Q_array[2*antennae.y+1] += qValue;


        int num_cols = 2 * num_recievers;
        //CALCULATE JAcobian product on A matrix... 2a2a, 2a2a+1, 2a2b, 2a2b+1
        //2a2a
        double aValue = part_respect_to_real_gain_a.x * part_respect_to_real_gain_a.x +
                        part_respect_to_real_gain_a.y * part_respect_to_real_gain_a.y;

        int aIndex = (2 *  antennae.x * num_cols) + (2 * antennae.x);

        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //2a2a+1,
        aValue = part_respect_to_real_gain_a.x * part_respect_to_imag_gain_a.x +
                 part_respect_to_real_gain_a.y * part_respect_to_imag_gain_a.y;

        aIndex = (2 *  antennae.x * num_cols) + (2 * antennae.x + 1);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //2a2b
        aValue = part_respect_to_real_gain_a.x * part_respect_to_real_gain_b.x +
                 part_respect_to_real_gain_a.y * part_respect_to_real_gain_b.y;

        aIndex = (2 *  antennae.x * num_cols) + (2 * antennae.y);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //2a2b+1
        aValue = part_respect_to_real_gain_a.x * part_respect_to_imag_gain_b.x +
                 part_respect_to_real_gain_a.y * part_respect_to_imag_gain_b.y;

        aIndex = (2 *  antennae.x * num_cols) + (2 * antennae.y+1);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //CACLUATE JAcobian product on A matrix... [2a+1,2a], [2a+1,2a+1], [2a+1,2b], [2a+1,2b+1]
        //2a+1,2a
        aValue = part_respect_to_imag_gain_a.x * part_respect_to_real_gain_a.x +
                 part_respect_to_imag_gain_a.y * part_respect_to_real_gain_a.y;

        aIndex = ((2 *  antennae.x+1) * num_cols) + (2 * antennae.x);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //2a+1,2a+1
        aValue = part_respect_to_imag_gain_a.x * part_respect_to_imag_gain_a.x +
                 part_respect_to_imag_gain_a.y * part_respect_to_imag_gain_a.y;

        aIndex = ((2 *  antennae.x+1) * num_cols) + (2 * antennae.x+1);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //2a+1,2b
        aValue = part_respect_to_imag_gain_a.x * part_respect_to_real_gain_b.x +
                 part_respect_to_imag_gain_a.y * part_respect_to_real_gain_b.y;

        aIndex = ((2 *  antennae.x+1) * num_cols) + (2 * antennae.y);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //2a+1,2b+1
        aValue = part_respect_to_imag_gain_a.x * part_respect_to_imag_gain_b.x +
                 part_respect_to_imag_gain_a.y * part_respect_to_imag_gain_b.y;

        aIndex = ((2 *  antennae.x+1) * num_cols) + (2 * antennae.y+1);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //CACLUATE JAcobian product on A matrix... [2b,2a], [2b,2a+1], [2b,2b], [2b,2b+1]
        //2b,2a
        aValue = part_respect_to_real_gain_b.x * part_respect_to_real_gain_a.x +
                 part_respect_to_real_gain_b.y * part_respect_to_real_gain_a.y;

        aIndex = (2 *  antennae.y * num_cols) + (2 * antennae.x);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //2b,2a+1
        aValue = part_respect_to_real_gain_b.x * part_respect_to_imag_gain_a.x +
                 part_respect_to_real_gain_b.y * part_respect_to_imag_gain_a.y;

        aIndex = (2 *  antennae.y * num_cols) + (2 * antennae.x+1);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //2b,2b
        aValue = part_respect_to_real_gain_b.x * part_respect_to_real_gain_b.x +
                 part_respect_to_real_gain_b.y * part_respect_to_real_gain_b.y;

        aIndex = (2 *  antennae.y * num_cols) + (2 * antennae.y);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //2b, 2b+1
        aValue = part_respect_to_real_gain_b.x * part_respect_to_imag_gain_b.x +
                 part_respect_to_real_gain_b.y * part_respect_to_imag_gain_b.y;

        aIndex = (2 *  antennae.y * num_cols) + (2 * antennae.y + 1);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //CALCULATE JAcobian product on A matrix... [2b+1,2a], [2b+1,2a+1], [2b+1,2b], [2b+1,2b+1]
        //2b+1,2a
        aValue =  part_respect_to_imag_gain_b.x * part_respect_to_real_gain_a.x +
                  part_respect_to_imag_gain_b.y * part_respect_to_real_gain_a.y;

        aIndex = ((2 *  antennae.y+1) * num_cols) + (2 * antennae.x);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //2b+1,2a+1
        aValue =  part_respect_to_imag_gain_b.x * part_respect_to_imag_gain_a.x+
                  part_respect_to_imag_gain_b.y * part_respect_to_imag_gain_a.y;

        aIndex = ((2 *  antennae.y+1) * num_cols) + (2 * antennae.x+1);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //2b+1,2b
        aValue =  part_respect_to_imag_gain_b.x * part_respect_to_real_gain_b.x+
                  part_respect_to_imag_gain_b.y * part_respect_to_real_gain_b.y;

        aIndex = ((2 *  antennae.y+1) * num_cols) + (2 * antennae.y);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;

        //2b+1, 2b+1
        aValue = part_respect_to_imag_gain_b.x * part_respect_to_imag_gain_b.x +
                 part_respect_to_imag_gain_b.y * part_respect_to_imag_gain_b.y;

        aIndex = ((2 *  antennae.y+1) * num_cols) + (2 * antennae.y+1);
        // atomicAdd(&(A_array[aIndex]), aValue);
        A_array[aIndex] += aValue;
    }
}


PRECISION2 complex_multiply_CPU(const PRECISION2 z1, const PRECISION2 z2)
{
    return MAKE_PRECISION2(z1.x * z2.x - z1.y * z2.y, z1.x * z2.y + z1.y * z2.x);
}

PRECISION2 flip_for_i_CPU(const PRECISION2 z)
{
    return MAKE_PRECISION2(-z.y, z.x);
}

PRECISION2 flip_for_neg_i_CPU(const PRECISION2 z)
{
    return MAKE_PRECISION2(z.y, -z.x);
}

PRECISION2 complex_conjugate_CPU(const PRECISION2 z1)
{
    return MAKE_PRECISION2(z1.x, -z1.y);
}

PRECISION2 complex_subtract_CPU(const PRECISION2 z1, const PRECISION2 z2)
{
    return MAKE_PRECISION2(z1.x - z2.x, z1.y - z2.y);
}