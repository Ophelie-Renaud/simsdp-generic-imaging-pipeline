#ifndef TIMINGS_H
#define TIMINGS_H

#ifndef CPU_VERSION
#define CPU_VERSION 1
#endif


#ifdef __cplusplus
extern "C" {
#endif

	#include "preesm.h"
	#include "top.h"
	#include "dft_run.h"
	#include "gains_apply_run.h"
	#include "major_loop_iter.h"
	#include "gridding_run.h"
	#include "degridgrid.h"
	#include "fft_run.h"
	#include "convolution_correction_run.h"
	#include "deconvolution_run.h"

	void time_constant_setups(int NUM_SAMPLES);
	void time_gridsize_setups(int NUM_SAMPLES, int GRID_SIZE);
	void time_visibility_setups(int NUM_SAMPLES, int NUM_VISIBILITIES);
	void time_save_output(int NUM_SAMPLES, int GRID_SIZE);
	void time_dft(int NUM_SAMPLES, int NUM_MINOR_CYCLES, int NUM_VISIBILITIES, int NUM_ACTUAL_VISIBILITIES);
	void time_gains_application(int NUM_SAMPLES, int NUM_VISIBILITIES, int NUM_ACTUAL_VISIBILITIES);
	void time_add_visibilities(int NUM_SAMPLES, int NUM_VISIBILITIES);
	void time_prolate(int NUM_SAMPLES, int GRID_SIZE);
	void time_finegrid(int NUM_SAMPLES, int NUM_VISIBILITIES, int NUM_ACTUAL_VISIBILITIES);
	void time_subtract_ispace(int NUM_SAMPLES, int GRID_SIZE);
	void time_fftshift(int NUM_SAMPLES, int GRID_SIZE);
	void time_fft(int NUM_SAMPLES, int GRID_SIZE);
	void time_hogbom(int NUM_SAMPLES, int GRID_SIZE, int NUM_MINOR_CYCLES);
	void time_grid(int NUM_SAMPLES, int GRID_SIZE, int NUM_VISIBILITIES);
	void time_degrid(int NUM_SAMPLES, int GRID_SIZE, int NUM_VISIBILITIES);
	void time_s2s_degrid(int NUM_SAMPLES, int GRID_SIZE, int NUM_VISIBILITIES);

#include <stddef.h>

	typedef struct {
		// Atennas specs
		size_t nrows;
		// Observation specs
		size_t Nx;
		size_t half_Nx;
		size_t Ny;
		size_t half_Ny;
		size_t Nchan;
		size_t spw_selected;
		size_t oversampling_factor;
		size_t half_support_function;
		size_t full_support_function;
		size_t nb_w_planes;
		size_t nb_vis_polarization;
		size_t nb_grid_polarization;
		void* grid_channel_idx;
		size_t no_grid_index;
		size_t grid_channel_width;
		size_t no_chan_spw; // Number of nb_channels x nb_spw
		size_t nb_grid_chan;
		size_t polarization_step;
		double cell_size_l;
		double cell_size_m;
		void* u_scale;
		void* v_scale;
		// S2S
		size_t do_s2s;
		void* len_s2s_coo;
		void* a_coo;
		void* u_coo;
		void* v_coo;
		void* ch_coo;
		void* uvach_coo;
		size_t tot_size_coo;
		// Frequencies
		void* chan_wavelength;
		// Grids
		void* input_grid;
		void* output_grid;
		void* psf_grid;
		// Data specs
		void* visibilities;
		void* visibility_weight;
		void* uvw_coordinates;
		// Convolution functions specs
		void* gridding_conv_function;
		size_t filter_size;
		void* filter_AA_2D;
		size_t filter_AA_2D_size;
		size_t filter_choice;
		float max_w;
		void* conv_norm_weight;
		// W-proj Convolution functions specs
	} interpolation_parameters;


#ifdef __cplusplus
}
#endif


#endif
