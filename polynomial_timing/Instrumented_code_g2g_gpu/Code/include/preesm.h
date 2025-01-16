/*
	============================================================================
	Name        : preesm.h
	Author      : kdesnos
	Version     : 1.0
	Copyright   : CECILL-C, IETR, INSA Rennes
	Description : Usefull declarations for all headers used in preesm.
	============================================================================
*/

#ifndef PREESM_H
#define PREESM_H

#define IN
#define OUT
//#define PREESM_VERBOSE

#define PREESM_LOOP_SIZE	(1)

#ifdef __cplusplus
extern "C" {
#endif

#include "fifoFunction.h"

// #include "timer.h"
// #include "controller.h"
// #include "gains.h"
// #include "gridder.h"
// #include "deconvolution.h"
// #include "direct_fourier_transform.h"
#include "common.h"

#ifdef __cplusplus
}
#endif

#include "md5.h"

typedef unsigned char uchar;

#ifdef USE_CUDA_EMULATOR
// En mode CPU ou émulation, on définit une version vide de CUDA_CHECK_RETURN
#define CUDA_CHECK_RETURN(x) ((void)0)

#endif






#endif
