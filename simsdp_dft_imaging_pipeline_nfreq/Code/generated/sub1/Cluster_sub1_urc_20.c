/**
* @file /Cluster_sub1_urc_20.c/h
* @generated by CodegenScape
* @date Tue Dec 10 11:11:35 CET 2024
*/

#include "Cluster_sub1_urc_20.h"

static int cfg_0 = 2458;
 void Cluster_sub1_urc_20Init(){
}
void Cluster_sub1_urc_20(int cfg_0,PRECISION *delta_psi_3_delta_psf_psf){

// buffer declaration

PRECISION delta_psi_3_delta_psf_if_psf__delta_psi_3_delta_psf_sink_psf[6041764];

// body 
//delta_psi_3_delta_psf
memcpy(delta_psi_3_delta_psf_if_psf__delta_psi_3_delta_psf_sink_psf + 0,delta_psi_3_delta_psf_psf + 0,6041764*sizeof(PRECISION));
psf_sink(GRID_SIZE,delta_psi_3_delta_psf_if_psf__delta_psi_3_delta_psf_sink_psf);


// free buffer
}
