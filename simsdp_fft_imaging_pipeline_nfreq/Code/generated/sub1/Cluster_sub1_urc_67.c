/**
* @file /Cluster_sub1_urc_67.c/h
* @generated by CodegenScape
* @date Tue Dec 10 11:11:38 CET 2024
*/

#include "Cluster_sub1_urc_67.h"

 void Cluster_sub1_urc_67Init(){
}
void Cluster_sub1_urc_67(PRECISION *delta_psi_2_clean_psf_clean_psf){

// buffer declaration

PRECISION delta_psi_2_clean_psf_if_clean_psf__snk_out_0_in[6041764];

// body 
//delta_psi_2_clean_psf
memcpy(delta_psi_2_clean_psf_if_clean_psf__snk_out_0_in + 0,delta_psi_2_clean_psf_clean_psf + 0,6041764*sizeof(PRECISION));
sub1_snk_out_0(delta_psi_2_clean_psf_if_clean_psf__snk_out_0_in);


// free buffer
}
