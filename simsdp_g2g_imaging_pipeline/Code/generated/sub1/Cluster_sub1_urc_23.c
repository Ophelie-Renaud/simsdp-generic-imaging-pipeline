/**
* @file /Cluster_sub1_urc_23.c/h
* @generated by CodegenScape
* @date Tue Dec 10 11:11:35 CET 2024
*/

#include "Cluster_sub1_urc_23.h"

 void Cluster_sub1_urc_23Init(){
}
void Cluster_sub1_urc_23(PRECISION *delta_psi_4_psi_psf_psf){

// buffer declaration

PRECISION delta_psi_4_psi_psf_if_psf__snk_out_77_in[6041764];

// body 
//delta_psi_4_psi_psf
memcpy(delta_psi_4_psi_psf_if_psf__snk_out_77_in + 0,delta_psi_4_psi_psf_psf + 0,6041764*sizeof(PRECISION));
sub1_snk_out_77(delta_psi_4_psi_psf_if_psf__snk_out_77_in);


// free buffer
}
