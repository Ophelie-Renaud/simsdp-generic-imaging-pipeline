/**
* @file /Cluster_sub9_urc_1.c/h
* @generated by CodegenScape
* @date Tue Dec 10 11:12:07 CET 2024
*/

#include "Cluster_sub9_urc_1.h"

 void Cluster_sub9_urc_1Init(){
}
void Cluster_sub9_urc_1(PRECISION *delta_psi_3_psi_delta_image_delta_image){

// buffer declaration

PRECISION delta_psi_3_psi_delta_image_if_delta_image__snk_out_0_in[6041764];

// body 
//delta_psi_3_psi_delta_image
memcpy(delta_psi_3_psi_delta_image_if_delta_image__snk_out_0_in + 0,delta_psi_3_psi_delta_image_delta_image + 0,6041764*sizeof(PRECISION));
sub9_snk_out_0(delta_psi_3_psi_delta_image_if_delta_image__snk_out_0_in);


// free buffer
}
