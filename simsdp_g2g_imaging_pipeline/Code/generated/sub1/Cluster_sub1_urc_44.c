/**
* @file /Cluster_sub1_urc_44.c/h
* @generated by CodegenScape
* @date Tue Dec 10 11:11:37 CET 2024
*/

#include "Cluster_sub1_urc_44.h"

 void Cluster_sub1_urc_44Init(){
}
void Cluster_sub1_urc_44(int2 *delta_psi_2_degridding_kernel_supports_degridding_kernel_supports){

// buffer declaration

int2 delta_psi_2_degridding_kernel_supports_if_degridding_kernel_supports__delta_psi_2_delta_degridding_kernel_supports_degridding_kernel_supports[17];

int2 delta_psi_2_delta_degridding_kernel_supports_if_degridding_kernel_supports__snk_out_129_in[68];

// body 
//delta_psi_2_degridding_kernel_supports
memcpy(delta_psi_2_degridding_kernel_supports_if_degridding_kernel_supports__delta_psi_2_delta_degridding_kernel_supports_degridding_kernel_supports + 0,delta_psi_2_degridding_kernel_supports_degridding_kernel_supports + 0,17*sizeof(int2));
//delta_psi_2_delta_degridding_kernel_supports
memcpy(delta_psi_2_delta_degridding_kernel_supports_if_degridding_kernel_supports__snk_out_129_in + 0,delta_psi_2_degridding_kernel_supports_if_degridding_kernel_supports__delta_psi_2_delta_degridding_kernel_supports_degridding_kernel_supports + 0,68*sizeof(int2));
sub1_snk_out_129(delta_psi_2_delta_degridding_kernel_supports_if_degridding_kernel_supports__snk_out_129_in);


// free buffer
}
