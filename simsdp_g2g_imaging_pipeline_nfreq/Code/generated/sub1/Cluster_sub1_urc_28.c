/**
* @file /Cluster_sub1_urc_28.c/h
* @generated by CodegenScape
* @date Tue Dec 10 11:11:36 CET 2024
*/

#include "Cluster_sub1_urc_28.h"

 void Cluster_sub1_urc_28Init(){
}
void Cluster_sub1_urc_28(PRECISION *delta_psi_1_delta_convolution_correction_run_prolate_prolate){

// buffer declaration

PRECISION delta_psi_1_delta_convolution_correction_run_prolate_if_prolate__snk_out_99_in[1229];

// body 
//delta_psi_1_delta_convolution_correction_run_prolate
memcpy(delta_psi_1_delta_convolution_correction_run_prolate_if_prolate__snk_out_99_in + 0,delta_psi_1_delta_convolution_correction_run_prolate_prolate + 0,1229*sizeof(PRECISION));
sub1_snk_out_99(delta_psi_1_delta_convolution_correction_run_prolate_if_prolate__snk_out_99_in);


// free buffer
}
