/**
* @file /Cluster_sub0_urc_25.c/h
* @generated by CodegenScape
* @date Tue Dec 10 11:11:32 CET 2024
*/

#include "Cluster_sub0_urc_25.h"

 void Cluster_sub0_urc_25Init(){
}
void Cluster_sub0_urc_25(PRECISION *delta_psi_prolate_prolate){

// buffer declaration

PRECISION delta_psi_prolate_if_prolate__snk_out_23_in[1229];

// body 
//delta_psi_prolate
memcpy(delta_psi_prolate_if_prolate__snk_out_23_in + 0,delta_psi_prolate_prolate + 0,1229*sizeof(PRECISION));
sub0_snk_out_23(delta_psi_prolate_if_prolate__snk_out_23_in);


// free buffer
}
