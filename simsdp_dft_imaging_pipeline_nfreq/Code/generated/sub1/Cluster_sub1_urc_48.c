/**
* @file /Cluster_sub1_urc_48.c/h
* @generated by CodegenScape
* @date Tue Dec 10 11:11:37 CET 2024
*/

#include "Cluster_sub1_urc_48.h"

 void Cluster_sub1_urc_48Init(){
}
void Cluster_sub1_urc_48(int2 *delta_psi_3_receiver_pairs_receiver_pairs,int2 *delta_psi_3_delta_correct_to_finegrid_receiver_pairs_if_receiver_pairs){

// buffer declaration

int2 delta_psi_3_receiver_pairs_if_receiver_pairs__delta_psi_3_delta_receiver_pairs_receiver_pairs[130816];

int2 delta_psi_3_delta_receiver_pairs_if_receiver_pairs__delta_psi_3_delta_correct_to_finegrid_receiver_pairs_receiver_pairs[130816];

// body 
//delta_psi_3_receiver_pairs
memcpy(delta_psi_3_receiver_pairs_if_receiver_pairs__delta_psi_3_delta_receiver_pairs_receiver_pairs + 0,delta_psi_3_receiver_pairs_receiver_pairs + 0,130816*sizeof(int2));
//delta_psi_3_delta_receiver_pairs
memcpy(delta_psi_3_delta_receiver_pairs_if_receiver_pairs__delta_psi_3_delta_correct_to_finegrid_receiver_pairs_receiver_pairs + 0,delta_psi_3_receiver_pairs_if_receiver_pairs__delta_psi_3_delta_receiver_pairs_receiver_pairs + 0,130816*sizeof(int2));
//delta_psi_3_delta_correct_to_finegrid_receiver_pairs
memcpy(delta_psi_3_delta_correct_to_finegrid_receiver_pairs_if_receiver_pairs + 0,delta_psi_3_delta_receiver_pairs_if_receiver_pairs__delta_psi_3_delta_correct_to_finegrid_receiver_pairs_receiver_pairs + 0,130816*sizeof(int2));


// free buffer
}
