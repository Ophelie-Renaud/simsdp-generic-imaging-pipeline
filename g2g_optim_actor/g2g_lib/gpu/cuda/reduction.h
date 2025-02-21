#pragma once

#include "../../common/interpolation_parameters.h"
#include "../../common/polarization_common.h"
#include "../../common/spec_types.h"
#include "../../common/complex_structure.h"
#include "../../common/polarization_vector.h"

__device__ void warpReduce_v4(volatile float* sdata, size_t tid, size_t left_id){
    size_t right_id = 0;
    right_id = 8*(tid + 32);

    sdata[left_id+0] += sdata[right_id+0];
    sdata[left_id+1] += sdata[right_id+1];
    sdata[left_id+2] += sdata[right_id+2];
    sdata[left_id+3] += sdata[right_id+3];
    sdata[left_id+4] += sdata[right_id+4];
    sdata[left_id+5] += sdata[right_id+5];
    sdata[left_id+6] += sdata[right_id+6];
    sdata[left_id+7] += sdata[right_id+7];

    right_id = 8*(tid + 16);
    sdata[left_id+0] += sdata[right_id+0];
    sdata[left_id+1] += sdata[right_id+1];
    sdata[left_id+2] += sdata[right_id+2];
    sdata[left_id+3] += sdata[right_id+3];
    sdata[left_id+4] += sdata[right_id+4];
    sdata[left_id+5] += sdata[right_id+5];
    sdata[left_id+6] += sdata[right_id+6];
    sdata[left_id+7] += sdata[right_id+7];

    right_id = 8*(tid + 8);
    sdata[left_id+0] += sdata[right_id+0];
    sdata[left_id+1] += sdata[right_id+1];
    sdata[left_id+2] += sdata[right_id+2];
    sdata[left_id+3] += sdata[right_id+3];
    sdata[left_id+4] += sdata[right_id+4];
    sdata[left_id+5] += sdata[right_id+5];
    sdata[left_id+6] += sdata[right_id+6];
    sdata[left_id+7] += sdata[right_id+7];

    right_id = 8*(tid + 4);
    sdata[left_id+0] += sdata[right_id+0];
    sdata[left_id+1] += sdata[right_id+1];
    sdata[left_id+2] += sdata[right_id+2];
    sdata[left_id+3] += sdata[right_id+3];
    sdata[left_id+4] += sdata[right_id+4];
    sdata[left_id+5] += sdata[right_id+5];
    sdata[left_id+6] += sdata[right_id+6];
    sdata[left_id+7] += sdata[right_id+7];

    right_id = 8*(tid + 2);
    sdata[left_id+0] += sdata[right_id+0];
    sdata[left_id+1] += sdata[right_id+1];
    sdata[left_id+2] += sdata[right_id+2];
    sdata[left_id+3] += sdata[right_id+3];
    sdata[left_id+4] += sdata[right_id+4];
    sdata[left_id+5] += sdata[right_id+5];
    sdata[left_id+6] += sdata[right_id+6];
    sdata[left_id+7] += sdata[right_id+7];

    right_id = 8*(tid + 1);
    sdata[left_id+0] += sdata[right_id+0];
    sdata[left_id+1] += sdata[right_id+1];
    sdata[left_id+2] += sdata[right_id+2];
    sdata[left_id+3] += sdata[right_id+3];
    sdata[left_id+4] += sdata[right_id+4];
    sdata[left_id+5] += sdata[right_id+5];
    sdata[left_id+6] += sdata[right_id+6];
    sdata[left_id+7] += sdata[right_id+7];
}

template<unsigned int blockSize>
__device__ void warpReduce_v5(float* sdata, size_t tid, size_t left_id){
    size_t right_id = 0;
    if(blockSize >= 64){
        right_id = 8*(tid + 32);
        sdata[left_id+0] += sdata[right_id+0];
        sdata[left_id+1] += sdata[right_id+1];
        sdata[left_id+2] += sdata[right_id+2];
        sdata[left_id+3] += sdata[right_id+3];
        sdata[left_id+4] += sdata[right_id+4];
        sdata[left_id+5] += sdata[right_id+5];
        sdata[left_id+6] += sdata[right_id+6];
        sdata[left_id+7] += sdata[right_id+7];
    }
    if(blockSize >= 32){
        right_id = 8*(tid + 16);
        sdata[left_id+0] += sdata[right_id+0];
        sdata[left_id+1] += sdata[right_id+1];
        sdata[left_id+2] += sdata[right_id+2];
        sdata[left_id+3] += sdata[right_id+3];
        sdata[left_id+4] += sdata[right_id+4];
        sdata[left_id+5] += sdata[right_id+5];
        sdata[left_id+6] += sdata[right_id+6];
        sdata[left_id+7] += sdata[right_id+7];
    }
    if(blockSize >= 16){
        right_id = 8*(tid + 8);
        sdata[left_id+0] += sdata[right_id+0];
        sdata[left_id+1] += sdata[right_id+1];
        sdata[left_id+2] += sdata[right_id+2];
        sdata[left_id+3] += sdata[right_id+3];
        sdata[left_id+4] += sdata[right_id+4];
        sdata[left_id+5] += sdata[right_id+5];
        sdata[left_id+6] += sdata[right_id+6];
        sdata[left_id+7] += sdata[right_id+7];
    }
    if(blockSize >= 8){
        right_id = 8*(tid + 4);
        sdata[left_id+0] += sdata[right_id+0];
        sdata[left_id+1] += sdata[right_id+1];
        sdata[left_id+2] += sdata[right_id+2];
        sdata[left_id+3] += sdata[right_id+3];
        sdata[left_id+4] += sdata[right_id+4];
        sdata[left_id+5] += sdata[right_id+5];
        sdata[left_id+6] += sdata[right_id+6];
        sdata[left_id+7] += sdata[right_id+7];
    }
    if(blockSize >= 4){ 
        right_id = 8*(tid + 2);
        sdata[left_id+0] += sdata[right_id+0];
        sdata[left_id+1] += sdata[right_id+1];
        sdata[left_id+2] += sdata[right_id+2];
        sdata[left_id+3] += sdata[right_id+3];
        sdata[left_id+4] += sdata[right_id+4];
        sdata[left_id+5] += sdata[right_id+5];
        sdata[left_id+6] += sdata[right_id+6];
        sdata[left_id+7] += sdata[right_id+7];
    }
    if(blockSize >= 2){ 
        right_id = 8*(tid + 1);
        sdata[left_id+0] += sdata[right_id+0];
        sdata[left_id+1] += sdata[right_id+1];
        sdata[left_id+2] += sdata[right_id+2];
        sdata[left_id+3] += sdata[right_id+3];
        sdata[left_id+4] += sdata[right_id+4];
        sdata[left_id+5] += sdata[right_id+5];
        sdata[left_id+6] += sdata[right_id+6];
        sdata[left_id+7] += sdata[right_id+7];
    }
}



