//
// Created by orenaud on 4/14/25.
//

#ifndef CASACORE_WRAPPER_H
#define CASACORE_WRAPPER_H



#ifdef __cplusplus
extern "C" {
#endif

#include "common.h"


    void load_visibilities_from_ms(const char* ms_path, int num_vis,
                                   Config* config, Visibility* uvw_coords, Complex* measured_vis);
    void psf_host_set_up_ms(int GRID_SIZE, int PSF_GRID_SIZE, Config *config, PRECISION *psf, double *psf_max_value) ;

#ifdef __cplusplus
}
#endif

#endif //CASACORE_WRAPPER_H
