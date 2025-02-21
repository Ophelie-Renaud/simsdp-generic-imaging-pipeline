

import numpy as np
from ctypes import CDLL, c_double, c_int, c_void_p, c_float, c_bool, c_size_t


"""
global cell_size_type
cell_size_type = None
global cell_scale_type
cell_scale_type = None
global freq_type
freq_type = None
global visibility_type
visibility_type = None
global weight_type
weight_type = None
global uvw_coordinates_type
uvw_coordinates_type = None
"""
def set_precision(precision):
    global cell_scale_type
    global conv_function_type
    if precision == 'single':
        cell_size_type                  = c_float#np.float32
        cell_scale_type                 = c_float#np.float32
        freq_type                       = c_float#np.float32
        visibility_type                 = c_float#np.float32
        weight_type                     = c_float#np.float32
        uvw_coordinates_type            = c_float#np.float32
        conv_function_type              = c_float#np.float32
        norm_weight_type                = c_float#np.float32
    elif precision == 'double':
        cell_size_type                  = c_double#np.float64
        cell_scale_type                 = c_double#np.float64
        freq_type                       = c_double#np.float64
        visibility_type                 = c_double#np.float64
        weight_type                     = c_double#np.float64
        uvw_coordinates_type            = c_double#np.float64
        conv_function_type              = c_double#np.float64
        norm_weight_type                = c_double#np.float64
    else:
        raise Exception("Warning : Set precision in spec_type.py is not single nor double. Check paramater")

