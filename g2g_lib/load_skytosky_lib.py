from ctypes import CDLL
import os

def load_lib(precision, architecture):

    if precision == 'single':
        if architecture == 'cpu':
            lib = CDLL('libcpu_skytosky_single.so')
        elif architecture == 'gpu':
            lib = CDLL('libgpu_skytosky_single.so')
        else:
            raise Exception("Error while loading S2S library. Wrong architecture, choose cpu or gpu")
    elif precision == 'double':
        if architecture == 'cpu':
            lib = CDLL('libcpu_skytosky_double.so')
        elif architecture == 'gpu':
            lib = CDLL('libgpu_skytosky_single.so')
        else:
            raise Exception("Error while loading S2S library. Wrong architecture, choose cpu or gpu")
    else:
        raise Exception("Error while loading S2S library. Wrong precision, choose single or double")

    return lib

