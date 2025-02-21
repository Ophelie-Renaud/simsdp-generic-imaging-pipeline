

#from moris.gridder.optim import spec_types
import numpy as np
import moris.gridder.optim.spec_types as spec_types
from ctypes import CDLL, c_double, c_int, c_void_p, c_float, c_bool, c_size_t, Structure, byref, POINTER, cast, c_ulong, c_long



class interpolation_parameters(Structure):
    _fields_ = [
        # Atennas specs
        ("nrows", c_size_t),
        # Observation specs
        ("Nx", c_size_t),
        ("half_Nx", c_size_t),
        ("Ny", c_size_t),
        ("half_Ny", c_size_t),
        ("Nchan", c_size_t),
        ("spw_selected", c_size_t),
        ("oversampling_factor", c_size_t),
        ("half_support_function", c_size_t),
        ("full_support_function", c_size_t),
        ("nb_w_planes", c_size_t),
        ("nb_vis_polarization", c_size_t),
        ("nb_grid_polarization", c_size_t),
        ("grid_channel_idx", c_void_p),
        ("no_grid_index", c_size_t),
        ("grid_channel_width", c_size_t),
        ("no_chan_spw", c_size_t), # Number of nb_channels x nb_spw
        ("nb_grid_chan", c_size_t),
        ("polarization_step", c_size_t),
        ("cell_size_l", c_double),#spec_types.cell_size_type),
        ("cell_size_m", c_double),#spec_types.cell_size_type),
        ("u_scale", spec_types.cell_scale_type),
        ("v_scale", spec_types.cell_scale_type),
        # S2S
        ("do_s2s", c_size_t),
        ("len_s2s_coo", c_void_p),
        ("a_coo", c_void_p),
        ("u_coo", c_void_p),
        ("v_coo", c_void_p),
        ("ch_coo", c_void_p),
        ("uvach_coo", c_void_p),
        ("tot_size_coo", c_size_t),

        # Frequencies
        ("chan_wavelength", c_void_p),
        # Grids
        ("input_grid", c_void_p),
        ("output_grid", c_void_p),
        ("psf_grid", c_void_p),
        # Data specs
        #("flags", c_void_p),
        #("data_desc_id", c_void_p),
        ("visibilities", c_void_p),
        ("visibility_weight", c_void_p),
        ("uvw_coordinates", c_void_p),
        # Convolution functions specs
        ("gridding_conv_function", c_void_p),
        ("filter_size", c_size_t),
        ("filter_AA_2D", c_void_p),
        ("filter_AA_2D_size", c_size_t),
        ("filter_choice", c_size_t),
        ("max_w", c_float),
        ("conv_norm_weight", c_void_p)
        # W-proj Convolution functions specs
        ]

    def __init__(self, 
                 ms_handler, 
                 Nx, 
                 Ny, 
                 cell_size_l, 
                 cell_size_m, 
                 spw_selected, 
                 no_grid_index, 
                 grid_channel_idx, 
                 grid_channel_width):
        
        self.nrows = ms_handler.nrows
        self.Nx = Nx
        self.half_Nx = Nx//2
        self.Ny = Ny
        self.half_Ny = Ny//2
        self.Nchan = ms_handler.nb_chan
        self.cell_size_l = c_double(np.float64(cell_size_l))
        self.cell_size_m = c_double(np.float64(cell_size_m))
        self.spw_selected = spw_selected
        self.no_grid_index = no_grid_index
        self.no_chan_spw = np.int32(ms_handler.chan_freq.shape[0]*ms_handler.chan_freq.shape[1])
        self.grid_channel_idx = grid_channel_idx.ctypes.data_as(c_void_p)
        self.grid_channel_width = grid_channel_width


    def s2s_param(self, a_coo_p, u_coo_p, v_coo_p, ch_coo_p, uvach_coo_p):
        self.a_coo = a_coo_p
        self.u_coo = u_coo_p
        self.v_coo = v_coo_p
        self.ch_coo = ch_coo_p
        self.uvach_coo = uvach_coo_p
        self.tot_size_coo = 0


    def fill_C_structure(self, 
                         output_grid, 
                         psf_grid, 
                         input_grid, 
                         chan_wavelength, 
                         visi, 
                         visibility_weight, 
                         uvw, 
                         filter_AA_2D, 
                         filter_choice, 
                         conv_norm_weight, 
                         oversampling_factor, 
                         half_support_filter, 
                         nb_vis_polarization, 
                         nb_grid_polarization, 
                         pola_step, 
                         ms_handler, 
                         grid_channel_idx) :
        self.do_s2s = 0
        self.output_grid = output_grid.ctypes.data_as(c_void_p)
        self.psf_grid = psf_grid.ctypes.data_as(c_void_p)
        self.input_grid = input_grid.ctypes.data_as(c_void_p)
        self.chan_wavelength = chan_wavelength.ctypes.data_as(c_void_p)
        self.visibilities = visi.ctypes.data_as(c_void_p)
        self.visibility_weight = visibility_weight.ctypes.data_as(c_void_p)
        self.uvw_coordinates = uvw.ctypes.data_as(c_void_p)
        self.gridding_conv_function = filter_AA_2D.ctypes.data_as(c_void_p)
        self.filter_size = filter_AA_2D.shape[0]
        self.filter_AA_2D = filter_AA_2D.ctypes.data_as(c_void_p)
        self.filter_AA_2D_size = filter_AA_2D.shape[0]
        self.filter_choice = filter_choice
        self.conv_norm_weight = conv_norm_weight.ctypes.data_as(c_void_p)
        self.oversampling_factor = oversampling_factor
        self.half_support_function = half_support_filter
        self.full_support_function = half_support_filter*2+1
        self.nb_vis_polarization = nb_vis_polarization
        self.nb_grid_polarization = nb_grid_polarization
        self.polarization_step = pola_step
        self.max_w = np.max(np.abs(ms_handler.uvw[:,2]))
        self.grid_channel_idx = grid_channel_idx.ctypes.data_as(c_void_p)
