import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

lightspeed = 299792458.

def _gen_coeffs(x, y, order):
    ncols = (order+1)**2
    coeffs = np.empty((x.size, ncols), dtype=x.dtype)
    c = 0

    for i in range(order+1):
        for j in range(order+1):
            for k in range(x.size):
                coeffs[k, c] = x[k]**i * y[k]**j

            c += 1

    return coeffs

def polyfit2d(x, y, z, order=3):
    """
    Given ``x`` and ``y`` data points and ``z``, some
    values related to ``x`` and ``y``, fit a polynomial
    of order ``order`` to ``z``.

    Derived from https://stackoverflow.com/a/7997925
    """
    return np.linalg.lstsq(_gen_coeffs(x, y, order), z, rcond=None)[0]


#@numba.jit(nopython=True, nogil=True, cache=True)
def polyval2d(x, y, coeffs):
    """
    Reproduce values from a two-dimensional polynomial fit.

    Derived from https://stackoverflow.com/a/7997925
    """
    order = int(np.sqrt(coeffs.size)) - 1
    z = np.zeros_like(x)
    c = 0

    for i in range(order+1):
        for j in range(order+1):
            a = coeffs[c]
            for k in range(x.shape[0]):
                z[k] += a * x[k]**i * y[k]**j

            c += 1

    return z

P = np.array([
    [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
    [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])

Q = np.array([
    [1.0000000e0, 8.212018e-1, 2.078043e-1],
    [1.0000000e0, 9.599102e-1, 2.918724e-1]])

def spheroidal_2d(npix, factor=1.0):
    result = np.empty((npix, npix), dtype=np.float32)
    c = np.linspace(-1.0, 1.0, npix)

    for y, yc in enumerate(c):
        y_sqrd = yc**2
        for x, xc in enumerate(c):
            r = np.sqrt(xc**2 + y_sqrd)*factor

            if r >= 0.0 and r < 0.75:
                poly = 0
                end = 0.75
            elif r >= 0.75 and r <= 1.00:
                poly = 1
                end = 1.00
            else:
                result[y, x] = 0.0
                continue

            sP = P[poly]
            sQ = Q[poly]

            nu_sqrd = r**2
            del_nu_sqrd = nu_sqrd - end*end

            top = sP[0]
            del_nu_sqrd_pow = del_nu_sqrd

            for i in range(1, 5):
                top += sP[i]*del_nu_sqrd_pow
                del_nu_sqrd_pow *= del_nu_sqrd

            bot = sQ[0]
            del_nu_sqrd_pow = del_nu_sqrd

            for i in range(1, 3):
                bot += sQ[i]*del_nu_sqrd_pow
                del_nu_sqrd_pow *= del_nu_sqrd

            result[y, x] = (1.0 - nu_sqrd) * (top/bot)

    return result


def zero_pad(img, npix):
    """ Zero pad ``img`` up to ``npix`` """

    if isinstance(npix, int):
#        print("Zero pad correctly isinstance")
        npix = (npix,)*img.ndim

    padding = []

#    print(npix, img.shape)
    for dim, npix_ in zip(img.shape, npix):
        # Pad and half-pad amount
        p = npix_ - dim
        hp = p // 2

        # Pad the imagew
        padding.append((hp, hp) if p % 2 == 0 else (hp+1, hp))

    return np.pad(img, padding, 'constant', constant_values=0)

def spheroidal_aa_filter(npix, support=11, spheroidal_support=111):
    # Convolution filter
    cf = spheroidal_2d(spheroidal_support).astype(np.complex128)
    # Fourier transformed convolution filter
    fcf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cf)))

    # Cut the support out
    xc = spheroidal_support//2
    start = xc-support//2
    end = 1 + xc+support//2
    
    fcf = fcf[start:end, start:end].copy()

    # Inverse fourier transform of the cut
    # if_cut_fcf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fcf)))

    # Pad and ifft2 the fourier transformed convolution filter
#    print("Parameters to zero pad =", fcf.shape, npix)
    zfcf = zero_pad(fcf, int(npix))
    ifzfcf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(zfcf)))
    
    


    return cf, fcf, ifzfcf


def find_max_support(radius, maxw, min_wave):
    """
    Find the maximum support
    """
    # Assumed maximum support
    max_support = 501

    # Work out the spheroidal convolution filter for
    # the maximum support size
#    print("Max support = ", max_support)
    _, _, spheroidal_w = spheroidal_aa_filter(max_support)

    # Compute l, m and n-1 over the area of maximum support
    ex = radius*np.sqrt(2.)
    l, m = np.mgrid[-ex:ex:max_support*1j, -ex:ex:max_support*1j]
    n_1 = np.sqrt(1.0 - l**2 - m ** 2) - 1.0

    # Compute the w term
    w = np.exp(-2.0*1j*np.pi*(maxw/min_wave)*n_1)*spheroidal_w
    fw = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(w)))

    # Want to interpolate across fw. fw is symmetric
    # so take a slice across fw at the halfway point
    fw1d = np.abs(fw[(max_support-1)//2, :])
    # normalise
    fw1d /= np.max(fw1d)
    # Then take half again due to symmetry
    fw1d = fw1d[(max_support-1)//2::]

    ind = np.argsort(fw1d)

    from scipy.interpolate import interp1d

    # TODO(sjperkins)
    # Find a less clunky way to find the maximum support
    interp_fn = interp1d(fw1d[ind], np.arange(fw1d.shape[0])[ind])
    max_support = interp_fn(1./1000)

#    print("Local max support =", max_support)

    return max_support


def delta_n_coefficients(l0, m0, radius=1., order=4):
    """
    Returns polynomical coefficients representing the difference
    of coordinate n between a grid of (l,m) values centred
    around (l0, m0) and (l0, m0).
    """

    Np = 100

    l, m = np.mgrid[l0-radius:l0+radius:Np*1j, m0-radius:m0+radius:Np*1j]

    dl = l-l0
    dm = m-m0

    dl = dl.flatten()
    dm = dm.flatten()
    y = np.sqrt(1-(dl+l0)**2-(dm+m0)**2)-np.sqrt(1-l0**2-m0**2)
    coeff = polyfit2d(dl, dm, y, order=order)
    C = coeff.reshape((order+1, order+1))
    Cl = C[0, 1]
    Cm = C[1, 0]
    C[0, 1] = 0
    C[1, 0] = 0

    return Cl, Cm, coeff



def reorganise_convolution_filter(cf, oversampling):
    """
    TODO(sjperkins)
    Understand what's going on here...

    Parameters
    ----------
    cf : np.ndarray
        Oversampled convolution filter
    oversampling : integer
        Oversampling factor

    Returns
    -------
    np.ndarray
        Reorganised convolution filter for more efficient memory access

    """
    support = cf.shape[0]//oversampling
    result = np.empty((oversampling, oversampling, support, support),
                      dtype=cf.dtype)

    for i in range(oversampling):
        for j in range(oversampling):
            result[i, j, :, :] = cf[i::oversampling, j::oversampling]

    return result.reshape(cf.shape)


def deapodization(Nx, Ny, filt, K):
#    filter_2D = np.outer(self.filter_taps.astype(float), self.filter_taps.astype(float))
    pad = np.pad(filt, ((Nx*K)//2 - len(filt)//2, (Ny*K)//2-len(filt)//2))
    pad_shift = np.fft.ifftshift(pad)
    pad_im = np.fft.ifft2(pad_shift)
    
    correction = np.real(np.fft.fftshift(pad_im))[(Nx*K)//2 - Nx//2:(Nx*K)//2 + Nx//2,
                                                  (Ny*K)//2 - Ny//2:(Ny*K)//2 + Ny//2]
    
    return correction

def compute_kernels(wplanes, cell_size_arcsec, half_support, max_w, grid_size, oversampling_factor, min_freq, isotropic):
    support = half_support * 2
    radius_radians = np.pi * ((cell_size_arcsec / 3600) * (grid_size / 2)) / 180
    
    min_wavelength = lightspeed / min_freq
    max_w_support = find_max_support(radius_radians, max_w, min_wavelength)
    w_values = np.linspace(0, max_w, wplanes)
    w_supports = np.linspace(support, max(max_w_support, support), wplanes, dtype=np.int64)
    w_supports[w_supports % 2 == 0] += 1

    _, _, poly_coeffs = delta_n_coefficients(0, 0, 3 * radius_radians, order=5)


    wplanes = []
    wplanes_conj = []
    sphere = []

    for i, (w_value, w_support) in enumerate(zip(w_values, w_supports)):
        norm_w_value = w_value / min_wavelength
        
        _, _, spheroidal = spheroidal_aa_filter(w_support, support=w_support)
        
        ex = radius_radians - radius_radians/w_support
        l, m = np.mgrid[-ex:ex:w_support*1j, -ex:ex:w_support*1j]
        n_1 = polyval2d(l, m, poly_coeffs)
        w = np.exp(-2.0*1j*np.pi*norm_w_value*n_1)*np.abs(spheroidal)
        
        # zero pad w, adding oversampling
        zw = zero_pad(w, w.shape[0]*oversampling_factor)
        #zw /= np.ndarray.sum(zw)#np.linalg.norm(zw)
        zw_conj = np.conj(zw)

        fzw = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(zw)))
        fzw_conj = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(zw_conj)))
        
        taps = np.linspace(-1.0, 1.0, w.shape[0]*oversampling_factor)
        sinc = np.sinc(taps)
        sinc2D = np.outer(sinc, sinc)
        #fzw *= sinc2D

        fzw = np.require(fzw.astype(np.complex64),requirements=["A", "C"])
        fzw_conj = np.require(fzw_conj.astype(np.complex64),requirements=["A", "C"])

        if isotropic:
            spheroidal = spheroidal[half_support * oversampling_factor:(2 * half_support + 1) * oversampling_factor, half_support * oversampling_factor:(2 * half_support + 1) * oversampling_factor].copy()
            fzw = fzw[half_support * oversampling_factor:(2 * half_support + 1) * oversampling_factor, half_support * oversampling_factor:(2 * half_support + 1) * oversampling_factor].copy()
            fzw_conj = fzw_conj[half_support * oversampling_factor:(2 * half_support + 1) * oversampling_factor, half_support * oversampling_factor:(2 * half_support + 1) * oversampling_factor].copy()
            #spheroidal = spheroidal[0:(half_support + 1) * oversampling_factor, 0:(half_support + 1) * oversampling_factor].copy()
            #fzw = fzw[0:(half_support + 1) * oversampling_factor, 0:(half_support + 1) * oversampling_factor].copy()
            #fzw_conj = fzw_conj[0:(half_support + 1) * oversampling_factor, 0:(half_support + 1) * oversampling_factor].copy()

        wplanes.append(fzw)
        wplanes_conj.append(fzw_conj)
        sphere.append(spheroidal)

    return wplanes, wplanes_conj, sphere, w_supports



def write_kernels_to_csv(gridding_kernels, degridding_kernels, supports_g, supports_dg, filename, oversampling_factor):
    dk_filename_r = filename + "_degridding_kernels_real_x" + str(oversampling_factor) + ".csv"
    dk_filename_i = filename + "_degridding_kernels_imag_x" + str(oversampling_factor) + ".csv"
    gk_filename_r = filename + "_gridding_kernels_real_x" + str(oversampling_factor) + ".csv"
    gk_filename_i = filename + "_gridding_kernels_imag_x" + str(oversampling_factor) + ".csv"
    ds_filename = filename + "_degridding_kernel_supports_x" + str(oversampling_factor) + ".csv"
    gs_filename = filename + "_gridding_kernel_supports_x" + str(oversampling_factor) + ".csv"


    #write supports
    f_gs = open(gs_filename, 'w')
    f_dgs = open(ds_filename, 'w')
    gs_writer = csv.writer(f_gs, delimiter = '\n')
    dgs_writer = csv.writer(f_dgs, delimiter = '\n')

    gs_writer.writerow(supports_g)
    dgs_writer.writerow(supports_dg)

    f_gs.close()
    f_dgs.close()

    g_kernels_flat_r = []
    g_kernels_flat_i = []

    dg_kernels_flat_r = []
    dg_kernels_flat_i = []

    #flatten kernels to easily write to files, also separates real and imaginary components into seperate files because this is what SEP expects
    #assumes tuples are (real, imag)
    total_gridding_samples = 0
    total_degridding_samples = 0

    for g_kernel in gridding_kernels:
        flat_g_kernel = np.ndarray.flatten(g_kernel)
        total_gridding_samples += len(flat_g_kernel)
        g_kernels_flat_r += [v.real for v in flat_g_kernel]
        g_kernels_flat_i += [v.imag for v in flat_g_kernel]

    for dg_kernel in degridding_kernels:
        flat_dg_kernel = np.ndarray.flatten(dg_kernel)
        total_degridding_samples += len(flat_dg_kernel)
        dg_kernels_flat_r += [v.real for v in flat_dg_kernel]
        dg_kernels_flat_i += [v.imag for v in flat_dg_kernel]

    print(str(total_gridding_samples) + " " + str(total_degridding_samples))

    f_gi = open(gk_filename_i, 'w')
    f_gr = open(gk_filename_r, 'w')
    f_dgi = open(dk_filename_i, 'w')
    f_dgr = open(dk_filename_r, 'w')

    ki_writer = csv.writer(f_gi, delimiter = ' ')
    ki_writer.writerow(g_kernels_flat_i)

    kr_writer = csv.writer(f_gr, delimiter = ' ')
    kr_writer.writerow(g_kernels_flat_r)

    dki_writer = csv.writer(f_dgi, delimiter = ' ')
    dki_writer.writerow(dg_kernels_flat_i)

    dkr_writer = csv.writer(f_dgr, delimiter = ' ')
    dkr_writer.writerow(dg_kernels_flat_r)

    f_gi.close()
    f_gr.close()
    f_dgr.close()
    f_dgi.close()

def read_gridding_kernels_and_conj(support_filename, real_filename, imag_filename, sep, oversampling_factor):
    supports = np.loadtxt(support_filename, dtype=int, unpack=True)
    reals = np.ndarray.flatten(np.genfromtxt(real_filename, delimiter=sep))
    imags = np.ndarray.flatten(np.genfromtxt(imag_filename, delimiter=sep))

    output_gridding_kernels = []
    output_degridding_kernels = []
    full_gkernels_r = []
    full_gkernels_i = []
    full_dgkernels_r = []
    full_dgkernels_i = []
    offset = 0

    for i, support in enumerate(supports):
        dim = (support + 1) * oversampling_factor
        num_data = dim * dim

        gridding_kernel = 1j * imags[offset:offset + num_data]
        gridding_kernel += reals[offset:offset + num_data]
        gridding_kernel = np.reshape(gridding_kernel, (dim, dim))

        #gk = gridding_kernel.copy()

        mirrored_gridding_kernel = np.zeros((2 * dim, 2 * dim), dtype=np.complex64)
        mirrored_gridding_kernel[dim: 2 * dim, dim: 2 * dim] = gridding_kernel
        mirrored_gridding_kernel[0:dim, 0:dim] = np.flip(gridding_kernel, (0, 1))
        mirrored_gridding_kernel[0:dim, dim: 2 * dim] = np.flip(gridding_kernel, 0)
        mirrored_gridding_kernel[dim: 2 * dim, 0:dim] = np.flip(gridding_kernel, 1)

        smgk = np.fft.ifft2(np.fft.ifftshift(mirrored_gridding_kernel))
        smdgk = np.conj(smgk)

        dgk = np.fft.fftshift(np.fft.fft2(smdgk))
        mdgk = dgk.copy()

        #dgk = gridding_kernel

        dgk = dgk[dim: 2*dim, dim: 2*dim]

        output_gridding_kernels.append(gridding_kernel)
        output_degridding_kernels.append(dgk)

        offset = offset + num_data

        g_real = np.real(mirrored_gridding_kernel)
        g_imag = np.imag(mirrored_gridding_kernel)
        dg_real = np.real(mdgk)
        dg_imag = np.imag(mdgk)

        full_gkernels_r.append(g_real)
        full_gkernels_i.append(g_imag)
        full_dgkernels_r.append(dg_real)
        full_dgkernels_i.append(dg_imag)

    return output_gridding_kernels, output_degridding_kernels, supports, full_gkernels_r, full_gkernels_i, full_dgkernels_r, full_dgkernels_i

def print_kernels(greals, gimags, dgreals, dgimags):
    for i, k in enumerate(greals):
        gkernel_filename_r = "gkernel_test_r_" + str(i) + ".csv"
        gkernel_filename_i = "gkernel_test_i_" + str(i) + ".csv"
        dgkernel_filename_r = "dgkernel_test_r_" + str(i) + ".csv"
        dgkernel_filename_i = "dgkernel_test_i_" + str(i) + ".csv"

        f_gkr = open(gkernel_filename_r, 'w')
        f_gki = open(gkernel_filename_i, 'w')
        f_dgkr = open(dgkernel_filename_r, 'w')
        f_dgki = open(dgkernel_filename_i, 'w')
        
        gkwr = csv.writer(f_gkr, delimiter = ',')
        gkwi = csv.writer(f_gki, delimiter = ',')
        dgkwr = csv.writer(f_dgkr, delimiter = ',')
        dgkwi = csv.writer(f_dgki, delimiter = ',')

        rows = k.shape[0]

        for j in range(rows):
            gkwr.writerow(greals[i][j])
            gkwi.writerow(gimags[i][j])
            dgkwr.writerow(dgreals[i][j])
            dgkwi.writerow(dgimags[i][j])

        f_gkr.close()
        f_gki.close()
        f_dgkr.close()
        f_dgki.close()


#wplanes, wplanes_conj, sphere, w_supports = compute_kernels(17, 1.7578, 4, 1900, 2048, 16, 0.14e9, True)
#w_supports = [v // 2 for v in w_supports]

#write_kernels_to_csv(wplanes, wplanes_conj, w_supports, w_supports, "input/kernels/new/wproj_spheroidalaa", 16)

support_filename = sys.argv[1]
real_filename = sys.argv[2]
imag_filename = sys.argv[3]

wplanes, wplanes_conj, w_supports, gr, gi, dgr, dgi = read_gridding_kernels_and_conj(support_filename, real_filename, imag_filename, ' ', 16)
print_kernels(gr, gi, dgr, dgi)

write_kernels_to_csv(wplanes, wplanes_conj, w_supports, w_supports, "input/kernels/new/wproj_manualconj", 16)