import numpy
import sys

import astropy
from astropy.io import fits

def write_nparr_to_fits(data, filename):
    hdu = fits.PrimaryHDU(data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename, overwrite=True)
    hdulist.close()

result = numpy.genfromtxt(sys.argv[1], delimiter=sys.argv[3])[:,:-1]
result = numpy.flip(result)

write_nparr_to_fits(result, sys.argv[2])