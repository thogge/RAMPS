#!/usr/bin/env python
# encoding: utf-8
"""
crop_cube.py

Crops the nans from the top and bottom of a cube, 
leaving a rectangular shaped cube with only 
finite values. This is meant to crop partial tiles
from RAMPS and is not suited to crop pixels in 
the x direction. Also, can crop along the
spectral axis by giving the new first and last
channel.

Example:
python crop_cube.py -i L30_Tile01.fits
                    -o L30_Tile01_cropped.fits
                    -s 800 
                    -e 16000



-i : Input      -- Input file (reduced by pipeline)
-o : Output     -- Output file
-s : start      -- The new start channel
-e : end        -- The new end channel
-h : Help       -- Display this help

"""

import sys,os,getopt
try:
    import astropy.io.fits as pyfits
except:
    import pyfits
import numpy as np
from spectral_cube import SpectralCube
import pyspeckit

def main():
    output_file = "default.fits"
    spec_crop = False
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:o:s:e:h")
    except getopt.GetoptError,err:
        print(str(err))
        print(__doc__)
        sys.exit(2)
    for o,a in opts:
        if o == "-i":
            input_file = a
        elif o == "-o":
            output_file = a
        elif o == "-s":
            new_channel_start = int(a)
            spec_crop = True
        elif o == "-e":
            new_channel_end = int(a)
            spec_crop = True
        elif o == "-h":
            print(__doc__)
            sys.exit(1)
        else:
            assert False, "unhandled option"
            print(__doc__)
            sys.exit(2)

    """
    Read the data, remove the extra axis if necessary.
    """
    d,h = pyfits.getdata(input_file,header=True)
    temp_file = input_file[:-5]+"_temp.fits"
    if len(d.shape) > 3:
        d = np.squeeze(d)
        h = strip_header(h,4)
    pyfits.writeto(temp_file,d,h,clobber=True)
    """
    Crop the cube along its spectral axis. Fix the
    header and write to temporary file.
    """
    if spec_crop:
        d_crop = d[new_channel_start:new_channel_end,:,:]
        h_crop = fix_header(new_channel_start,new_channel_end,h)
        pyfits.writeto(temp_file,d_crop,h_crop,clobber=True)
        d = d_crop
        h = h_crop

    """
    Find locations where the entire row (in x direction)
    is finite valued.
    """
    ylist = []
    for i in np.arange(d.shape[1]):
        ylist.append(np.isfinite(np.sum(d[:,i,:])))
    finite_y_loc = np.where(ylist)

    """
    Determine the new center pixels for the cube, as well
    as the new height. The width will be the same, since
    there shouldn't be any nans on the left or right side
    of the cube.
    """
    y_min = np.min(finite_y_loc[0])
    y_max = np.max(finite_y_loc[0])
    y_center = round(np.mean(finite_y_loc[0]))
    x_center = round(d.shape[2]/2.)
    y_halfwidth = round((y_max-y_min)/2.)
    x_halfwidth = x_center

    """
    Use Pyspeckit to crop the cube spatially and write it to the output file.
    """
    cube = SpectralCube.read(temp_file)
    os.system('rm ' + temp_file)
    new_cube = pyspeckit.cubes.subcube(cube,x_center,x_halfwidth,y_center,y_halfwidth)
    new_cube.write(output_file,format='fits',overwrite=True)

    """
    Fix the header info that Pyspeckit changed. Should probably 
    figure out how to stop it from doing that. 
    Also, check that all of the nans were removed.
    """
    d,h = pyfits.getdata(output_file,header=True)
    if np.isnan(np.sum(d)):
        print 'Not all nans removed from ' + input_file
    h['CTYPE3'] = 'VELO-LSR'
    pyfits.writeto(output_file,d,h,clobber=True)


def strip_header(h,n):
    """
    Remove the nth dimension from a FITS header
    """
    h['NAXIS'] = n-1
    try:
        del h['NAXIS'+str(n)]
        del h['CTYPE'+str(n)]
        del h['CRVAL'+str(n)]
        del h['CDELT'+str(n)]
        del h['CRPIX'+str(n)]
    except:
        pass
    return(h)

def fix_header(channel_start,channel_end,h):
    """
    Change the header to relected the changed
    spectral axis.
    """
    h['NAXIS3'] = channel_end - channel_start
    h['CRPIX3'] = h['CRPIX3'] - channel_start
    return h


if __name__ == '__main__':
    main()

