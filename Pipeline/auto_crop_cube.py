#!/usr/bin/env python
# encoding: utf-8
"""
auto_crop_cube.py

Crops the nans from the top and bottom of a cube, 
leaving a rectangular shaped cube with only 
finite values. This is meant to crop partial tiles
from RAMPS and is not suited to crop pixels in 
the x direction. 

Example:
python aut_crop_cube.py
       -i L30_Tile01.fits
       -o L30_Tile01_cropped.fits
       -h help


-i : Input      -- Input file (reduced by pipeline)
-o : Output     -- Output file
-h : Help       -- Display this help

"""

import sys,os,getopt
import pdb
try:
    import astropy.io.fits as pyfits
except:
    import pyfits
import numpy as np
import pyspeckit
from spectral_cube import SpectralCube
import matplotlib.pyplot as plt

def main():
    output_file = "default.fits"
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:o:h")
    except getopt.GetoptError,err:
        print(str(err))
        print(__doc__)
        sys.exit(2)
    for o,a in opts:
        if o == "-i":
            input_file = a
        elif o == "-o":
            output_file = a
        elif o == "-h":
            print(__doc__)
            sys.exit(1)
        else:
            assert False, "unhandled option"
            print(__doc__)
            sys.exit(2)


    d,h = pyfits.getdata(input_file,header=True)
    d = np.squeeze(d)

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
    Use Pyspeckit to crop the cube and write it to the output file.
    """
    cube = SpectralCube.read(input_file)
    new_cube = pyspeckit.cubes.subcube(cube,x_center,x_halfwidth,y_center,y_halfwidth)
    new_cube.write(output_file,format='fits',overwrite=True)

    d,h = pyfits.getdata(output_file,header=True)
    if np.isnan(np.sum(d)):
        print 'Not all nans removed from ' + input_file
    h['CTYPE3'] = 'VELO-LSR'
    pyfits.writeto(output_file,d,h,clobber=True)


if __name__ == '__main__':
    main()

