"""
crop_cube_spectra.py

Script to crop a RAMPS data cube along
its spectral axis. 

Example:
python crop_cube_spectra.py -i L23_Tile01_23694_MHz_line.fits
                            -o L23_Tile01_23694_MHz_line_cropped.fits
                            -s 800 
                            -e 16000

-i : input  -- The output file name
-o : output -- The output file name
-s : start  -- The new start channel
-e : end    -- The new end channel
-h : Help   -- Display this help

"""

import sys,os,getopt
import astropy.io.fits as pyfits
import numpy as np
import math
import pdb

def channel_to_velocity(channel,h):
    """
    Get the velocity value of a channel.
    """
    velocity = (channel - h['CRPIX3'])*0.001*h['CDELT3'] + h['CRVAL3']
    return velocity

def fix_header(channel_start,channel_end,h):
    """
    Change the header to relected the changed
    spectral axis.
    """
    h['NAXIS3'] = channel_end - channel_start
    h['CRPIX3'] = h['CRPIX3'] - channel_start
    return h

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
    elif o == "-e":
        new_channel_end = int(a)
    elif o == "-h":
        print(__doc__)
        sys.exit(1)
    else:
        assert False, "unhandled option"
        print(__doc__)
        sys.exit(2)

#Load the data
d,h = pyfits.getdata(input_file,header=True)
d = np.squeeze(d)
#Crop the cube along its spectral axis
d_crop = d[new_channel_start:new_channel_end,:,:]
#Change the header and write the cropped cube
h_crop = fix_header(new_channel_start,new_channel_end,h)
pyfits.writeto(output_file,d_crop,h_crop,clobber=True)
