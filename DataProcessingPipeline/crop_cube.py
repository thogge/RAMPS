#!/usr/bin/env python
# encoding: utf-8
"""
crop_cube.py

Perform spatial cropping on 3D data cube.
Supports use of world coordinates.

Example:
python crop_cube.py -i L30_Tile01.fits
                    -o L30_Tile01_cropped.fits
                    -x 10. 
                    -y 0.
                    -c 0.2
                    -d 0.2
                    -w 


-i : Input      -- Input file (reduced by pipeline)
-o : Output     -- Output file
-x : xcenter    -- X axis center 
-y : ycenter    -- Y axis center
-c : xhw        -- Halfwidth of x axis
-d : yhw       	-- Halfwidth of	y axis
-h : Help       -- Display this help

"""

import sys,os,getopt
try:
    import astropy.io.fits as fits
except:
    import fits
from astropy.wcs import WCS
import numpy as np
from spectral_cube import SpectralCube
import pyspeckit
import pdb

def main():
    output_file = "default.fits"
    xcrop = False
    ycrop = False
    wc = False
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:o:x:y:c:d:w")
    except getopt.GetoptError,err:
        print(str(err))
        print(__doc__)
        sys.exit(2)
    for o,a in opts:
        if o == "-i":
            input_file = a
        elif o == "-o":
            output_file = a
        elif o == "-x":
            xc = float(a)
            xcrop = True
        elif o == "-y":
            yc = float(a)
            ycrop = True
        elif o == "-c":
            xhw = float(a)
        elif o == "-d":
            yhw = float(a)
        elif o == "-w":
            wc = True
        else:
            assert False, "unhandled option"
            print(__doc__)
            sys.exit(2)

    """
    Read the data, remove the extra axis if necessary.
    """
    d,h = fits.getdata(input_file,header=True,memmap=False)
    if len(d.shape) > 3:
        d = np.squeeze(d)
        h = strip_header(h,4)
        temp_file = input_file[:-5]+"_temp.fits"
        fits.writeto(temp_file,d,h,overwrite=True)
        temp = True
    else:
        temp_file = input_file
        temp = False

    try:
        """
        If the cube is a multi-component parameter map, the CUNIT3
        key will not be recognized by SpectralCube. Change it to
        km/s temporarily so SpectralCube does not throw an error.
        """
        if h['CUNIT3'] == 'Components':
            htemp = h[:]
            htemp['CUNIT3'] = 'km/s'
            #pdb.set_trace()
            temp_file = input_file[:-5]+"_temp.fits"
            fits.writeto(temp_file,d,htemp,overwrite=True)
            temp = True
    
        else:
            temp_file = input_file
            temp = False
    except:
        temp_file = input_file
        temp = False
    
    """        
    Get the crop center, width, and height in pixels if these 
    are given as world coordinates.
    """
    if wc:
        h = fits.getheader(temp_file)
        if xcrop and not ycrop:
            ymax = h['NAXIS2']-1
            yc = int(round(ymax/2.))
            xt,yc_wc = px2coord(0,yc,h)
            xt,ymax_wc = px2coord(0,ymax,h)
            yhw_wc = abs(ymax_wc-yc_wc)
            xc,yc = coord2px(xc,yc_wc,h)
            xhw = int(round(xhw/abs(h['CDELT1'])))
            yhw = int(round(yhw_wc/abs(h['CDELT2'])))
        elif ycrop and not xcrop:
            xmax = h['NAXIS1']-1
            xc = int(round(xmax/2.))
            xc_wc,yt = px2coord(xc,0,h)
            xmax_wc,yt = px2coord(xmax,0,h)
            xhw_wc = abs(xc_wc-xmax_wc)
            xc,yc = coord2px(xc_wc,yc,h)
            xhw = int(round(xhw_wc/abs(h['CDELT1'])))
            yhw = int(round(yhw/abs(h['CDELT2'])))
        else:
            xc,yc = coord2px(xc,yc,h)
            xhw = int(round(xhw/abs(h['CDELT1'])))
            yhw = int(round(yhw/abs(h['CDELT2'])))
    else:
        if xcrop and not ycrop:
            ymax = h['NAXIS2']-1
            yc = int(ymax/2)
            yhw = yc
        elif ycrop and not xcrop:
            xmax = h['NAXIS1']-1
            xc = int(xmax/2)
            xhw = xc
        else:
            xc = int(xc)
            yc = int(yc)
            xhw = int(xhw)
            yhw = int(yhw)

    print(xc, yc, xhw, yhw)

    """
    Use Pyspeckit to crop the cube spatially and write 
    it to the output file.
    """
    cube = SpectralCube.read(temp_file)
    cropped_cube = pyspeckit.cubes.subcube(cube,xc,xhw,yc,yhw)
    cropped_cube.write(output_file,format='fits',overwrite=True)

    """
    Fix the header info that Pyspeckit changed and overwrite 
    the output file. 
    """
    dd,hh = fits.getdata(output_file,header=True,memmap=False)
    hh['CTYPE3'] = h['CTYPE3']
    hh['CUNIT3'] = h['CUNIT3']
    fits.writeto(output_file,dd,hh,overwrite=True)

    #Delete the temporary file if it exists
    if temp:
        os.system('rm ' + temp_file)

def strip_header(h,n):
    """
    Remove the nth dimension from a FITS header
    """
    try:
        h['NAXIS'] = n-1
        h['WCSAXES'] = n-1
    except:
        h['NAXIS'] = n-1
    keys = ['NAXIS','CTYPE','CRVAL','CDELT','CRPIX','CUNIT','CROTA']
    for k in keys:
        try:
            del h[k+str(n)]
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

def get_coord_grids(h):
    XX,YY = np.meshgrid(np.arange(h['NAXIS1']),np.arange(h['NAXIS2']))
    w = WCS(h)
    LL,BB = w.all_pix2world(XX,YY,0)
    return LL,BB

def px2coord(x,y,h):
    if h['NAXIS']>2 or h['WCSAXES']>2:
        htemp = strip_header(h.copy(),3)
    else:
        htemp = h.copy()
    w = WCS(htemp)
    l,b = w.all_pix2world(x,y,0)
    return(l,b)

def coord2px(l,b,h):
    if h['NAXIS']>2 or h['WCSAXES']>2:
        htemp = strip_header(h.copy(),3)
    else:
        htemp = h.copy()
    w = WCS(htemp)
    x,y = w.all_world2pix(l,b,0)
    try:
        return(np.around(x).astype(int),np.around(y).astype(int))
    except:
        return(int(round(x)),int(round(y)))


if __name__ == '__main__':
    main()

