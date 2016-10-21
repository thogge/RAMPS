"""
make_noise_map.py

Calculates the noise for each pixel of a datacube and outputs a map of the noise.

Example:
python make_noise_map.py 
       -i L30_Tile01_NH3_1-1_fixed.fits 
       -o L30_Tile01_NH3_1-1_noise.fits

-i : Input      -- Input file (reduced by pipeline)
-o : Output     -- Output file 
-h : Help       -- Display this help

"""

import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb
import numpy.ma as ma
import sys,os,getopt
import multiprocessing, logging
import datetime

def main():
    
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
    numcores = 16                
    s = np.array_split(d, numcores, 2)
    ps = []
    for num in range(len(s)):
        ps.append(multiprocessing.Process(target=do_chunk_noise,args=(num,s[num])))
    for p in ps:
        p.start()
        
    for p in ps:
        p.join()
    noise = recombine_noise(numcores)

    if h['NAXIS'] > 3:
        hout = strip_header(h,4)
        hout = strip_header(hout,3)
    else:
        hout = strip_header(h,3)
    pyfits.writeto(output_file,noise,hout,clobber=True)  
    os.system("rm temp*")


def noise_est(spec):
    """
    Estimate the noise in a spectrum from the channel-to-channel
    variation. This method is relatively unaffected by signal
    and baseline effects.
    """
    if len(spec) <= 1:
	noise_est = np.nan
    else:
        cropped_spec = spec[int(0.2*len(spec)):int(0.8*len(spec))]
        noise_sq = np.zeros(len(cropped_spec))
        for i in range(len(cropped_spec)-1):
            noise_sq[i] = (cropped_spec[i] - cropped_spec[i+1])**2
        noise_est = np.sqrt(np.mean(noise_sq)/2.)
    return noise_est

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
        del h['CROTA'+str(n)]
    except:
        pass
    return(h)

def do_chunk_noise(num,data):
    """
    Basic command to process data
    apply_along_axis is the fastest way
    I have found to do this.
    """
    ya =  np.apply_along_axis(noise_est,0,data)
    pyfits.writeto("temp"+str(num)+".fits",ya,clobber=True)

def recombine_noise(numparts,output_file="test_final.fits"):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    for n in range(numparts):
        d = pyfits.getdata("temp"+str(n)+".fits")
        indata.append(d)
    final = np.column_stack(indata)
    return(final)

if __name__ == '__main__':
    main()
