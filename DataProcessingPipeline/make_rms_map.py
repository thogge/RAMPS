"""
make_rms_map.py

Calculates the rms noise for each spectrum in a data cube and 
constructs an rms noise map.

Example:
python make_rms_map.py 
       -i L30_Tile01_NH3_1-1_fixed.fits 
       -o L30_Tile01_NH3_1-1_noise.fits
       -n 16

-i : Input      -- Input file (reduced by pipeline)
-o : Output     -- Output file 
-n : Numcores   -- Number of cores for parallel processing
-h : Help       -- Display this help

"""

import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb
import numpy.ma as ma
import sys,os,getopt
import multiprocessing as mp

def main():
    #Defaults
    numcores = 1                
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:o:n:h")
    except getopt.GetoptError as err:
        print(err.msg)
        print(__doc__)
        sys.exit(2)
    for o,a in opts:
        if o == "-i":
            input_file = a
        elif o == "-o":
            output_file = a
        elif o == "-n":
            numcores = int(a)
        elif o == "-h":
            print(__doc__)
            sys.exit(1)
        else:
            assert False, "unhandled option"
            print(__doc__)
            sys.exit(2)
    
    #Read and data and remove extra dimension if it exists
    d,h = fits.getdata(input_file,header=True)
    if len(d.shape) > 3:
        d = np.squeeze(d)
        h = strip_header(h,4)
        
    #Check that numcores does not exceed the number of cores available
    avail_cores = mp.cpu_count()
    if numcores > avail_cores:
        print("numcores variable exceeds the available number of cores.")
        print("Setting numcores equal to "+str(avail_cores))
        numcores = avail_cores

    if numcores > 1:
        #Split the data and run the rms function in parallel 
        s = np.array_split(d, numcores, 2)
        procs = []
        for num in range(len(s)):
            procs.append(mp.Process(target=do_chunk_rms,args=(num,s[num])))
        for proc in procs:
            proc.start()
        
        for proc in procs:
            proc.join()
    else:
        #If using only one core, run the rms function on the full cube
        do_chunk_rms(0,d)
    
    #Recombine the temporary rms map chunks to construct full rms map
    rms_map = recombine_rms(numcores)

    #Modify header and write to FITS file
    hout = strip_header(h,3)
    hout['DATAMAX'] = np.nanmax(rms_map)
    hout['DATAMIN'] = 0.
    fits.writeto(output_file,rms_map,hout,clobber=True) 
    
    #Delete temporary rms map files
    for n in np.arange(numcores).astype(str):
        os.system("rm rms_temp"+n+".fits")


def do_chunk_rms(num,data):
    """
    Calculate the rms for each spectrum in this chunk of the data cube
    """
    ya =  np.apply_along_axis(get_rms,0,data)
    fits.writeto("rms_temp"+str(num)+".fits",ya,clobber=True)

def recombine_rms(numparts):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    for n in range(numparts):
        d = fits.getdata("rms_temp"+str(n)+".fits")
        indata.append(d)
    final = np.column_stack(indata)
    return(final)

def noise_est(spec):
    """
    Estimates the noise in a spectrum by the 
    channel-to-channel variations. 
    """
    diff = np.diff(spec)
    noise_est = np.sqrt(np.mean(diff*diff)/2.)
    return noise_est

def get_rms(spec):
    """
    Calculate the rms noise of a spectrum with masked signal.
    """
    if np.isnan(spec).all():
        rms = np.nan
    else:
        m = mask_for_rms(spec)
        rms = ((m**2).mean())**(0.5)
    return rms

def mask_for_rms(spec,nsig=3.):
    """
    Mask the signal in a spectrum for the purpose of calculating the rms
    noise. Use the noise_est to estimate the noise for masking. This method
    is susceptible to error in the presence of bright, narrow line emission
    (such as maser emission), so estimating the noise a second time from the 
    masked data is more accurate.
    """
    #First calculate the noise estimate from the unmasked array 
    noise = noise_est(spec)
    masked = ma.masked_where(np.logical_or(spec > nsig*noise,
                                           np.isnan(spec)),spec)
    #Mask 10 channels around each masked channel to capture faint line wings
    mask_ext = 10
    if ma.is_masked(masked):
        where_masked = np.where(masked.mask)[0]
        for channel in where_masked:
            masked[slice(channel-mask_ext,channel+mask_ext,1)] = ma.masked

    #Calculate the noise estimate from the masked array 
    noise = noise_est(masked)
    masked = ma.masked_where(np.logical_or(spec > nsig*noise,
                                           np.isnan(spec)),spec)
    #Mask 10 channels around each masked channel to capture faint line wings
    if ma.is_masked(masked):
        where_masked = np.where(masked.mask)[0]
        for channel in where_masked:
            masked[slice(channel-mask_ext,channel+mask_ext,1)] = ma.masked
    return(masked)

def strip_header(h,n):
    """
    Remove the nth dimension from a FITS header
    """
    try:
        h['NAXIS'] = n-1
        h['WCSAXES'] = n-1
    except:
        h['NAXIS'] = n-1

    keys = ['NAXIS','CTYPE','CRVAL','CDELT','CRPIX','CUNIT']
    for k in keys:
        try:
            del h[k+str(n)]
        except:
            pass
    return(h)
    
if __name__ == '__main__':
    main()
