"""
prefit_raw_data_baselines.py

Remove zeroth order baseline from raw data. This avoids 
poorly combined spectra in overlap region of combined 
data cubes if the baselines are unstable in the 
individual observations. Can optionally run this 
process in parallel.

Example:
python prefit_raw_data_baselines.py 
       -i L30_Tile01-04_23694_MHz_line.fits 
       -o L30_Tile01-04_23694_MHz_line_prefit.fits -n 16 

-i : Input      -- Input file (from the Green Bank pipeline)
-o : Output     -- Output file 
-n : Numcores   -- Number of cores available for parallized computing
-h : Help       -- Display this help

"""




import sys,os,getopt
try:
    import astropy.io.fits as pyfits
except:
    import pyfits
import scipy.ndimage as im
import numpy as np
import numpy.ma as ma
import scipy.signal as si
import matplotlib.pyplot as plt
import multiprocessing as mp
import my_pad
import pdb

def main():

    #Defaults
    numcores = 1
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:o:n:h")
    except getopt.GetoptError:
        print("Invalid arguments")
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
        
    #Read in data into array, remove single-dimensional entries
    d,h = pyfits.getdata(input_file,header=True)
    d = np.squeeze(d)

    """
    Check that numcores does not exceed the number 
    of cores available
    """
    avail_cores = mp.cpu_count()
    if numcores > avail_cores:
        print("numcores variable exceeds the available number of cores.")
        print("Setting numcores equal to "+str(avail_cores))
        numcores = avail_cores
    if numcores > 1:
        #Split the data
        s = np.array_split(d, numcores, 2)
        procs = []
        #Fit baselines and write to temporary files
        for num in range(len(s)):
            procs.append(mp.Process(target=do_chunk,
                                              args=(num,s[num])))
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
            
    else:
        do_chunk(0,d)
    
    #Recombine baselined temporary files 
    dout = recombine(numcores)
    hout = strip_header(h[:],4)
    pyfits.writeto(output_file,dout,hout,overwrite=True)

    #Remove temporary files
    for n in np.arange(numcores):
        os.system("rm prefit_temp"+str(n)+".fits")


def do_chunk(num,data):
    """
    Use apply_along_axis to apply 
    baseline fitting to each spectrum in 
    this chunk of the cube.
    """
    print("Fitting chunk"+str(num)+"...")
    ya =  np.apply_along_axis(baseline_and_deglitch,0,data)
    pyfits.writeto("prefit_temp"+str(num)+".fits",ya,overwrite=True)
    
        
def recombine(numparts,output_file="test_final.fits"):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    for n in range(numparts):
        d = pyfits.getdata("prefit_temp"+str(n)+".fits")
        indata.append(d)
    final = np.dstack(indata)
    return(final)
        
def rolling_window(a,window):
    """
    Magic code to quickly create a second dimension
    with the elements in a rolling window. This
    allows us to apply numpy operations over this
    extra dimension MUCH faster than using the naive approach.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)    
    strides = a.strides+(a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def fit_baseline(masked,xx,ndeg=0):
    """
    Fit the masked array with a polynomial of 
    the given order.
    """
    ya = ma.polyfit(xx,masked,ndeg)
    basepoly = np.poly1d(ya)
    return(basepoly)

def baseline_and_deglitch(orig_spec,window_halfwidth=20):
    """
    Function that fits a zeroth order polynomial baseline function 
    to a spectrum and subtracts it.
    
    Parameters
    ----------
    orig_spec : ndarray
        The original spectrum with full resolution.
    window_halfwidth : int
        Half of the window width used to mask the spectrum for fitting.

    Returns
    -------
    sub : ndarray
        Baseline-subtracted spectrum.
    """

    #Mask and smooth spectrum
    masked= mask_for_baseline(orig_spec,window_halfwidth=window_halfwidth)
    #Get best-fit zeroth order polynomial baseline
    xx = np.arange(masked.size)
    poly = fit_baseline(masked,xx)
    baseline = poly(xx)
    #Subtract baseline and smooth spectrum
    sub = orig_spec - baseline

    return(sub)


def mask_for_baseline(spec,sigma_cut=1.5,window_halfwidth=15):
    """
    Mask the spectral channels that contain signal. Search for 
    signal by comparing the local standard deviation within a 
    window width of 2*window_halfwidth to the median local standard deviation 
    in all windows throughout the spectrum.
    """
    ya = rolling_window(spec,window_halfwidth*2)
    #Calculate local standard dev for each channel and pad the output
    stds = my_pad.pad(np.nanstd(ya,-1),(window_halfwidth-1,window_halfwidth),
                      mode='edge')

    #Figure out which bits of the spectrum have signal/glitches
    med_std = np.nanmedian(stds)
    std_std = np.nanstd(stds)
    sigma_x_bar = med_std/np.sqrt(window_halfwidth)
    sigma_s = (1./np.sqrt(2.))*sigma_x_bar

    #Mask out signal for baseline
    masked = ma.masked_where(np.logical_or(stds > med_std+sigma_cut*sigma_s,
                                           np.isnan(spec)),spec)
    #Mask 10 channels around each masked channel to capture faint line wings
    if ma.is_masked(masked):
        where_masked = np.where(masked.mask)[0]
        mask_extension = 10
        for channel in where_masked:
            masked[slice(channel-10,channel+10,1)] = ma.masked
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

