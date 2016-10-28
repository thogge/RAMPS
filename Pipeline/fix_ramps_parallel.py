#!/usr/bin/env python
# encoding: utf-8
"""
fix_ramps_parallel_NH3.py

Fit baselines and remove glitches/spikes from RAMPS data.
Optionally transforms to velocity and outputs a (masked)
moment zero (integrated intensity map) and first moment
(velocity field) 

This version runs in parallel, which is useful because
the process is fairly slow.

Example:
python fix_ramps_parallel_NH3.py 
       -i L30_Tile01-04_23694_MHz_line.fits 
       -o L30_Tile01-04_fixed.fits -s 11 -n 16 -fv01

-i : Input      -- Input file (reduced by pipeline)
-o : Output     -- Output file 
-s : Smooth     -- Size of kernel for median smooth
-n : Numcores   -- Number of cores available for parallized computing
-f : Fit        -- Flag to fit and remove baseline
-v : Velocity   -- Flag to convert to velocity 
-0 : Zeroth Moment Map -- Flag to produce a moment zero
                   map (called Ouput+_mom0.fits)
-1 : First Moment Map -- Flag to produce a first moment
                   map (called Ouput+_mom1.fits)
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
import multiprocessing, logging
import my_pad

def main():

    #Set 'numcores' to the default number of 
    #processers available for parallelized 
    #baselining and moment making.
    numcores = 16
    output_file = "default.fits"
    do_fit = False
    do_vel = False
    do_mom0 = False
    do_mom1 = False
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:o:s:n:fv01h")
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
            filter_width = a
        elif o == "-n":
            numcores = a
        elif o == "-f":
            do_fit = True
        elif o == "-v":
            do_vel = True
        elif o == "-0":
            do_mom0 = True
        elif o == "-1":
            do_mom1 = True
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

    if do_fit:
        #Split the data
        s = np.array_split(d, numcores, 2)
        ps = []
        #Fit baselines and write to temporary files
        for num in range(len(s)):
            ps.append(multiprocessing.Process(target=do_chunk_fit,
                                              args=(num,s[num],
                                                    filter_width=filter_width,
                                                    do_vel)))
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        #Recombine baselined temporary files 
        dout = recombine(numcores)

        if do_vel:
            #Edit header to put spectral axis in velocity space
            hout = downsample_header(change_to_velocity(strip_header(h[:],4)))
            old_ref_channel =  float(hout['CRPIX3'])
            hout['CRPIX3'] = float(len(dout[:,1,1]))-old_ref_channel
        else:
            hout = downsample_header(strip_header(h[:],4))
	hout['DATAMIN'] = -3.
        hout['DATAMAX'] = 3.
        pyfits.writeto(output_file,dout,hout,clobber=True)
        #Remove temporary files
        os.system("rm temp*")
    else:
        dout = d[:]
        hout = h[:]
    
    if do_mom0:
        # Check that the header has spectral information
        if (hout['CTYPE3'] == 'VELO-LSR'):
            if (hout['NAXIS'] == 4):
                hout = strip_header(hout[:],4)
        elif (hout['CTYPE3'] == 'FREQ'):
            if (hout['NAXIS'] == 4):
                hout = change_to_velocity(strip_header(hout[:],4))
            elif (hout['NAXIS'] == 3):
                hout = change_to_velocity(hout[:])
        else:
            raise ValueError('Header should hold spectral information') 

        #Split the data
        s = np.array_split(dout, numcores, 2)
        ps = []
        #Create integrated intensity maps and write to temporary files
        for num in range(len(s)):
            ps.append(multiprocessing.Process(target=do_chunk_mom0,args=(num,s[num],hout)))
        for p in ps:
            p.start()
        
        for p in ps:
            p.join()
        #Recombine temporary files
        mom0 = recombine_moments(numcores)
        mom0[np.where(mom0 == 0)] = np.nan
        mom0_file = output_file[0:-5]+"_mom0.fits"  
        #Update header and write the data
        hmom = strip_header(hout[:],3)
        hmom['BUNIT'] = 'K*km/s'
        hmom['DATAMIN'] = 0.
       	hmom['DATAMAX'] = 20.
        pyfits.writeto(mom0_file,mom0,hmom,clobber=True)  
        #Remove temporary files
        os.system("rm temp*")

        
    if do_mom1:
        # Check that the header has spectral information
        if (hout['CTYPE3'] == 'VELO-LSR'):
            if (hout['NAXIS'] == 4):
                hout = strip_header(hout[:],4)
        elif (hout['CTYPE3'] == 'FREQ'):
            if (hout['NAXIS'] == 4):
                hout = change_to_velocity(strip_header(hout[:],4))
            elif (hout['NAXIS'] == 3):
                hout = change_to_velocity(hout[:])
        else:
            raise ValueError('Header should hold spectral information') 

        #Split the data
        s = np.array_split(dout, numcores, 2)
        ps = []
        #Create velocity maps and write to temporary files
        for num in range(len(s)):
            ps.append(multiprocessing.Process(target=do_chunk_mom1,args=(num,s[num],hout)))
        for p in ps:
            p.start()
        
        for p in ps:
            p.join()
        #Recombine temporary files
        mom1 = recombine_moments(numcores)
        mom1[np.where(mom1 == 0)] = np.nan
        mom1_file = output_file[0:-5]+"_mom1.fits"
        #Update header and write the data
        hmom = strip_header(hout[:],3)
        hmom['BUNIT'] = 'km/s'
        good_data_mom1 = mom1[np.where(np.isfinite(mom1))]
        mom1_min = np.min(good_data_mom1)
        mom1_max = np.max(good_data_mom1)
        hmom['DATAMIN'] = mom1_min
        hmom['DATAMAX'] = mom1_max
        pyfits.writeto(mom1_file,mom1,hmom,clobber=True)
        #Remove temporary files
        os.system("rm temp*")


def do_chunk_fit(num,data,filter_width,do_vel):
    """
    Use apply_along_axis to apply 
    baseline fitting to each spectrum in 
    this chunk of the cube.
    """
    ya =  np.apply_along_axis(baseline_and_deglitch,0,data,
                              filter_width=filter_width,do_vel)
    pyfits.writeto("temp"+str(num)+".fits",ya,clobber=True)
    
    
def do_chunk_mom0(num,data,header):
    """
    Use apply_along_axis to calculate 
    the integrated intensity of each 
    spectrum in this chunk of the cube.
    """
    ya =  np.ma.apply_along_axis(sum_over_signal,0,data)
    ya = ya.data
    #Convert to K km/s. Assumes CTYPE3 is LSR Velocity
    good_data_locations = np.where(ya > 0.)
    ya[good_data_locations] *= 0.001*abs(header['CDELT3'])
    pyfits.writeto("temp"+str(num)+".fits",ya,clobber=True)
    
    
def do_chunk_mom1(num,data,header):
    """
    Use apply_along_axis to calculate 
    the integrated intensity of each 
    spectrum in this chunk of the cube.
    """
    ya =  np.ma.apply_along_axis(first_moment,0,data,header)
    ya = ya.data
    pyfits.writeto("temp"+str(num)+".fits",ya,clobber=True)
    
    
def recombine(numparts,output_file="test_final.fits"):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    for n in range(numparts):
        d = pyfits.getdata("temp"+str(n)+".fits")
        indata.append(d)
    final = np.dstack(indata)
    return(final)
    
def recombine_moments(numparts,output_file="test_final.fits"):
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

def noise_est(spec):
    """
    Estimates the noise in a spectrum by the 
    channel-to-channel variations. Perform this
    function on only the inner portion of the 
    spectrum to avoid any residual single channel
    birdies in the data.
    """
    spec = spec[int(0.2*len(spec)):int(0.8*len(spec))]
    if np.isnan(np.sum(spec)):
        noise_est = np.nan
    else:
        noisesq = np.zeros(len(spec))
        for i in range(len(spec)-1):
            noisesq[i] = (spec[i] - spec[i+1])**2
        noise_est = np.sqrt(np.mean(noisesq)/2.)
    return noise_est

def fit_baseline(masked,xx,ndeg=2):
    """
    Fit the masked array with a polynomial of 
    the given order.
    """
    ya = ma.polyfit(xx,masked,ndeg)
    basepoly = np.poly1d(ya)
    return(basepoly)

def find_best_baseline(masked,xx,max_order=7,prior_penalty=1):
    """
    Consider polynomial baselines up to order max_order.
    Select the baseline with the lowest reduced chi-squared, 
    where prior_penalty can be used to increase the penalty
    for fitting with a higher order baseline. 
    """
    chisqs = np.zeros(max_order)
    ndegs = np.arange(max_order)
    for i,ndeg in enumerate(ndegs):
        basepoly = fit_baseline(masked,xx,ndeg=ndeg)
        base = basepoly(xx)
        chisqs[i] = np.sum((masked-base)**2)/(ma.count(masked)
                                              -1-prior_penalty*ndeg)
    
    return(np.argmin(chisqs))
    
def baseline_and_deglitch(orig_spec,
                          do_vel,
                          ww=25,
                          sigma_cut=1.5,
                          poly_n=2.,
                          filter_width=1,
                          max_order=7,
                          prior_penalty=1.):
    """
    (1) Calculate a rolling standard deviation (s) in a window
        of 2*ww pixels
    (2) Mask out portions of the spectrum where s is more than
        sigma_cut times the median value for s. This seems to 
        be mis-calibrated (perhaps not independent?). A value 
        of 1.5 works well to remove all signal.
    (3) Downsample the masked spectrum (to avoid memory bug)
        and find the minimum order polynomial baseline that 
        does a good job of fitting the data.
    (3) Median filter (with a filter width of filter_width)
        to remove the single-channel spikes seen.
    """
    smoothed = im.median_filter(orig_spec,filter_width)[::filter_width]
    ya = rolling_window(smoothed,ww*2)
    #Calculate standard dev and pad the output
    stds = my_pad.pad(np.std(ya,-1),(ww-1,ww),mode='edge')

    #Figure out which bits of the spectrum have signal/glitches
    med_std = np.median(stds)
    std_std = np.std(stds)
    sigma_x_bar = med_std/np.sqrt(ww)
    sigma_s = (1./np.sqrt(2.))*sigma_x_bar

    #Mask out signal for baseline
    masked = ma.masked_where(stds > med_std+sigma_cut*sigma_s,smoothed)
    xx = np.arange(masked.size)
    xx = xx.astype(np.float32) #To avoid bug with ma.polyfit in np1.6
    npoly = find_best_baseline(masked,xx)
    basepoly = fit_baseline(masked,xx,ndeg=npoly)
    #Some kludgy code to refactor the baseline polynomial to
    #the full size spectra
    xxx = np.arange(orig_spec.size)
    params = np.asarray(basepoly)
    rr = filter_width
    newparams = []
    for i,p in enumerate(params[::-1]):
        newparams.append(p/rr**i)
    newparams = newparams[::-1]
    newpoly = np.poly1d(newparams)
    newbaseline = newpoly(xxx) 
    #Subtract off baseline
    sub = orig_spec-newbaseline
    filtered_spectrum = im.median_filter(sub,filter_width)[::filter_width]

    if do_vel:
        #Reverse spectral array so positive velocity is on the right
        final = filtered_spectrum[::-1]
    else:
	final = filtered_spectrum

    return(final)


def downsample_header(h,filter_width=1):
    """
    Since we downsample our spectra by a factor 
    of filter_width we have to change the header as well.
    Currently this is not well-integrated. filter_width 
    here _needs_ to be the same as filter_width in
    baseline_and_deglitch
    """
    h['CDELT3'] = h['CDELT3']*float(filter_width)
    h['CRPIX3'] = h['CRPIX3']/float(filter_width)
    return(h)

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
    
    
def change_to_velocity(h):
    if ('RESTFRQ' in h):
        h.rename_keyword('RESTFRQ','RESTFREQ')
    else:
        pass
    n = (h['RESTFREQ']-h['CRVAL3'])/h['CDELT3']+1
    h['CTYPE3'] = 'VELO-LSR'
    delta_freq = h['CDELT3']
    h['CDELT3'] = 299792.458/float(h['CRVAL3'])*1000*delta_freq
    h['CRVAL3'] = 0.
    h['CRPIX3'] = int(n)
    del h['CUNIT3']
    del h['SPECSYS']
    return(h)
    

def sum_over_signal(spec,nsig=3.):
    """
    Sum over regions of significant signal.
    Could be modified to output a mask in order
    to enable quick calculation of multiple moments.
    Run this on spectra that have already been
    baseline subtracted.
    """
    if len(np.where(np.isnan(spec))[0][:]) == len(spec):
        """
        If the spectrum is all nans, then it means we haven't
        mapped this location. Fill it with a large negative
        number to differentiate it from spectra that don't
        have significant signal.
        """
        mom0=-999
    else:
        sigma = noise_est(spec)
        #Mask out noise for integrated intensity
        masked = ma.masked_where(spec < nsig*sigma,spec)

        """
        Require that each unmasked channel be contiguous with two
        other unmasked channels. This prevents the inclusion of three 
        sigma channels that are purely noise.
        """
        temp_masked= ma.zeros(ma.shape(masked))
        w = 1
	masked[0:w+1] = ma.masked
        masked[len(masked)-w:len(masked)] = ma.masked
        for i in range(w,len(masked)-w-1):
            if (ma.getmask(masked)[i] == False):
                if (not ma.getmask(masked)[i-w-1:i].any()):
                    temp_masked[i] = ma.getmask(masked)[i]
                elif ((not ma.getmask(masked)[i-w:i].any()) and 
                      (not ma.getmask(masked)[i+1:i+w+1].any())):
                    temp_masked[i] = ma.getmask(masked)[i]
                elif (not ma.getmask(masked)[i+1:i+w+2].any()):
                    temp_masked[i] = ma.getmask(masked)[i]
                else:
                    temp_masked[i] = ma.masked
            else:
                temp_masked[i] = ma.masked
        masked = ma.masked_where(np.ma.getmask(temp_masked), masked) 
        spec_sum = ma.sum(masked)
    return(spec_sum)

def first_moment(spec,h,nsig=3.):
    """
    Take the first moment over regions of significant signal.
    Run	this on	spectra	that have already been 
    baseline subtracted.
    """
    if len(np.where(np.isnan(spec))[0][:]) == len(spec):
        """
        If the spectrum is all nans, then it means we haven't
        mapped this location. Fill it with a large negative
        number to differentiate it from spectra that don't
        have significant signal.
        """
        mom1 = -999
    else:
        sigma = noise_est(spec)
        #Mask out noise for first moment
        masked = ma.masked_where(spec < nsig*sigma,spec)

        """
	Require that each unmasked channel be contiguous with two
        other unmasked channels. This prevents the inclusion of three
        sigma channels that are purely noise.
        """
	temp_masked= ma.zeros(ma.shape(masked))
        w = 1
        masked[0:w+1] = ma.masked
        masked[len(masked)-w:len(masked)] = ma.masked
        for i in range(w,len(masked)-w-1):
            if (ma.getmask(masked)[i] == False):
                if (not ma.getmask(masked)[i-w-1:i].any()):
                    temp_masked[i] = ma.getmask(masked)[i]
                elif ((not ma.getmask(masked)[i-w:i].any()) and 
                      (not ma.getmask(masked)[i+1:i+w+1].any())):
                    temp_masked[i] = ma.getmask(masked)[i]
                elif (not ma.getmask(masked)[i+1:i+w+2].any()):
                    temp_masked[i] = ma.getmask(masked)[i]
                else:
                    temp_masked[i] = ma.masked
            else:
                temp_masked[i] = ma.masked
        masked = ma.masked_where(np.ma.getmask(temp_masked), masked)

        """
        Create a masked spectrum of data*velocity to calculate 
        the first moment.
        """
        temp_masked = ma.zeros(ma.shape(masked))
        temp_masked.mask = ma.getmask(masked)
        vel = np.arange(len(masked))
        for i in range(len(masked)):
            vel[i] = 0.001*(vel[i] - h['CRPIX3'])*h['CDELT3'] + h['CRVAL3']
            temp_masked.data[i] = masked.data[i]*vel[i]
        mom1 = ma.sum(temp_masked)/ma.sum(masked)
    return(mom1)


if __name__ == '__main__':
    main()

