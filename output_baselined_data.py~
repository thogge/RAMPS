#!/usr/bin/env python
# encoding: utf-8
"""
OutputBaselinedData.py

Fit baselines and remove glitches/spikes from RAMPS data.
Transforms to velocity axis. Outputs baseline-subtracted data cube. 
Optionally outputs a (masked) moment zero map (integrated intensity) 
and first moment map (velocity field).

This version runs in parallel, which is useful because
the process is fairly slow.

Example:
python OutputBaselinedData.py 
       -i L30_Tile01_23694_MHz_line.fits 
       -o L30_Tile01_NH3_1-1 -s 11 -w 6 -n 16 -p 2 -f01

-i : Input              -- Input file (reduced by pipeline)
-o : Output             -- Output file base name
-s : Smooth             -- Size of kernel for median smooth
-n : Numcores           -- Number of cores available for parallized computing
-w : Window HalfWidth   -- 1/2 width of rolling window used for baseline 
                           fit mask
-p : Max order          -- Highest order polynomial with which to fit
-f : Perform fit        -- Flag to produce a data cube with a baseline
                           removed (called Ouput+_cube.fits)
-0 : Zeroth Moment Map  -- Flag to produce a moment zero
                           map (called Ouput+_mom0.fits)
-1 : First Moment Map   -- Flag to produce a first moment
                           map (called Ouput+_mom1.fits)
-h : Help               -- Display this help

"""




import sys,os,getopt
try:
    import astropy.io.fits as fits
except:
    import pyfits as fits
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
    output_base = "default.fits"
    numcores = 1
    max_order = 2
    filter_width = 1
    window_halfwidth = 15
    do_fit = False
    do_mom0 = False
    do_mom1 = False
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:o:s:n:w:p:f01h")
    except getopt.GetoptError:
        print("Invalid key")
        print(__doc__)
        sys.exit(2)
    for o,a in opts:
        if o == "-i":
            input_file = a
        elif o == "-o":
            output_base = a
        elif o == "-n":
            numcores = int(a)
        elif o == "-s":
            filter_width = int(a)
        elif o == "-w":
            window_halfwidth = int(a)
        elif o == "-p":
            max_order = int(a)
        elif o == "-f":
            do_fit = True
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
    d,h = fits.getdata(input_file,header=True,memmap=False)
    d = np.squeeze(d)
    if do_fit:
        """
        Check that numcores does not exceed the number 
        of cores available
        """
        avail_cores = mp.cpu_count()
        if numcores > avail_cores:
            print("numcores variable exceeds the available number of cores.")
            print("Setting numcores equal to "+str(avail_cores))
            numcores = avail_cores
        #Fit baselines and write to temporary files
        if numcores > 1:
            s = np.array_split(d, numcores, 2)
            procs = []
            for num in range(len(s)):
                procs.append(mp.Process(target=do_chunk,
                                         args=(num,s[num],
                                         window_halfwidth,
                                         max_order,
                                         filter_width)))
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join()
        else:
            do_chunk(0,d,window_halfwidth,max_order,filter_width)
        #Recombine baselined temporary files 
        baselined_cube = recombine(numcores,"cube")
    
        #Edit headers, write data
        hout = downsample_header(change_to_velocity(strip_header(h[:],4)),
                                 filter_width)
        if '.fits' in output_base:
            fits.writeto(output_base,baselined_cube,hout,overwrite=True)
        else:
            fits.writeto(output_base+'_cube.fits',
                         baselined_cube,hout,overwrite=True)
        #Remove temporary files
        delete_temporary_files("cube",numcores)

    else:
        baselined_cube = d
        hout = h[:]
    
    if do_mom0:
        # Check that the header has spectral information
        if 'V' in hout['CTYPE3']:
            if (hout['NAXIS'] == 4):
                hout = strip_header(hout[:],4)
        elif (hout['CTYPE3'] == 'FREQ'):
            if (hout['NAXIS'] == 4):
                hout = change_to_velocity(strip_header(hout[:],4))
            elif (hout['NAXIS'] == 3):
                hout = change_to_velocity(hout[:])
        else:
            raise ValueError('Header should hold spectral information') 

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
            s = np.array_split(baselined_cube, numcores, 2)
            procs = []
            #Create integrated intensity maps and write to temporary files
            for num in range(len(s)):
                procs.append(mp.Process(target=do_chunk_mom0,
                                        args=(num,s[num],hout)))
            for proc in procs:
                proc.start()
                
            for proc in procs:
                proc.join()
        else:          
            do_chunk_mom0(0,baselined_cube,hout)
        #Recombine temporary files
        mom0 = recombine(numcores,"mom0")
        mom0_file = output_base+"_mom0.fits"  
        #Update header and write the data
        hmom = strip_header(hout[:],3)
        hmom['BUNIT'] = 'K*km/s'
        hmom['DATAMIN'] = 0.
       	hmom['DATAMAX'] = 20.
        fits.writeto(mom0_file,mom0,hmom,overwrite=True)  
        #Remove temporary files
        delete_temporary_files("mom0",numcores)
        
    if do_mom1:
        # Check that the header has spectral information
        if 'V' in hout['CTYPE3']:
            if (hout['NAXIS'] == 4):
                hout = strip_header(hout[:],4)
        elif (hout['CTYPE3'] == 'FREQ'):
            if (hout['NAXIS'] == 4):
                hout = change_to_velocity(strip_header(hout[:],4))
            elif (hout['NAXIS'] == 3):
                hout = change_to_velocity(hout[:])
        else:
            raise ValueError('Header should hold spectral information') 

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
            s = np.array_split(baselined_cube, numcores, 2)
            procs = []
            #Create velocity maps and write to temporary files
            for num in range(len(s)):
                procs.append(mp.Process(target=do_chunk_mom1,
                                        args=(num,s[num],hout)))
            for proc in procs:
                proc.start()
        
            for proc in procs:
                proc.join()
        else:
            do_chunk_mom1(0,baselined_cube,hout)
        #Recombine temporary files
        mom1 = recombine(numcores,"mom1")
        mom1_file = output_base+"_mom1.fits"
        #Update header and write the data
        hmom = strip_header(hout[:],3)
        hmom['BUNIT'] = 'km/s'
        hmom['DATAMIN'] = np.nanmin(mom1)
        hmom['DATAMAX'] = np.nanmax(mom1)
        fits.writeto(mom1_file,mom1,hmom,overwrite=True)
        #Remove temporary files
        delete_temporary_files("mom1",numcores)


def do_chunk(num,data,window_halfwidth,max_order,filter_width):
    """
    Loop over the pixels in a data cube and apply the
    baseline fitting function to each spectrum in 
    this chunk of the cube.
    """
    print(num)
    #Reverse the spectral axis to arange spectra from low to high LSR velocity
    data = data[::-1,:,:]
    nax1 = data.shape[2]
    nax2 = data.shape[1]
    nax3 = len(data[:,0,0][::filter_width])
    outcube = np.full((nax3,nax2,nax1,),np.nan)
    """
    rel_diff_std is the standard deviation of rel_diff for a large 
    number of simulated spectra that have no residual baseline. If a 
    baseline subtracted spectrum has an rel_diff value significantly 
    above this value, then it likely contains a significant residual 
    baseline.
    """
    rel_diff_std = 0.00404*(filter_width)**(0.5)
    nsig = 5.
    for i in range(nax2):
        for j in range(nax1):
            if np.isfinite(data[int(data.shape[0]/2),i,j]):
                rel_diff,outcube[:,i,j] = fit_wrapper(data[:,i,j],
                                                      window_halfwidth,
                                                      max_order,
                                                      filter_width)
                if rel_diff > nsig*rel_diff_std:
                    rel_diff,outcube[:,i,j] = fit_wrapper(data[:,i,j],
                                                          3*window_halfwidth,
                                                          max_order,
                                                          filter_width)
                if rel_diff > nsig*rel_diff_std:
                    rel_diff,outcube[:,i,j] = fit_wrapper(data[:,i,j],
                                                          3*window_halfwidth,
                                                          0,filter_width)
    fits.writeto("cube_temp"+str(num)+".fits",outcube,overwrite=True)
    
def do_chunk_mom0(num,data,header):
    """
    Loop over pixels to calculate 
    the integrated intensity of each 
    spectrum in this chunk of the cube.
    """
    ya = np.full(data[0,:,:].shape,np.nan)
    hw = len(data[:,0,0])/2.
    for i in range(ya.shape[0]):
        for j in range(ya.shape[1]):
            if np.isfinite(data[int(hw),i,j]):
                ya[i,j] = sum_over_signal(data[:,i,j])
    #Convert to K km/s. Assumes CTYPE3 is LSR Velocity
    #loc = np.where(ya>0.)
    #print np.nansum(ya[loc])
    #good_data_locations = np.where(ya > 0.)
    #ya[good_data_locations] *= 0.001*abs(header['CDELT3'])
    ya = ya*0.001*abs(header['CDELT3'])
    fits.writeto("mom0_temp"+str(num)+".fits",ya,overwrite=True)
      
def do_chunk_mom1(num,data,header):
    """
    Loop over pixels to calculate 
    the integrated intensity of each 
    spectrum in this chunk of the cube.
    """
    ya = np.full(data[0,:,:].shape,np.nan)
    for i in range(ya.shape[0]):
        for j in range(ya.shape[1]):
            ya[i,j] = first_moment(data[:,i,j],header)
    fits.writeto("mom1_temp"+str(num)+".fits",ya,overwrite=True)
    
def recombine(numparts,output_base):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    data_naxes = []
    for n in range(numparts):
        d = fits.getdata(output_base+"_temp"+str(n)+".fits")
        indata.append(d)
        data_naxes.append(len(d.shape))
    data_naxes = np.asarray(data_naxes)
    if (data_naxes == 3).all():
        final = np.dstack(indata)
    elif (data_naxes == 2).all():
        final = np.column_stack(indata)
    else:
        print("FITS array chunks must all have 2 axes or all have 3 axes.")
    return(final)

def delete_temporary_files(output_base,numcores):
    for n in np.arange(numcores):
        print("rm "+output_base+"_temp"+str(n)+".fits")
        os.system("rm "+output_base+"_temp"+str(n)+".fits")


def fit_wrapper(orig_spec,window_halfwidth,max_order,filter_width):
    """
    Wrapper that fits a polynomial baseline function to a spectrum
    
    Parameters
    ----------
    orig_spec : numpy array of floats
        The original spectrum with full resolution.
    window_halfwidth : int
        Half of the window width used to mask the spectrum for fitting.
    max_order : int
        The maximum polynomial order allowed for baseline fitting.
    filter_width : int
        The smoothing factor. The resulting spectrum will have a length
        reduced by a factor of filter_width.

    Returns
    -------
    rel_diff : float
        rel_diff value of baseline-subtracted spectrum. 
    final : ndarray
        Baseline-subtracted spectrum.
    """
    #Mask and smooth spectrum
    m = ma.masked_invalid(orig_spec)
    sm = im.median_filter(m,filter_width)[::filter_width]
    masked= mask_for_baseline(sm,window_halfwidth=window_halfwidth)
    #Get best-fit polynomial
    mm = masked.mask
    xx = np.arange(masked.size)
    basepoly = output_basepoly(masked,max_order=max_order)
    bpc = len(basepoly.c)-1
    bp = basepoly(xx)
    xxx = np.arange(orig_spec.size)
    params = np.asarray(basepoly)
    #Refactor best-fit parameters for unsmoothed spectrum
    rr = filter_width
    newparams = []
    for k,p in enumerate(params[::-1]):
        newparams.append(p/rr**k)
    newparams = newparams[::-1]
    newpoly = np.poly1d(newparams)
    newbaseline = newpoly(xxx)
    #Subtract baseline and smooth spectrum
    final = im.median_filter(orig_spec - newbaseline,
                             filter_width)[::filter_width]
    rel_diff = get_rel_diff(sm - bp)
    return(rel_diff,final)

def fit_baseline(masked,xx,ndeg=2):
    """
    Fit the masked array with a polynomial of 
    the given order.
    """
    ya = ma.polyfit(xx,masked,ndeg)
    basepoly = np.poly1d(ya)
    return(basepoly)

def find_best_baseline(masked,xx,max_order=2,prior_penalty=1):
    """
    Consider polynomial baselines up to order max_order.
    Select the baseline with the lowest reduced chi-squared, 
    where prior_penalty can be used to increase the penalty
    for fitting with a higher order baseline. 
    """
    chisqs = np.zeros(max_order+1)
    ndegs = np.arange(max_order+1)
    for i,ndeg in enumerate(ndegs):
        basepoly = fit_baseline(masked,xx,ndeg=ndeg)
        base = basepoly(xx)
        chisqs[i] = np.sum((masked-base)**2)/(ma.count(masked)
                                              -prior_penalty*ndeg)
    
    return(np.argmin(chisqs))
    
def output_basepoly(masked,max_order=2,prior_penalty=1.):
    """
    Returns the polynomial coefficients of the best-fit baseline model.
    """
    xx = np.arange(masked.size)
    xx = xx.astype(np.float32) #To avoid bug with ma.polyfit in np1.6
    npoly = find_best_baseline(masked,xx,max_order=max_order,
                               prior_penalty=prior_penalty)
    basepoly = fit_baseline(masked,xx,ndeg=npoly)

    return(basepoly)

def rolling_window(a,window):
    """
    Magic code to quickly create a second dimension
    with the elements in a rolling window. This
    allows us to apply numpy operations over this
    extra dimension MUCH faster than using the naive approach.
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)    
    strides = a.strides+(a.strides[-1],)
    return(np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides))

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

def mask_for_moment(spec,nsig=3.):
    """
    Mask the regions of a spectrum that do not contain signal for 
    the purpose of moment analysis. 
    """
    #Calculate the noise and mask channels with intensity less than nsig*noise
    noise = get_rms(spec)
    masked = ma.masked_where(np.logical_or(spec < nsig*noise,
                                           np.isnan(spec)),spec)
    #Locate the unmasked channels if present
    unmasked_loc = np.where(masked.mask == False)[0]
    absdiff = abs(np.diff(unmasked_loc))
    if 1 not in absdiff:
        masked = ma.masked_inside(spec,-np.inf,np.inf)
    else:
        """
        Real signal exhibits significant emission over several channels. Mask
        unmasked channels that are not contiguous with two other unmasked 
        channels.
        """
        for i,ii in enumerate(unmasked_loc):
            if i==0:
                crit = (absdiff[i] != 1)
            elif i==len(unmasked_loc)-1:
                crit = (absdiff[i-1] != 1)
            else:
                crit = (absdiff[i] != 1 and absdiff[i-1] != 1) 
            if crit:
                masked.mask[ii] = True
    return(masked)


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
    if ma.is_masked(masked):
        where_masked = np.where(masked.mask)[0]
        mask_extension = 10
        for channel in where_masked:
            masked[slice(channel-10,channel+10,1)] = ma.masked

    #Calculate the noise estimate from the masked array 
    noise = noise_est(masked)
    masked = ma.masked_where(np.logical_or(spec > nsig*noise,
                                           np.isnan(spec)),spec)
    #Mask 10 channels around each masked channel to capture faint line wings
    if ma.is_masked(masked):
        where_masked = np.where(masked.mask)[0]
        mask_extension = 10
        for channel in where_masked:
            masked[slice(channel-10,channel+10,1)] = ma.masked
    return(masked)

def get_rms(spec):
    if len(np.where(np.isnan(spec))[0]) == len(spec):
        rms = np.nan
    else:
        m = mask_for_rms(spec)
        rms = (m**2).mean()**(0.5)
    return(rms)

def noise_est(spec):
    """
    Estimates the noise in a spectrum by the 
    channel-to-channel differences.
    """
    diff = np.diff(spec)
    noise_est = np.sqrt(np.nanmean(diff*diff)/2.)
    return(noise_est)

def get_rel_diff(spec):
    """
    Calculates the relative difference between the 
    rms and the noise estimate from channel-to-channel
    differences. The rms is affected by residual baseline, 
    while the channel-to-channel differences are not. 
    Fitted spectra with larger rel_diff values have more
    significant residual baselines.
    """
    m = mask_for_rms(spec)
    rel_diff = 1-noise_est(m)/((np.nanmean(m**2))**(0.5))
    return(rel_diff)

def downsample_header(h,filter_width=1):
    """
    Since we downsample our spectra by a factor 
    of filter_width we have to change the header as well.
    filter_width here _needs_ to be the same as 
    filter_width in baseline_and_deglitch
    """
    h['CDELT3'] = h['CDELT3']*float(filter_width)
    h['CRPIX3'] = h['CRPIX3']/float(filter_width)
    return(h)

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
    
    
def change_to_velocity(h):
    """
    Change the spectral information in the header
    from frequency to LSR velocity.
    """
    if not ('RESTFREQ' in h):
        h.rename_keyword('RESTFRQ','RESTFREQ')
    else:
        pass
    n = (h['RESTFREQ']-h['CRVAL3'])/h['CDELT3']+h['CRPIX3']
    h['CTYPE3'] = 'VELO-LSR'
    delta_freq = h['CDELT3']
    h['CDELT3'] = 299792.458/float(h['CRVAL3'])*1000*delta_freq
    h['CRVAL3'] = 0.
    h['CRPIX3'] = h['NAXIS3']-int(n)
    try:
        del h['CUNIT3']
    except:
        pass
    try:
        del h['SPECSYS']
    except:
        pass
    return(h)
    
def sum_over_signal(spec,nsig=5.):
    """
    Sum over regions of significant signal.
    Run this on spectra that have already been
    baseline subtracted.
    """
    if np.isnan(spec).all():
        """
        If the spectrum is all nans, then it means we haven't
        mapped this location. Fill it with a large negative
        number to differentiate it from spectra that don't
        have significant signal.
        """
        spec_sum=-999
    else:
        masked = mask_for_moment(spec)
        """
        Sum the unmasked channels.
        """
        spec_sum = ma.sum(masked)
    return(spec_sum)

def first_moment(spec,h,nsig=5.):
    """
    Take the first moment over regions of significant signal.
    Run	this on	spectra	that have already been 
    baseline subtracted.
    """
    if np.isnan(spec).all():
        """
        If the spectrum is all nans, then it means we haven't
        mapped this location. Fill it with a large negative
        number to differentiate it from spectra that don't
        have significant signal.
        """
        mom1 = -999
    else:
        #Mask out noise for first moment
        masked = mask_for_moment(spec)
        """
        Create the velocity axis and calculate the first moment 
        using the unmasked channels.
        """
        vax = get_vax(h)
        mom1 = (masked*vax).sum()/masked.sum()
    return(mom1)

def c2v(c,h):
    """
    Use the header information to translate from channels to velocity.
    """
    v = ((c - (h['CRPIX3']-1))*h['CDELT3'] + h['CRVAL3'])*0.001
    return(v)

def v2c(v,h):
    """
    Use the header information to translate from velocity to channels.
    """
    c = ((v*1000. - h['CRVAL3'])/h['CDELT3'] + h['CRPIX3']-1)
    return(int(round(c)))

def get_vax(h):
    """
    Use the header information to calculate the velocity axis of a spectrum.
    """
    vmin = c2v(0,h)
    vmax = c2v(h['NAXIS3']-1,h)
    vax = np.linspace(vmin,vmax,h['NAXIS3'])
    return(vax)

if __name__ == '__main__':
    main()

