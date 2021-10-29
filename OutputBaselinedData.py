#!/usr/bin/env python
# encoding: utf-8
"""
OutputBaselinedData.py

Fit baselines and remove glitches/spikes from RAMPS data.
Transforms to velocity axis. Outputs mask, baseline, baselined
data, relative diff between rms and sig_diff, oversmoothed data 
used for fit, and the polynomial order of baseline fit. Optionally 
outputs a (masked) moment zero (integrated intensity map) and 
first moment (velocity field).

This version runs in parallel, which is useful because
the process is fairly slow.

Example:
python OutputBaselinedData.py 
       -i L30_Tile01_23694_MHz_line.fits 
       -o L30_Tile01 -s 11 -v 6 -b 55 -n 16 -p 2 -01

-i : Input      -- Input file (reduced by pipeline)
-o : Output     -- Output file 
-s : Smooth     -- Size of kernel for median smooth
-b : Big smooth -- Size of oversmoothing kernel for baseline fit
-n : Numcores   -- Number of cores available for parallized computing
-v : Windwidth  -- 1/2 width of rolling window used for baseline fit mask
-p : Max order  -- Highest order polynomial with which to fit
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
import pdb
from datetime import datetime

def main():

    #Defaults
    output_base = "default.fits"
    numcores = 1
    max_order = 2
    filter_width = 1
    big_filter_width = 1
    vv = 15
    do_fit = False
    do_mom0 = False
    do_mom1 = False
    keep = False
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:o:s:b:n:u:v:w:p:m:f01kh")
    except getopt.GetoptError:
        print("Invalid key")
        print(__doc__)
        sys.exit(2)
    for o,a in opts:
        if o == "-i":
            input_file = a
        elif o == "-o":
            output_base = a
        elif o == "-s":
            filter_width = int(a)
        elif o == "-b":
            big_filter_width = int(a)
        elif o == "-n":
            numcores = int(a)
        elif o == "-v":
            vv = int(a)
        elif o == "-p":
            max_order = int(a)
        elif o == "-f":
            do_fit = True
        elif o == "-0":
            do_mom0 = True
        elif o == "-1":
            do_mom1 = True
        elif o == "-k":
            keep = True
        elif o == "-h":
            print(__doc__)
            sys.exit(1)
        else:
            assert False, "unhandled option"
            print(__doc__)
            sys.exit(2)
    t1 = datetime.now()
    #Read in data into array, remove single-dimensional entries
    d,h = pyfits.getdata(input_file,header=True,memmap=False)
    d = np.squeeze(d)
    if do_fit:
        #Fit baselines and write to temporary files
        if numcores > 1:
            s = np.array_split(d, numcores, 2)
            ps = []
            for num in range(len(s)):
                ps.append(multiprocessing.Process(target=do_chunk,
                                                  args=(num,s[num],vv,
                                                        max_order,
                                                        filter_width,
                                                        big_filter_width,
                                                        keep)))
            for p in ps:
                p.start()
            for p in ps:
                p.join()
        else:
            do_chunk(0,d,vv,max_order,filter_width,big_filter_width,keep)
        #Recombine baselined temporary files 
        if keep:
            mask_cube = recombine_a(numcores)
            baseline_cube = recombine_b(numcores)
            temp_cube = recombine_c(numcores)
            rdiff_map = recombine_d(numcores)
            npoly_map = recombine_e(numcores)
            fixed_cube = recombine_f(numcores)
        else:
            fixed_cube = recombine_f(numcores)
    
        #Edit headers, write data
        if keep:
            hout_temp = downsample_header(change_to_velocity(strip_header(h[:],4)),big_filter_width)
            hout_temp['DATAMIN'] = -3
            hout_temp['DATAMAX'] = 3
            pyfits.writeto(output_base+'_mask.fits',mask_cube,hout_temp,overwrite=True)
            pyfits.writeto(output_base+'_baseline.fits',baseline_cube,hout_temp,overwrite=True)
            pyfits.writeto(output_base+'_temp.fits',temp_cube,hout_temp,overwrite=True)
            hout = downsample_header(change_to_velocity(strip_header(h[:],4)),filter_width)
            pyfits.writeto(output_base+'_fixed.fits',fixed_cube,hout,overwrite=True)
            hout2 = strip_header(hout[:],3)
            hout2['DATAMIN'] = -0.05
            hout2['DATAMAX'] = 0.05
            pyfits.writeto(output_base+'_rdiff.fits',rdiff_map,hout2,overwrite=True)
            hout2['DATAMIN'] = 0
            hout2['DATAMAX'] = np.nanmax(npoly_map)
            pyfits.writeto(output_base+'_npoly.fits',npoly_map,hout2,overwrite=True)
            os.system("rm temp*")
        else:
            hout = downsample_header(change_to_velocity(strip_header(h[:],4)),filter_width)
            if '.fits' in output_base:
                pyfits.writeto(output_base,fixed_cube,hout,overwrite=True)
            else:
                pyfits.writeto(output_base+'_fixed.fits',fixed_cube,hout,overwrite=True)
            os.system("rm tempf*")

    else:
        fixed_cube = d
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

        if numcores > 1:
            #Split the data
            s = np.array_split(fixed_cube, numcores, 2)
            ps = []
            #Create integrated intensity maps and write to temporary files
            for num in range(len(s)):
                ps.append(multiprocessing.Process(target=do_chunk_mom0,
                                                  args=(num,s[num],hout)))
            for p in ps:
                p.start()
                
            for p in ps:
                p.join()
        else:          
            do_chunk_mom0(0,fixed_cube,hout)
       #Recombine temporary files
        mom0 = recombine_moments(numcores)
        mom0[np.where(mom0 == 0)] = np.nan
        mom0_file = output_base+"_mom0.fits"  
        #Update header and write the data
        hmom = strip_header(hout[:],3)
        hmom['BUNIT'] = 'K*km/s'
        hmom['DATAMIN'] = 0.
       	hmom['DATAMAX'] = 20.
        pyfits.writeto(mom0_file,mom0,hmom,overwrite=True)  
        #Remove temporary files
        os.system("rm temp*")

        
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

        if numcores > 1:
            #Split the data
            s = np.array_split(fixed_cube, numcores, 2)
            ps = []
            #Create velocity maps and write to temporary files
            for num in range(len(s)):
                ps.append(multiprocessing.Process(target=do_chunk_mom1,
                                                  args=(num,s[num],hout)))
            for p in ps:
                p.start()
        
            for p in ps:
                p.join()
        else:
            do_chunk_mom1(0,fixed_cube,hout)
        #Recombine temporary files
        mom1 = recombine_moments(numcores)
        mom1[np.where(mom1 == 0)] = np.nan
        mom1_file = output_base+"_mom1.fits"
        #Update header and write the data
        hmom = strip_header(hout[:],3)
        hmom['BUNIT'] = 'km/s'
        good_data_mom1 = mom1[np.where(np.isfinite(mom1))]
        try:
            mom1_min = np.min(good_data_mom1)
            mom1_max = np.max(good_data_mom1)
        except:
            mom1_min = 0.
            mom1_max = 0.            
        hmom['DATAMIN'] = mom1_min
        hmom['DATAMAX'] = mom1_max
        pyfits.writeto(mom1_file,mom1,hmom,overwrite=True)
        #Remove temporary files
        os.system("rm temp*")

    t2 = datetime.now()
    print('Elapsed time: '+str(t2-t1))

def do_chunk(num,data,vv,max_order,filter_width,big_filter_width,keep):
    """
    Use apply_along_axis to apply 
    baseline fitting to each spectrum in 
    this chunk of the cube.
    """
    #pdb.set_trace()
    print(num)
    nax1 = data.shape[2]
    nax2 = data.shape[1]
    nax3 = len(data[:,0,0][::filter_width])
    if keep:
        ya = np.full((nax3,nax2,nax1,),np.nan)
        yb = np.full((nax3,nax2,nax1,),np.nan)
        yc = np.full((nax3,nax2,nax1,),np.nan)
        yd = np.full((nax2,nax1,),np.nan)
        ye = np.full((nax2,nax1,),np.nan)
        yf = np.full((nax3,nax2,nax1,),np.nan)
    else:
        yd = np.full((nax2,nax1,),np.nan)
        yf = np.full((nax3,nax2,nax1,),np.nan)
    rel_err_sig = 0.00404*(filter_width)**(0.5)
    nsig = 5.
    for i in range(nax2):
        for j in range(nax1):
            #if j==0:
            #print (float(i)*nax1+float(j)+1)/(nax1*nax2)
            orig_spec = data[::-1,i,j]
            #print num, i,j, orig_spec[len(orig_spec)/2]
            #pdb.set_trace()
            if keep:
                if np.isfinite(orig_spec[len(orig_spec)/2]):
                    ya[:,i,j],yb[:,i,j],yc[:,i,j],yd[i,j],ye[i,j],yf[:,i,j] = fit_wrapper(orig_spec,vv,max_order,filter_width,filter_width,keep)
                    if yd[i,j] > nsig*rel_err_sig:
                        #print 'test1'
                        ya[:,i,j],yb[:,i,j],yc[:,i,j],yd[i,j],ye[i,j],yf[:,i,j] = fit_wrapper(orig_spec,3*vv,max_order,filter_width,filter_width,keep)
                        if yd[i,j] > nsig*rel_err_sig:
                            #print 'test2'
                            ya[:,i,j],yb[:,i,j],yc[:,i,j],yd[i,j],ye[i,j],yf[:,i,j] = fit_wrapper(orig_spec,3*vv,0,filter_width,filter_width,keep)
            else:
                if np.isfinite(orig_spec[len(orig_spec)/2]):
                    yd[i,j],yf[:,i,j] = fit_wrapper(orig_spec,vv,max_order,filter_width,filter_width,keep)
                    if yd[i,j] > nsig*rel_err_sig:
                        #print 'test1'
                        yd[i,j],yf[:,i,j] = fit_wrapper(orig_spec,3*vv,max_order,filter_width,filter_width,keep)
                        if yd[i,j] > nsig*rel_err_sig:
                            #print 'test2'
                            yd[i,j],yf[:,i,j] = fit_wrapper(orig_spec,3*vv,0,filter_width,filter_width,keep)
    if keep:
        pyfits.writeto("tempa"+str(num)+".fits",ya,overwrite=True)
        pyfits.writeto("tempb"+str(num)+".fits",yb,overwrite=True)
        pyfits.writeto("tempc"+str(num)+".fits",yc,overwrite=True)
        pyfits.writeto("tempd"+str(num)+".fits",yd,overwrite=True)
        pyfits.writeto("tempe"+str(num)+".fits",ye,overwrite=True)
        pyfits.writeto("tempf"+str(num)+".fits",yf,overwrite=True)
    else:
        pyfits.writeto("tempf"+str(num)+".fits",yf,overwrite=True)
    
def do_chunk_mom0(num,data,header):
    """
    Use apply_along_axis to calculate 
    the integrated intensity of each 
    spectrum in this chunk of the cube.
    """
    ya = np.full(data[0,:,:].shape,np.nan)
    hw = len(data[:,0,0])/2.
    for i in range(ya.shape[0]):
        for j in range(ya.shape[1]):
            #print(i,j)
            if np.isfinite(data[int(hw),i,j]):
                ya[i,j] = sum_over_signal(data[:,i,j])
                #pdb.set_trace()
            if ya[i,j] > 0.:
                #print ya[i,j]
                print(i,j)
    #Convert to K km/s. Assumes CTYPE3 is LSR Velocity
    loc = np.where(ya>0.)
    #print np.nansum(ya[loc])
    good_data_locations = np.where(ya > 0.)
    ya[good_data_locations] *= 0.001*abs(header['CDELT3'])
    pyfits.writeto("temp"+str(num)+".fits",ya,overwrite=True)
      
def do_chunk_mom1(num,data,header):
    """
    Use apply_along_axis to calculate 
    the integrated intensity of each 
    spectrum in this chunk of the cube.
    """
    ya = np.full(data[0,:,:].shape,np.nan)
    for i in range(ya.shape[0]):
        for j in range(ya.shape[1]):
            ya[i,j] = first_moment(data[:,i,j],header)
            if ya[i,j] > -999.:
                print(ya[i,j])
    pyfits.writeto("temp"+str(num)+".fits",ya,overwrite=True)
    
def recombine_a(numparts,output_base="test_final.fits"):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    for n in range(numparts):
        d = pyfits.getdata("tempa"+str(n)+".fits")
        indata.append(d)
    final = np.dstack(indata)
    return(final)
    
def recombine_b(numparts,output_base="test_final.fits"):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    for n in range(numparts):
        d = pyfits.getdata("tempb"+str(n)+".fits")
        indata.append(d)
    final = np.dstack(indata)
    return(final)
    
def recombine_c(numparts,output_base="test_final.fits"):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    for n in range(numparts):
        d = pyfits.getdata("tempc"+str(n)+".fits")
        indata.append(d)
    final = np.dstack(indata)
    return(final)
    
def recombine_d(numparts,output_base="test_final.fits"):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    for n in range(numparts):
        d = pyfits.getdata("tempd"+str(n)+".fits")
        indata.append(d)
    final = np.column_stack(indata)
    return(final)
    
def recombine_e(numparts,output_base="test_final.fits"):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    for n in range(numparts):
        d = pyfits.getdata("tempe"+str(n)+".fits")
        indata.append(d)
    final = np.column_stack(indata)
    return(final)
    
def recombine_f(numparts,output_base="test_final.fits"):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    for n in range(numparts):
        d = pyfits.getdata("tempf"+str(n)+".fits")
        indata.append(d)
    final = np.dstack(indata)
    return(final)
    
def recombine_moments(numparts,output_base="test_final.fits"):
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
    
def fit_wrapper(orig_spec,vv,max_order,filter_width,big_filter_width,keep):
    #pdb.set_trace()
    m = ma.masked_invalid(orig_spec)
    sm = im.median_filter(m,filter_width)[::filter_width]
    masked= mask_for_baseline(sm,vv=vv)
    mm = masked.mask
    xx = np.arange(masked.size)
    basepoly = output_basepoly(masked,max_order=max_order)
    bpc = len(basepoly.c)-1
    bp = basepoly(xx)
    xxx = np.arange(orig_spec.size)
    params = np.asarray(basepoly)
    rr = filter_width
    newparams = []
    for k,p in enumerate(params[::-1]):
        newparams.append(p/rr**k)
    newparams = newparams[::-1]
    newpoly = np.poly1d(newparams)
    newbaseline = newpoly(xxx)
    final = im.median_filter(orig_spec - newbaseline,filter_width)[::filter_width]
    pd = rdiff(sm - bp,vv=vv)
    #pdb.set_trace()
    if keep:
        return mm,bp,sm,pd,bpc,final
    else:
        return pd,final

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
        #print ndeg
        basepoly = fit_baseline(masked,xx,ndeg=ndeg)
        #print basepoly
        base = basepoly(xx)
        chisqs[i] = np.sum((masked-base)**2)/(ma.count(masked)
                                              -prior_penalty*ndeg)
    
    return(np.argmin(chisqs))
    
def output_basepoly(masked,
                    max_order=2,
                    prior_penalty=1.):

    xx = np.arange(masked.size)
    xx = xx.astype(np.float32) #To avoid bug with ma.polyfit in np1.6
    npoly = find_best_baseline(masked,xx,max_order=max_order)
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
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def mask_for_baseline(spec,sigma_cut=1.5,vv=30):
    ya = rolling_window(spec,vv*2)
    #Calculate standard dev and pad the output
    stds = my_pad.pad(np.nanstd(ya,-1),(vv-1,vv),mode='edge')

    #Figure out which bits of the spectrum have signal/glitches
    med_std = np.nanmedian(stds)
    std_std = np.nanstd(stds)
    sigma_x_bar = med_std/np.sqrt(vv)
    sigma_s = (1./np.sqrt(2.))*sigma_x_bar

    #Mask out signal for baseline
    masked = ma.masked_where(np.logical_or(stds > med_std+sigma_cut*sigma_s,np.isnan(spec)),spec)
    #if ma.is_masked(masked):
    #    masked.mask[1:-1] = ~(~masked.mask[2:]*~masked.mask[:-2]*~masked.mask[1:-1])
    temp_masked= ma.zeros(ma.shape(masked))
    w = 10
    if ma.is_masked(masked):
        for i in range(w,len(masked)-w):
            if (ma.getmask(masked)[i] == True):
                temp_masked[i-w:i+w+1] = ma.masked
    masked = ma.masked_where(np.ma.getmask(temp_masked), masked)
    return masked

def mask_for_moment(spec,nsig=3.):
    sigma = rms(spec)
    masked = ma.masked_where(np.logical_or(spec < nsig*sigma,np.isnan(spec)),spec)
    ww = np.where(masked.mask == False)[0]
    absdiff = abs(np.diff(ww))
    if 1 not in absdiff:
        masked = ma.masked_inside(spec,-np.inf,np.inf)
    else:
        for i,ii in enumerate(ww):
            if i==0:
                crit = (absdiff[i] != 1)
            elif i==len(ww)-1:
                crit = (absdiff[i-1] != 1)
            else:
                crit = (absdiff[i] != 1 and absdiff[i-1] != 1) 
            if crit:
                masked.mask[ii] = True
    return masked


def mask_for_rms(spec,nsig=3.):
    sigma = noise_est(spec)
    masked = ma.masked_where(np.logical_or(spec > nsig*sigma,np.isnan(spec)),spec)
    temp_masked= ma.zeros(ma.shape(masked))
    w = 10
    if ma.is_masked(masked):
        for i in range(w,len(masked)-w):
            if (ma.getmask(masked)[i] == True):
                temp_masked[i-w:i+w+1] = ma.masked
    masked = ma.masked_where(np.ma.getmask(temp_masked), masked)

    sigma = noise_est(masked)
    masked = ma.masked_where(np.logical_or(spec > nsig*sigma,np.isnan(spec)),spec)
    temp_masked= ma.zeros(ma.shape(masked))
    if ma.is_masked(masked):
        for i in range(w,len(masked)-w):
            if (ma.getmask(masked)[i] == True):
                temp_masked[i-w:i+w+1] = ma.masked
    masked = ma.masked_where(np.ma.getmask(temp_masked), masked)
    return(masked)

def rms(spec):
    if len(np.where(np.isnan(spec))[0]) == len(spec):
        rms = np.nan
    else:
        m = mask_for_rms(spec)
        antimask = (m.mask-1)*-1
        s = m.data*antimask
        s[np.where(s == 0)]=np.nan
        rms = (np.nanmean(s**2))**(0.5)
    return rms

def noise_est(spec):
    """
    Estimates the noise in a spectrum by the 
    channel-to-channel variations. Perform this
    function on only the inner portion of the 
    spectrum to avoid any residual single channel
    birdies in the data.
    """
    diff = np.diff(spec)
    noise_est = np.sqrt(np.nanmean(diff*diff)/2.)
    return noise_est

def rdiff(spec,vv):
    #m = mask_for_baseline(spec,vv=vv)
    m = mask_for_rms(spec)
    rd = 1-noise_est(m)/((np.nanmean(m**2))**(0.5))
    #pdb.set_trace()
    return rd

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
    Could be modified to output a mask in order
    to enable quick calculation of multiple moments.
    Run this on spectra that have already been
    baseline subtracted.
    """
    #print(noise_est(spec))
    if len(np.where(np.isnan(spec))[0]) == len(spec):
        """
        If the spectrum is all nans, then it means we haven't
        mapped this location. Fill it with a large negative
        number to differentiate it from spectra that don't
        have significant signal.
        """
        spec_sum=-999
    else:
        masked = mask_for_moment(spec)
        spec_sum = ma.sum(masked)

    return(spec_sum)

def first_moment(spec,h,nsig=5.):
    """
    Take the first moment over regions of significant signal.
    Run	this on	spectra	that have already been 
    baseline subtracted.
    """
    if np.isnan(noise_est(spec)):
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
        Create a masked spectrum of data*velocity to calculate 
        the first moment.
        """
        temp_masked = ma.zeros(ma.shape(masked))
        temp_masked.mask = ma.getmask(masked)
        vel_min = 0.001*(1 - h['CRPIX3'])*h['CDELT3'] + h['CRVAL3']
        vel_max = 0.001*(len(masked) - h['CRPIX3'])*h['CDELT3'] + h['CRVAL3']
        vel = np.linspace(vel_min,vel_max,len(masked),dtype=np.float)
        for i,v in enumerate(vel):
            temp_masked.data[i] = masked.data[i]*v
        mom1 = ma.sum(temp_masked)/ma.sum(masked)
        #pdb.set_trace()
    return(mom1)

if __name__ == '__main__':
    main()

