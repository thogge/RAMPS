="""
prefit_raw_data_baselines.py

Fit baselines and remove glitches/spikes from RAMPS data.
Optionally transforms to velocity and outputs a (masked)
moment zero (integrated intensity map) and first moment
(velocity field) 

This version runs in parallel, which is useful because
the process is fairly slow.

Example:
python prefit_raw_data_baselines.py 
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
    #pdb.set_trace()
    #test = baseline_and_deglitch(d[:,61,711],filter_width=filter_width,do_vel=do_vel)
    #test = sum_over_signal(d[:,196,729])
    #test = first_moment(d[:,106,135],h)
    if do_fit:
        #Split the data
        s = np.array_split(d, numcores, 2)
        ps = []
        #Fit baselines and write to temporary files
        for num in range(len(s)):
            ps.append(multiprocessing.Process(target=do_chunk_fit,
                                              args=(num,s[num],
                                                    filter_width,
                                                    do_vel)))
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        #Recombine baselined temporary files 
        dout = recombine(numcores)

        if do_vel:
            #Edit header to put spectral axis in velocity space
            hout = downsample_header(change_to_velocity(strip_header(h[:],4)),
                                     filter_width)
            old_ref_channel =  float(hout['CRPIX3'])
            hout['CRPIX3'] = float(len(dout[:,1,1]))-old_ref_channel
        else:
            hout = downsample_header(strip_header(h[:],4),filter_width)
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
            ps.append(multiprocessing.Process(target=do_chunk_mom0,
                                              args=(num,s[num],hout)))
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
            ps.append(multiprocessing.Process(target=do_chunk_mom1,
                                              args=(num,s[num],hout)))
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
        try:
            mom1_min = np.min(good_data_mom1)
            mom1_max = np.max(good_data_mom1)
        except:
            mom1_min = 0.
            mom1_max = 0.            
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
    print num
    nax1 = data.shape[2]
    nax2 = data.shape[1]
    #nax3 = int(round(data.shape[0]/filter_width))
    nax3 = len(im.median_filter(data[:,0,0],filter_width)[::filter_width])
    ya = np.full((nax3,nax2,nax1,),np.nan)
    for i in range(nax2):
        for j in range(nax1):
            print i,j
            ya[:,i,j] = baseline_and_deglitch(data[:,i,j],
                                              filter_width=filter_width,
                                              do_vel=do_vel)
            """
            try:
                ya[:,i,j] = baseline_and_deglitch(data[:,i,j],
                                                  filter_width=filter_width,
                                                  do_vel=do_vel)
            except:
                pass
            """
    """
    ya =  np.apply_along_axis(baseline_and_deglitch,0,data,                 
                              filter_width=filter_width,
                              do_vel=do_vel)
    """
    print num
    pyfits.writeto("temp"+str(num)+".fits",ya,clobber=True)
    
    
def do_chunk_mom0(num,data,header):
    """
    Use apply_along_axis to calculate 
    the integrated intensity of each 
    spectrum in this chunk of the cube.
    """
    ya = np.full(data[0,:,:].shape,np.nan)
    for i in range(ya.shape[0]):
        for j in range(ya.shape[1]):
            ya[i,j] = sum_over_signal(data[:,i,j])
            if ya[i,j] > 0.:
                print ya[i,j]
    #Convert to K km/s. Assumes CTYPE3 is LSR Velocity
    loc = np.where(ya>0.)
    #print np.nansum(ya[loc])
    good_data_locations = np.where(ya > 0.)
    ya[good_data_locations] *= 0.001*abs(header['CDELT3'])
    pyfits.writeto("temp"+str(num)+".fits",ya,clobber=True)
    
    
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
    diff = np.diff(spec)
    noise_est = np.sqrt(np.nanmean(diff*diff)/2.)
    return noise_est


def fit_baseline(masked,xx,ndeg=2):
    """
    Fit the masked array with a polynomial of 
    the given order.
    """
    ya = ma.polyfit(xx,masked,ndeg)
    basepoly = np.poly1d(ya)
    return(basepoly)

def find_best_baseline(masked,xx,max_order=1,prior_penalty=1):
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
                          do_vel=False,
                          vv=40,
                          ww=20,
                          sigma_cut=1.5,
                          poly_n=2.,
                          filter_width=1,
                          max_order=2,
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
    (4) Remove any large negative spikes that are still 
        present after filtering. 
    """
    """
    if filter_width == 1:
        smoothed = orig_spec[:]
    else:
        smoothed = im.median_filter(orig_spec,filter_width)[::filter_width]
    """
    smoothed = im.median_filter(orig_spec,filter_width)[::filter_width]

    #First mask
    ya1 = rolling_window(smoothed,ww*2)
    #Calculate standard dev and pad the output
    stds1 = my_pad.pad(np.std(ya1,-1),(ww-1,ww),mode='edge')

    #Figure out which bits of the spectrum have signal/glitches
    med_std1 = np.nanmedian(stds1)
    std_std1 = np.nanstd(stds1)
    sigma_x_bar1 = med_std1/np.sqrt(ww)
    sigma_s1 = (1./np.sqrt(2.))*sigma_x_bar1

    #Second mask
    ya2 = rolling_window(smoothed,vv*2)
    #Calculate standard dev and pad the output
    stds2 = my_pad.pad(np.std(ya2,-1),(vv-1,vv),mode='edge')

    #Figure out which bits of the spectrum have signal/glitches
    med_std2 = np.nanmedian(stds2)
    std_std2 = np.nanstd(stds2)
    sigma_x_bar2 = med_std1/np.sqrt(vv)
    sigma_s2 = (1./np.sqrt(2.))*sigma_x_bar2

    #Mask out signal for baseline
    masked = ma.masked_where(np.logical_or(np.logical_or(stds1 > med_std1+sigma_cut*sigma_s1,stds2 > med_std2+sigma_cut*sigma_s2),np.isnan(smoothed)),smoothed)
    """
    plt.plot(masked)
    plt.show()
    pdb.set_trace()
    """
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
    #pdb.set_trace()
    """
    if filter_width == 1:
        filtered_spectrum = orig_spec[:]
    else:
        filtered_spectrum = im.median_filter(sub,filter_width)[::filter_width]
    """
    filtered_spectrum = im.median_filter(sub,filter_width)[::filter_width]
    if do_vel:
        #Reverse spectral array so positive velocity is on the right
        final = filtered_spectrum[::-1]
    else:
	final = filtered_spectrum
    """
    plt.plot(final)
    plt.show()
    pdb.set_trace()
    """
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
    if not ('RESTFREQ' in h):
        h.rename_keyword('RESTFRQ','RESTFREQ')
    else:
        pass
    n = (h['RESTFREQ']-h['CRVAL3'])/h['CDELT3']+h['CRPIX3']
    h['CTYPE3'] = 'VELO-LSR'
    delta_freq = h['CDELT3']
    h['CDELT3'] = 299792.458/float(h['CRVAL3'])*1000*delta_freq
    h['CRVAL3'] = 0.
    h['CRPIX3'] = int(n)
    try:
        del h['CUNIT3']
    except:
        pass
    try:
        del h['SPECSYS']
    except:
        pass
    return(h)
    

def sum_over_signal(spec,nsig=3.):
    """
    Sum over regions of significant signal.
    Could be modified to output a mask in order
    to enable quick calculation of multiple moments.
    Run this on spectra that have already been
    baseline subtracted.
    """
    #print(noise_est(spec))
    if np.isnan(noise_est(spec)):
        """
        If the spectrum is all nans, then it means we haven't
        mapped this location. Fill it with a large negative
        number to differentiate it from spectra that don't
        have significant signal.
        """
        spec_sum=-999
    else:
        sigma = noise_est(spec)
        #print sigma
        #Mask out noise for integrated intensity
        masked = ma.masked_where(np.logical_or(spec < nsig*sigma,np.isnan(spec)),spec)
        #print masked.sum()
        #import pdb
        #pdb.set_trace()
        """
        Require that each unmasked channel be contiguous with two
        other unmasked channels. This prevents the inclusion of three 
        sigma channels that are purely noise.
        """
        temp_masked= ma.zeros(ma.shape(masked))
        w = 1
	masked[0:w+9] = ma.masked
        masked[len(masked)-w-9:len(masked)] = ma.masked
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
#        if np.isfinite(spec_sum):
#            print spec_sum
        #pdb.set_trace()
    return(spec_sum)

def first_moment(spec,h,nsig=3.):
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
        sigma = noise_est(spec)
        #Mask out noise for first moment
        masked = ma.masked_where(np.logical_or(spec < nsig*sigma,np.isnan(spec)),spec)

        """
	Require that each unmasked channel be contiguous with two
        other unmasked channels. This prevents the inclusion of three
        sigma channels that are purely noise.
        """
	temp_masked= ma.zeros(ma.shape(masked))
        w = 1
        masked[0:w+9] = ma.masked
        masked[len(masked)-w-9:len(masked)] = ma.masked
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
        vel_min = 0.001*(1 - h['CRPIX3'])*h['CDELT3'] + h['CRVAL3']
        vel_max = 0.001*(len(masked) - h['CRPIX3'])*h['CDELT3'] + h['CRVAL3']
        vel = np.linspace(vel_min,vel_max,len(masked),dtype=np.float)
        for i,v in enumerate(vel):
            temp_masked.data[i] = masked.data[i]*v
        mom1 = ma.sum(temp_masked)/ma.sum(masked)
    return(mom1)
if __name__ == '__main__':
    main()

