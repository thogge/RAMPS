"""
crop_cube_spectra.py

Crop a data cube along the spectral axis. The default units for the 
new start and end values for the spectral crop are channels, but can
optionally be given as velocities (units of km/s) or frequency 
(units of Hz).


Example:
python crop_cube_spectra.py -i L30_Tile01_NH3_1-1_cube.fits
                            -o L30_Tile01_NH3_1-1_cube_cropped.fits
                            -s -30 
                            -e 140
                            -v

-i : Input file     -- Input file
-o : Output file    -- Output file
-s : Crop start     -- New starting channel/velocity/frequency
-e : Crop end       -- New ending channel/velocity/frequency
-v : Velocity Crop  -- Flag to signal that the crop start and end values
                       refer to velocities with units of km/s
-f : Frequency Crop -- Flag to signal that the crop start and end values
                       refer to frequencies with units of Hz
-h : Help           -- Display this help

"""

import sys,os,getopt
import astropy.io.fits as pyfits
import numpy as np
import math
import pdb


def main():
    #Defaults
    vel_crop = False
    freq_crop = False
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:o:s:e:vfh")
    except getopt.GetoptError:
        print("Invalid arguments")
        print(__doc__)
        sys.exit(2)
    for o,a in opts:
        if o == "-i":
            input_file = a
        elif o == "-o":
            output_file = a
        elif o == "-s":
            crop_start = a
        elif o == "-e":
            crop_end = a
        elif o == "-v":
            vel_crop = True
        elif o == "-f":
            freq_crop = True
        elif o == "-h":
            print(__doc__)
            sys.exit(1)
        else:
            assert False, "unhandled option"
            print(__doc__)
            sys.exit(2)

    #Read the data and remove extra dimensions
    d,h = pyfits.getdata(input_file,header=True)
    if len(d.shape) > 3:
        d = np.squeeze(d)
        h = strip_header(h,4)

    """
    If crop start and crop end are given in terms of frequency or velocity, 
    translate these values to channels
    """
    if vel_crop:
        """
        If header does not contain velocity info, translate from 
        velocity to frequency before converting to channels.
        """
        if ('FREQ' or 'freq') in h['CTYPE3']:
            new_crop_start = freq_to_chan(vel_to_freq(float(crop_start),h),h)
            new_crop_end = freq_to_chan(vel_to_freq(float(crop_end),h),h)
        elif ('VEL' or 'vel') in h['CTYPE3']:
            new_crop_start = vel_to_chan(float(crop_start),h)
            new_crop_end = vel_to_chan(float(crop_end),h)
        else:
            print("Unrecognized header info")
            sys.exit(2)
   elif freq_crop:
        """
        If header does not contain frequency info, translate from 
        frequency to velocity before converting to channels.
        """
        if ('FREQ' or 'freq') in h['CTYPE3']:
            new_crop_start = freq_to_chan(float(crop_start),h)
            new_crop_end = freq_to_chan(float(crop_end),h)
        elif ('VEL' or 'vel') in h['CTYPE3']:
            new_crop_start = vel_to_chan(freq_to_vel(float(crop_start),h),h)
            new_crop_end = vel_to_chan(freq_to_vel(float(crop_end),h),h)
        else:
            print("Unrecognized header info")
            sys.exit(2)
    else:
        new_crop_start = int(crop_start)
        new_crop_end = int(crop_end)

    """
    If the new starting and ending channels are outside of the possible
    range, set them to the closest possible values.
    """
    if new_crop_start < 0:
        new_crop_start = 0
    if new_crop_end > d.shape[0]:
        new_crop_end = d.shape[0]

    #Crop the data cube and write to output file
    print(new_crop_start,new_crop_end)
    d_crop = d[int(new_crop_start):int(new_crop_end),:,:]
    h_crop = fix_header(int(new_crop_start),int(new_crop_end),h)
    pyfits.writeto(output_file,d_crop,h_crop,clobber=True)
    

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
    
def chan_to_freq(channel,h):
    """
    Use the header information to translate from channels to frequency.
    """
    frequency = (channel - h['CRPIX3'])*h['CDELT3'] + h['CRVAL3']
    return(frequency)

def freq_to_chan(frequency,h):
    """
    Use the header information to translate from frequency to channels.
    """
    channel = (frequency - h['CRVAL3'])/h['CDELT3'] + h['CRPIX3']
    return(int(round(channel)))

def chan_to_vel(channel,h):
    """
    Use the header information to translate from channels to velocity.
    """
    velocity = ((channel - h['CRPIX3'])*h['CDELT3'] + h['CRVAL3'])*0.001
    return(velocity)

def vel_to_chan(velocity,h):
    """
    Use the header information to translate from velocity to channels.
    """
    channel = ((velocity*1000. - h['CRVAL3'])/h['CDELT3'] + h['CRPIX3'])
    return(int(round(channel)))

def freq_to_vel(frequency,h):
    """
    Use the rest frequency to translate from frequency to velocity.
    """
    if 'RESTFREQ' in h:
        f0 = h['RESTFREQ']
    elif 'RESTFRQ' in h:
        f0 = h['RESTFRQ']
    velocity = 3e5*(1-(frequency/f0))
    return(velocity)

def vel_to_freq(velocity,h):
    """
    Use the rest frequency to translate from velocity to frequency.
    """
    if 'RESTFREQ' in h:
        f0 = h['RESTFREQ']
    elif 'RESTFRQ' in h:
        f0 = h['RESTFRQ']
    frequency = (f0)*(1-(velocity/3e5))
    return(frequency)

def fix_header(channel_start,channel_end,h):
    """
    Change the necessary header keys after cropping to keep the channel 
    values accurate.
    """
    h['NAXIS3'] = channel_end - channel_start 
    h['CRPIX3'] = h['CRPIX3'] - channel_start
    return(h)

if __name__ == '__main__':
    main()
