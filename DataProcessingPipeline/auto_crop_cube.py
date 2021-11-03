"""
auto_crop_cube.py

Uses pyspeckit to crop the nans from the top and bottom 
of a cube, leaving a rectangular shaped cube with only 
finite values. This is meant to crop partial tiles
from RAMPS and is not suited to crop pixels in 
the x direction. Can also crop along the
spectral axis by giving the new first and last
channels/frequencies/velocities.

Example:
python auto_crop_cube.py -i L30_Tile01.fits
                         -o L30_Tile01_cropped.fits
                         -s 800 
                         -e 16000



-i : Input          -- Input file (reduced by pipeline)
-o : Output         -- Output file
-s : start          -- The new start channel
-e : end            -- The new end channel
-v : Velocity Crop  -- Flag to signal that the crop start and end values
                       refer to velocities with units of km/s
-f : Frequency Crop -- Flag to signal that the crop start and end values
                       refer to frequencies with units of Hz
-h : Help           -- Display this help

"""

import sys,os,getopt
import astropy.io.fits as fits
import numpy as np
from spectral_cube import SpectralCube
import pyspeckit


def main():
    spec_crop = False
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
            spec_crop = True
        elif o == "-e":
            crop_end = a
            spec_crop = True
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

    """
    Read the data, remove the extra axis if necessary.
    """
    d,h = fits.getdata(input_file,header=True)
    spec_type = h['CTYPE3']
    if len(d.shape) > 3:
        d = np.squeeze(d)
        h = strip_header(h,4)
        fits.writeto(input_file,d,h,overwrite=True)

    """
    Crop the cube along its spectral axis. Fix the
    header and write to temporary file.
    """
    if spec_crop:
        """
        If crop start and crop end are given in terms of frequency or velocity, 
        translate these values to channels
        """
        if vel_crop:
            """
            If header does not contain velocity info, translate from 
            velocity to frequency before converting to channels.
            """
            if ("FREQ" or "freq") in h['CTYPE3']:
                new_crop_start = freq_to_chan(vel_to_freq(float(crop_start),h),h)
                new_crop_end = freq_to_chan(vel_to_freq(float(crop_end),h),h)
            elif ("VEL" or "vel") in h['CTYPE3']:
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
            if ("FREQ" or "freq") in h['CTYPE3']:
                new_crop_start = freq_to_chan(float(crop_start),h)
                new_crop_end = freq_to_chan(float(crop_end),h)
            elif ("VEL" or "vel") in h['CTYPE3']:
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

        #Crop the cube spectrally and write to a temporary file
        print(new_crop_start,new_crop_end)
        d_crop = d[int(new_crop_start):int(new_crop_end),:,:]
        h_crop = fix_header(int(new_crop_start),int(new_crop_end),h)
        temp_file = input_file[:-5]+"_temp.fits"
        fits.writeto(temp_file,d_crop,h_crop,overwrite=True)
        d = d_crop
        h = h_crop    

    """
    Determine the new y center value for the cube, as well
    as the new height and width. The x center value will 
    be the same, since the finite values in the partial cube 
    should be symmetric about the center.
    """
    mid_chan = int(d.shape[0]/2)
    finite_locs = np.where(np.isfinite(d[mid_chan,:,:]))
    xmin = min(finite_locs[1])
    xmax = max(finite_locs[1])
    ymin = min(finite_locs[0])
    ymax = max(finite_locs[0])
    xcen = int((xmax-xmin)/2+xmin)
    ycen = int((ymax-ymin)/2+ymin)
    
    """
    Loop through new xmin, ymin, and ymax values to determine
    which combination results in the largest area that does 
    not contain a nan valued channel. 
    """
    area = []
    ii = []
    jj = []
    kk = []
    for i in range(xmin,xcen+1):
        for j in range(ymin,ycen+1):
            for k in range(ycen,ymax+1):
                crop = d[mid_chan,j:k,xcen-i:xcen+i+1]
                if np.isfinite(crop.mean()):
                    area.append((2*i)*(k-j))
                    ii.append(i)
                    jj.append(j)
                    kk.append(k)
    max_area_loc = np.argmax(area)

    x_halfwidth = ii[max_area_loc]
    new_xmin = xcen - x_halfwidth
    new_xmax = xcen + x_halfwidth
    y_halfwidth = int((kk[max_area_loc]-jj[max_area_loc])/2.)
    new_ymin = jj[max_area_loc]
    new_ymax = kk[max_area_loc]
    new_ycen = new_ymin + y_halfwidth

    """
    Use Pyspeckit SpectralCube class to crop the cube spatially 
    and write it to the output file.
    """
    if spec_crop:
        cube = SpectralCube.read(temp_file)
        os.system("rm "+temp_file)
    else:
        cube = SpectralCube.read(input_file)
    new_cube = pyspeckit.cubes.subcube(cube,xcen,x_halfwidth,
                                       new_ycen,y_halfwidth)
    new_cube.write(output_file,format="fits",overwrite=True)

    """
    Fix the header info that Pyspeckit changed.
    Also, check that all of the nans were removed.
    """
    d,h = fits.getdata(output_file,header=True)
    if np.isnan(np.sum(d)):
        print("Not all nans removed from " + input_file)
    h['CTYPE3'] = spec_type
    fits.writeto(output_file,d,h,overwrite=True)



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

