"""
ammonia_clumpfind.py

Search NH3(1,1) or NH3(2,2) data cube for clumps and create
labeled data sets that denote and separate emission from 
different clumps. This method uses the NH3 satellite line
pattern to separate overlapping lines.

Example:
python ammonia_clumpfind.py -i cube_file
                              -r rms_file
                              -o outfilebase
                              -n 16
                              -t '11'
                              -s 100
                              
-i : Input       -- Input file
-r : rms         -- rms noise file
-o : Output      -- Output file 
-n : Numcores    -- Number of cores available for parallized computing
-t : Transition  -- The NH3 transition (11 for NH3(1,1) or 22 for NH3(2,2))
-t : Min Size    -- Minimum number of voxels for clump in output label cube
-h : Help        -- Display this help
"""

import sys,os,getopt
import copy
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import multiprocessing as mp
import astropy.io.fits as fits
import itertools
import math
from astropy.wcs import WCS
from skimage import measure
from skimage.morphology import remove_small_objects
import pdb

def main():
    #Defaults
    numcores =1
    trans = '11'
    min_size = 100
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:r:o:n:t:s:h")
    except getopt.GetoptError as err:
        print(err.msg())
        print(__doc__)
        sys.exit(2)
    for o,a in opts:
        if o == "-i":
            cube_file = a
        elif o == "-r":
            rms_file = a
        elif o == "-o":
            output_filebase = a
        elif o == "-n":
            numcores = int(a)
        elif o == "-t":
            trans = a
        elif o == "-s":
            min_size = int(a)
        elif o == "-h":
            print(__doc__)
            sys.exit(1)
        else:
            assert False, "unhandled option"
            print(__doc__)
            sys.exit(2)

    #Read in the data and create a signal-to-noise ratio cube
    cube,hcube = fits.getdata(cube_file,header=True)
    rms_map = fits.getdata(rms_file)
    rms_cube = np.repeat(rms_map[np.newaxis,:,:],cube.shape[0],axis=0)
    snr_cube = cube/rms_cube
    
    """
    Check that numcores does not exceed the number 
    of cores available
    """
    avail_cores = mp.cpu_count()
    if numcores > avail_cores:
        print("numcores variable exceeds the available number of cores.")
        print("Setting numcores equal to "+str(avail_cores))
        numcores = avail_cores
    
    """
    Split the data and locate the channels associated with 
    significant NH3 main line emission.
    """
    if numcores > 1:
        s = np.array_split(snr_cube, numcores, 2)
        procs = []
        for num in range(len(s)):
            procs.append(mp.Process(target=do_chunk,
                                    args=(num,s[num],hcube,
                                    output_filebase,trans)))
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
    else:
        do_chunk(0,snr_cube,hcube,output_filebase,trans)
    
    """
    Recombine the data to create a mask cube, where channels 
    associated with significant NH3 main line emission are labeled
    True and all other channels are labeled False.
    """
    mask_cube = recombine(numcores,output_filebase).astype(bool)
    #Remove temporary files
    for n in np.arange(numcores):
        os.system('rm '+output_filebase+'_temp'+str(n)+'.fits')

    """
    Clumps show up as three dimensional connected regions in the mask
    cube, but some of these sources are due to noise. Remove sources
    with fewer voxels than min_size in their connected regions, since
    these sources are more likely to arise from noise.
    """
    remove_small_objects(mask_cube,min_size=min_size,
                         connectivity=3,in_place=True)
    
    """
    Label the remaining three dimensional connected regions to identify
    individual clumps. Clumps are labeled with unique integer numbers,
    starting with 1. Channels not associated with a clump are set to 0.
    """
    clump_labels_3D = label_clumps(mask_cube.astype(float))
    
    """
    Collapse the clump_labels_3D array along the spectral axis for each
    clump. This shows the plane-of-sky extent of each clump. The third 
    axis of the clump_labels_2D array corresponds to individual clumps. 
    """
    clump_labels_2D = collapse_clump_labels(clump_labels_3D)

    """
    If the data cube is NH3(1,1), rearrange the labels from lower
    Galactic longitude to higher Galactic longitude.
    """
    if trans == '11':
        mask_cube,\
        clump_labels_3D,\
        clump_labels_2D = remove_small_clumps(mask_cube.astype(float),
                                              clump_labels_3D,
                                              clump_labels_2D,hcube)
        lcs,bcs = get_clump_positions(clump_labels_2D,
                                      clump_labels_3D,
                                      cube,hcube)
        clump_labels_3D = order_by_l(clump_labels_3D,lcs,bcs,'3D')
        clump_labels_2D = order_by_l(clump_labels_2D,lcs,bcs,'2D')
    #Change the data type of the label arrays
    clump_labels_3D = clump_labels_3D.astype(np.int16)
    clump_labels_2D = clump_labels_2D.astype(np.int16)
    #Write the label cubes
    fits.writeto(output_filebase+'_clump_labels_3D.fits',
                 clump_labels_3D,
                 edit_header(hcube,clump_labels_3D),
                 overwrite=True)
    fits.writeto(output_filebase+'_clump_labels_2D.fits',
                 clump_labels_2D,
                 edit_header(hcube,clump_labels_2D),
                 overwrite=True)

def do_chunk(num,data,h,output_filebase,trans):
    """
    Create a mask for this chunk of the data cube.
    The mask_ammonia_main_line function identifies the main line
    within the NH3(1,1) or (2,2) satellite line structure by 
    searching for significant emission at the velocity offsets 
    associated with the satellite lines. 
    """
    print(num)
    masked_chunk = np.full(data.shape,0)
    
    """
    In some cases, the satellite line emission is fainter than
    the noise in the spectrum. I started by searching for significant
    emission in the main line and all four satellite lines. If not 
    found, I search for emission in fewer satellite lines and at
    lower significance.
    """
    if trans == '11':
        nlines_snr_combos = [[5,5],[5,3],[3,5],[3,3],[1,5]]
    elif trans == '22':
        nlines_snr_combos = [[5,5],[5,3],[1,5]]
    #Loop through the spectra
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if np.nanmax(data[:,i,j] > 3):
                #Loop through nlines and signal-to-noise combinations
                for n,s in nlines_snr_combos:
                    masked_chunk[:,i,j] = mask_ammonia_main_line(data[:,i,j],
                                                                 h,
                                                                 snr=s,
                                                                 nlines=n,
                                                                 trans=trans)
                    #Break out of loop if main line emission is found
                    if masked_chunk[:,i,j].max() > 0:
                        break

    fits.writeto(output_filebase+"_temp"+str(num)+".fits",
                 masked_chunk,overwrite=True)


def recombine(numparts,output_filebase):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    for n in range(numparts):
        d = fits.getdata(output_filebase+"_temp"+str(n)+".fits")
        indata.append(d)
    final = np.dstack(indata)
    return(final)

def mask_ammonia_main_line(spec,h,window_width=5,snr=3,nlines=5,trans='11'):
    """
    Mask the main line emission in a NH3(1,1) or (2,2) spectrum by
    searching for emission that matches the respective satellite
    line pattern. A rolling sum is performed on the spectrum to
    improve the signal-to-noise.

    Parameters
    ----------
    spec : ndarray
        Input spectrum to search for emission.
    h : str
        FITS header of the cube.
    window_width : int, optional
        The width of the window used for the rolling sum. The default is 5.
    snr : float, optional
        The signal-to-noise cutoff for a detection. The default is 3.
    nlines : int, optional
        The number of significant lines required for a main line detection. 
        The default is 5.
    trans : str, optional
        The NH3 transition. Aclump_chansepted values are '11' or '22'. 
        The default is '11'.

    Returns
    -------
    final_mask : ndarray
        An integer array, where channels set to 1 are associated with
        main line emission and channels set to 0 are not. 

    """
    
    #Create LSR velocity axis from the header information
    vax = get_vax(h)

    #Create rolling sum and calculate the SNR of the summed channels
    roll_arr = rolling_window(spec,window_width)
    roll_arr_masked = mask_snr_spec(roll_arr,snr=1)
    va = rolling_window(vax,window_width)
    sums = ma.sum(roll_arr_masked,-1)
    sum_errs = np.sqrt(ma.count(roll_arr_masked,-1))
    sum_snrs = sums/sum_errs

    #Define the velocity offsets of the satellite lines
    if trans == '11':
        vsats = np.array([-19.421,-7.771,7.760,19.408])
    elif trans == '22':
        vsats = np.array([-25.859,-16.623,16.622,25.858])
    else:
        print("Transition not recognized. Transition must be '11' or '22'.")
        sys.exit(2)

    #Convert the velocity offsets to channel offsets
    csats = (vsats/(h['CDELT3']/1e3)).astype(int)
    """
    Shift the sum SNR array by the channel offsets for the 
    left outer (lo), left inner (li), right inner (ri), and 
    right outer (ro) satellite lines.
    """
    sum_lo = np.roll(sum_snrs,csats[0])
    sum_li = np.roll(sum_snrs,csats[1])
    sum_ri = np.roll(sum_snrs,csats[2])
    sum_ro = np.roll(sum_snrs,csats[3])

    #Search for significant channels that match the satellite line pattern
    if nlines==5:
        """
        The inner and outer satellite line sum_snrs 
        must be greater than snr
        """
        crit_inner = np.logical_and(sum_ri>snr,sum_li>snr)
        crit_outer = np.logical_and(sum_ro>snr,sum_lo>snr)
        """
        The main line emission should be brighter than the satellite 
        line emission, so I require the main line sum_snr to be
        greater than the inner and outer satellite line sum_snrs
        """
        crit_main = np.logical_and(np.logical_and(sum_snrs>sum_li,
                                                  sum_snrs>sum_ri),
                                   np.logical_and(sum_snrs>sum_ro,
                                                  sum_snrs>sum_lo))
        """
        Channels associated with a clump's main line meet the
        SNR criteria for the main line and inner and outer 
        satellite lines
        """
        clump_chans = np.where(np.logical_and(crit_main,
                                              np.logical_and(crit_inner,
                                                             crit_outer)))
    elif nlines==3:
        """
        The inner satellite line sum_snrs must be greater than snr
        """
        crit_inner = np.logical_and(sum_ri>snr,sum_li>snr)
        """
        The main line emission should be brighter than the
        satellite line emission, so the main line sum_snr must 
        be greater than the inner satellite line sum_snrs
        """
        crit_main = np.logical_and(sum_snrs>sum_li,sum_snrs>sum_ri)
        """
        Channels associated with a clump's main line meet the
        SNR criteria for the main line and inner satellite lines
        """
        clump_chans = np.where(np.logical_and(crit_main,crit_inner))    
    elif nlines==1:
        """
        For nlines=1, no satellite lines are searched for and 
        any line emission present must only have sum_snr>snr
        """
        clump_chans = np.where(sum_snrs>snr)

    """
    The rolling sum is unpadded, so the clump channels must be
    shifted by half of the rolling window width to correctly
    correspond to the spectral channels
    """
    final_mask = np.zeros_like(spec)
    final_mask[clump_chans[0]+int(np.ceil((window_width-1)/2))] = 1
    return(final_mask)

def label_clumps(clump_mask):
    """
    Label clumps that are separated in position and velocity
    space with unique integers and return the label cube.
    """
    clump_labels_3D = measure.label(clump_mask,connectivity=3)
    return(clump_labels_3D)

def collapse_clump_labels(clump_labels_3D):
    """
    Collapse the clump_labels_3D array along the spectral axis for each
    clump. This shows the plane-of-sky extent of each clump. The third 
    axis of the clump_labels_2D array corresponds to individual clumps. 
    """
    clump_labels_2D = np.zeros((clump_labels_3D.max(),
                                clump_labels_3D.shape[1],
                                clump_labels_3D.shape[2]))
    for i in np.arange(clump_labels_3D.max())+1:
        clump_chans = np.where(clump_labels_3D==i)
        clump_labels_2D[i-1,clump_chans[1],clump_chans[2]] = i
    return(clump_labels_2D)

def remove_small_clumps(mask,label3D,label2D,h):
    """
    Unresolved clumps (much smaller than the telescope beam angular 
    size) will have angular extents equal to the beam extent. If
    a detected source is smaller than this, there is a good chance
    that the source is due to noise. Remove these potentially 
    spurious detections.
    """
    bm_area = math.pi*(h['BMAJ']/2.)**2
    px_area = h['CDELT1']**2
    px_per_bm = bm_area/px_area
    for i in np.arange(label3D.max())+1:
        clump_pixs = np.where(label2D == i)
        clump_voxs = np.where(label3D == i)
        if len(clump_pixs[0]) < px_per_bm:
            mask[clump_voxs] = 0.
            label3D[clump_voxs] = 0.
            label2D[clump_pixs] = 0.

    #After removing spurious sources, reestablish clump index
    max_map = np.amax(label2D,axis=(1,2))
    old_clump_chans = np.where(max_map>0)[0]
    new_label3D = np.zeros(label3D.shape)
    new_label2D = np.zeros((len(old_clump_chans),
                            label2D.shape[1],
                            label2D.shape[2]))
    for j,c in enumerate(old_clump_chans):
        new_label2D[j,:,:] = label2D[c,:,:]
        oldnum = c+1
        newnum = j+1
        new_clump_pixs = np.where(new_label2D==oldnum)
        new_label2D[new_clump_pixs] = newnum
        new_clump_voxs = np.where(label3D==oldnum)
        new_label3D[new_clump_voxs] = newnum
        
    return(mask,new_label3D,new_label2D)

def get_clump_positions(labels2D,labels3D,cube,h):
    """
    Get the main line intensity-weighted position of each clump.
    Use the labels3D cube to mask the data cube for each clump
    and use the masked data to calculate the intensity weighted 
    Galactic positions.
    """
    #Galactic longitude (l) and Galactic latitude (b) meshgrids
    LL,BB = get_coord_grids(h)
    wls,wbs = [],[]
    for clump_idx in np.unique(labels3D)[1:]:
        print(clump_idx)
        """
        Get the boundary pixels and crop the labels, cube, and
        Galactic position arrays to speed the calculation up.
        """
        xmin,xmax,ymin,ymax = get_clump_corners(clump_idx,labels2D)
        labels3D_crop = labels3D[:,ymin:ymax,xmin:xmax]
        cube_crop = cube[:,ymin:ymax,xmin:xmax]
        LL_crop,BB_crop = LL[ymin:ymax,xmin:xmax],BB[ymin:ymax,xmin:xmax]
        clump_voxs = np.where(labels3D_crop==clump_idx)
        mask = np.zeros_like(cube_crop)
        mask[clump_voxs] = 1.
        #Make integrated intensity map using masked cube
        iimap = np.nansum(cube_crop*mask,axis=0)
        iipxs = np.where(np.isfinite(iimap))
        #Calculate weighted Galactic positions
        wls.append((LL_crop[iipxs]*iimap[iipxs]).sum()/iimap[iipxs].sum())
        wbs.append((BB_crop[iipxs]*iimap[iipxs]).sum()/iimap[iipxs].sum())
    return(np.asarray(wls),np.asarray(wbs))    
    
def order_by_l(clumps,ls,bs,labeltype):
    """
    Reorder the clump label indices by Galactic longitude (l).
    Return the reordered label array.
    """
    new_ls = np.sort(np.asarray(ls))
    new_ls_idx = np.argsort(ls)+1
    new_clumps = np.zeros(clumps.shape)
    if labeltype=='3D':
        for i,idx in enumerate(new_ls_idx):
            clump_voxs = np.where(clumps==idx)
            new_clumps[clump_voxs] = float(i+1)
    elif labeltype=='2D':
        for i,idx in enumerate(new_ls_idx):
            clump_pixs = np.where(clumps[idx-1,:,:]==idx)
            new_clumps[i,clump_pixs[0],clump_pixs[1]] = float(i+1)
    else:
        print("Must provide valid labeltype: 2D or 3D")
        sys.exit(2)
    return(new_clumps)

def get_clump_corners(num,label2D):
    """
    Return the minimum and maximum x and y pixels associated
    with a clump to facilitate cropping.
    """
    clump_chans = np.where(label2D == num)[0][0]
    clump_pixs = np.where(label2D[clump_chans,:,:] == num)
    xmin,xmax = clump_pixs[1].min(),clump_pixs[1].max()
    ymin,ymax = clump_pixs[0].min(),clump_pixs[0].max()
    return(xmin,xmax,ymin,ymax)

def mask_snr_spec(spec,snr=1):
    #Mask the SNR array where less than snr
    masked = ma.masked_where(np.logical_or(spec<snr,np.isnan(spec)),spec)
    return(masked)

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

def edit_header(h,d):
    """
    Edit the 3rd axis info in the FITS header
    """
    hout = copy.copy(h)
    if d.shape[0] != h['NAXIS3']:
        hout['NAXIS3'] = d.shape[0]
        hout['CRVAL3'] = 0.
        hout['CRPIX3'] = 0.
        hout['CDELT3'] = 1.
    hout['DATAMAX'] = d.max()
    hout['DATAMIN'] = 0.
    #Change the header array type to int16
    hout['BITPIX'] = 16
    return(hout)

def get_coord_grids(h):
    """
    Use astropy to create meshgrid arrays of the 
    Galactic longitude and latitude.
    """
    XX,YY = np.meshgrid(np.arange(h['NAXIS1']),np.arange(h['NAXIS2']))
    if h['NAXIS'] == 3 or h['WCSAXES'] == 3 :
        htemp = strip_header(copy.copy(h),3)
    w = WCS(htemp)
    LL,BB = w.all_pix2world(XX,YY,0)
    return(LL,BB)

def px2coord(x,y,h):
    """
    Use astropy to convert pixels to Galactic coordinates.
    """
    w = WCS(h)
    l,b = w.all_pix2world(x,y,0)
    return(l,b)

def coord2px(l,b,h):
    """
    Use astropy to convert Galactic coordinates to pixels.
    """
    w = WCS(h)
    x,y = w.all_world2pix(l,b,0)
    try:
        return(np.around(x).astype(int),np.around(y).astype(int))
    except:
        return(int(round(x)),int(round(y)))

def chan_to_vel(channel,h):
    """
    Use header info to convert channels to velocity.
    """
    velocity = ((channel - (h['CRPIX3']-1))*h['CDELT3'] + h['CRVAL3'])/1000.
    return(velocity)

def get_vax(h):
    """
    Return the data cube's velocity axis using the header information.
    """
    vax = np.linspace(chan_to_vel(0,h),
                      chan_to_vel(h['NAXIS3']-1,h),
                      h['NAXIS3'])
    return(vax)

if __name__ == '__main__':
    main()

