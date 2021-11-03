#!/usr/bin/env python
# encoding: utf-8
"""
Search NH3(1,1) or NH3(2,2) data cube for clumps and create
labeled data sets that denote and separate emission from 
different clump. This method uses the NH3 satellite line
pattern to separate overlapping lines.

Example:
python satellite_clumpfind.py -i cube_file
                              -r rms_file
                              -o outfilebase
"""

import sys,os,getopt
import copy
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import multiprocessing, logging
import astropy.io.fits as fits
import itertools
import math
from astropy.wcs import WCS
from skimage import measure
from skimage.morphology import remove_small_objects
import pdb

def main():
    numcores = 1
    trans = '11'
    min_size = 100
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:r:o:n:t:s:h")
    except getopt.GetoptError,err:
        print(str(err))
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

    cube,hcube = fits.getdata(cube_file,header=True)
    rms_map = fits.getdata(rms_file)
    rms_cube = np.repeat(rms_map[np.newaxis,:,:],cube.shape[0],axis=0)
    snr_cube = cube/rms_cube
    
    if numcores > 1:
        s = np.array_split(snr_cube, numcores, 2)
        ps = []
        for num in range(len(s)):
            ps.append(multiprocessing.Process(target=do_chunk,
                                              args=(num,s[num],hcube,
                                                    output_filebase,trans)))
        for p in ps:
            p.start()
        for p in ps:
            p.join()
    else:
        do_chunk(0,snr_cube,hcube,output_filebase,trans)
    mask_cube = recombine(numcores,output_filebase).astype(bool)
    for n in np.arange(numcores):
        os.system('rm '+output_filebase+'_temp'+str(n)+'.fits')

    remove_small_objects(mask_cube,min_size=min_size,connectivity=3,in_place=True)
    clump_labels_3D = label_clumps(mask_cube.astype(float))
    clump_labels_2D = collapse_clump_labels(clump_labels_3D)

    if trans == '11':
        mask_cube,clump_labels_3D,clump_labels_2D = remove_small_clumps(mask_cube.astype(float),clump_labels_3D,clump_labels_2D,hcube)
        lcs,bcs = get_clump_positions(clump_labels_2D,clump_labels_3D,cube,hcube)
        clump_labels_3D = order_by_l(clump_labels_3D,lcs,bcs,'3D')
        clump_labels_2D = order_by_l(clump_labels_2D,lcs,bcs,'2D')
    fits.writeto(output_filebase+'_clump_labels_3D.fits',clump_labels_3D.astype(float),edit_header(hcube,clump_labels_3D.astype(float)),overwrite=True)
    fits.writeto(output_filebase+'_clump_labels_2D.fits',clump_labels_2D.astype(float),edit_header(hcube,clump_labels_2D.astype(float)),overwrite=True)

def do_chunk(num,data,h,output_filebase,trans):
    print num
    ya = np.full(data.shape,0)
    if trans == '11':
        N_snr_combos = [[5,5],[5,3],[3,5],[3,3],[1,5]]
    elif trans == '22':
        N_snr_combos = [[5,5],[5,3],[1,5]]
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if np.nanmax(data[:,i,j] > 3):
                #print i,j
                for n,s in N_snr_combos:
                    ya[:,i,j] = satellite_roll_sum(data[:,i,j],h,snr=s,
                                                   N=n,trans=trans)
                    #print(ya[:,i,j].max() > 0)
                    if ya[:,i,j].max() > 0:
                        break

    fits.writeto(output_filebase+"_temp"+str(num)+".fits",ya,overwrite=True)


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

def satellite_roll_sum(spec,h,ww=5,snr=3,N=5,trans='11'):
    vax = get_vax(h)
    masked = np.zeros(h['NAXIS3'])
    ya = rolling_window(spec,ww)
    ya_ma = mask_snr_spec(ya,snr=1)
    va = rolling_window(vax,ww)
    sums = ma.sum(ya_ma,-1)
    sum_errs = np.sqrt(ma.count(ya_ma,-1))
    sum_snrs = sums/sum_errs

    if trans == '11':
        vsats = np.array([-19.421,-7.771,7.760,19.408])
    elif trans == '22':
        vsats = np.array([-25.859,-16.623,16.622,25.858])
    else:
        print("Transition not recognized.")

    csats = (vsats/(h['CDELT3']/1e3)).astype(int)
    sum_lo = np.roll(sum_snrs,csats[0])
    sum_li = np.roll(sum_snrs,csats[1])
    sum_ri = np.roll(sum_snrs,csats[2])
    sum_ro = np.roll(sum_snrs,csats[3])

    if N==5:
        crit_i = np.logical_and(sum_ri>snr,sum_li>snr)
        crit_o = np.logical_and(sum_ro>snr,sum_lo>snr)
        crit_m = np.logical_and(np.logical_and(sum_snrs>sum_li,sum_snrs>sum_ri),np.logical_and(sum_snrs>sum_ro,sum_snrs>sum_lo))
        cc = np.where(np.logical_and(crit_m,np.logical_and(crit_i,crit_o)))
    elif N==3:
        crit_i = np.logical_and(sum_ri>snr,sum_li>snr)
        crit_m = np.logical_and(sum_snrs>sum_li,sum_snrs>sum_ri)
        cc = np.where(np.logical_and(crit_m,crit_i))    
    elif N==1:
        cc = np.where(sum_snrs>snr)

    final_mask = np.zeros_like(spec)
    final_mask[cc[0]+2] = 1
    return(final_mask)

def label_clumps(clump_mask):
    clumps = measure.label(clump_mask,connectivity=3)
    return(clumps)

def collapse_clump_labels(clump_labels_3D):
    clump_labels_2D = np.zeros((clump_labels_3D.max(),clump_labels_3D.shape[1],clump_labels_3D.shape[2]))
    for i in np.arange(clump_labels_3D.max())+1:
        ww = np.where(clump_labels_3D==i)
        clump_labels_2D[i-1,ww[1],ww[2]] = i
    return(clump_labels_2D)

def remove_small_clumps(m,lcube,lcoll,h):
    bm_area = math.pi*(h['BMAJ']/2.)**2
    px_area = h['CDELT1']**2
    px_per_bm = bm_area/px_area
    for i in np.arange(lcube.max())+1:
        wc2D = np.where(lcoll == i)
        wc3D = np.where(lcube == i)
        if len(wc2D[0]) < px_per_bm:
            m[wc3D] = 0.
            lcube[wc3D] = 0.
            lcoll[wc2D] = 0.

    max_map = np.amax(lcoll,axis=(1,2))
    wc = np.where(max_map>0)[0]
    new_lcube = np.zeros(lcube.shape)
    new_lcoll = np.zeros((len(wc),lcoll.shape[1],lcoll.shape[2]))
    for j,w in enumerate(wc):
        new_lcoll[j,:,:] = lcoll[w,:,:]
        oldnum = w+1
        newnum = j+1
        new_wc2D = np.where(new_lcoll==oldnum)
        new_lcoll[new_wc2D] = newnum
        new_wc3D = np.where(lcube==oldnum)
        new_lcube[new_wc3D] = newnum
        
    return(m,new_lcube,new_lcoll)

def get_clump_positions(labels2D,labels3D,cube,h):
    LL,BB = get_coord_grids(h)
    lcs,bcs = [],[]
    for c in np.unique(labels3D)[1:]:
        print(c)
        xmin,xmax,ymin,ymax = get_clump_corners(c,labels2D)
        labels3D_crop = labels3D[:,ymin:ymax,xmin:xmax]
        cube_crop = cube[:,ymin:ymax,xmin:xmax]
        LL_crop,BB_crop = LL[ymin:ymax,xmin:xmax],BB[ymin:ymax,xmin:xmax]
        wc = np.where(labels3D_crop==c)
        mask = np.zeros_like(cube_crop)
        mask[wc] = 1.
        iimap = np.nansum(cube_crop*mask,axis=0)
        wp = np.where(np.isfinite(iimap))
        lcs.append(np.sum(LL_crop[wp]*iimap[wp])/np.sum(iimap[wp]))
        bcs.append(np.sum(BB_crop[wp]*iimap[wp])/np.sum(iimap[wp]))
        #pdb.set_trace()
    return(np.asarray(lcs),np.asarray(bcs))    
    
def order_by_l(clumps,ls,bs,labeltype):
    new_ls = np.sort(np.asarray(ls))
    wl = np.argsort(ls)
    new_bs = np.asarray(bs)[wl]
    new_clumps = np.zeros(clumps.shape)
    if labeltype=='3D':
        for i,c in enumerate(wl):
            #pdb.set_trace()
            ww = np.where(clumps==c+1)
            new_clumps[ww] = float(i+1)
    elif labeltype=='2D':
        for i,c in enumerate(wl):
            #pdb.set_trace()
            ww = np.where(clumps[c,:,:]==c+1)
            new_clumps[i,ww[0],ww[1]] = float(i+1)
    else:
        print("Must provide valid labeltype: 2D or 3D")
    return(new_clumps)

def get_clump_corners(num,label2D):
    cc = np.where(label2D == num)[0][0]
    wc2D = np.where(label2D[cc,:,:] == num)
    xmin,xmax = wc2D[1].min(),wc2D[1].max()
    ymin,ymax = wc2D[0].min(),wc2D[0].max()
    return(xmin,xmax,ymin,ymax)

def get_clump_sizes(clump_labels):
    clump_areas = np.zeros_like(clump_labels)
    for i in np.arange(clump_labels.max())+1:
        ww = np.where(clump_labels==i)
        clump_areas[ww] = len(ww[0])
    return(clump_areas)

def extend_clumps_3D(snr_cube,cube_mask,snr=3):
    cube_mask = cube_mask.astype(int)
    wn = np.where(get_vox_neighbors(cube_mask)==1)
    ws = np.where(snr_cube[wn]>snr)
    i=0
    while(len(ws[0])>0):
        print(i)
        i+=1
        cube_mask[wn[0][ws],wn[1][ws],wn[2][ws]] = 1
        wn = np.where(get_vox_neighbors(cube_mask)==1)
        ws = np.where(snr_cube[wn]>snr)
    return(cube_mask)

def extend_clumps_2D(snr_cube,cube_mask,snr=3):
    cube_mask = cube_mask.astype(int)
    wn = np.where(get_pix_neighbors(cube_mask)==1)
    ws = np.where(snr_cube[wn]>snr)
    i=0
    while(len(ws[0])>0):
        print(i)
        i+=1
        cube_mask[wn[0][ws],wn[1][ws],wn[2][ws]] = 1
        wn = np.where(get_pix_neighbors(cube_mask)==1)
        ws = np.where(snr_cube[wn]>snr)
    return(cube_mask)

def get_vox_neighbors(arr):
    npad = ((1,1), (1,1), (1,1))
    pad_arr = np.pad(arr, pad_width=npad, mode='constant', constant_values=0)
    arrout = np.zeros_like(pad_arr)
    for i in np.arange(3)-1:
        for j in np.arange(3)-1:
            for k in np.arange(3)-1:
                if i!=0 or j!=0 or k!=0:
                    temparr = pad_arr-np.roll(pad_arr,(i,j,k),axis=(0,1,2))
                    ww = np.where(temparr==-1)
                    arrout[ww] = 1
    return(arrout[1:-1,1:-1,1:-1])

def get_pix_neighbors(arr):
    npad = ((0,0), (1,1), (1,1))
    pad_arr = np.pad(arr, pad_width=npad, mode='constant', constant_values=0)
    arrout = np.zeros_like(pad_arr)
    for i in np.arange(3)-1:
        for j in np.arange(3)-1:
            if i!=0 or j!=0:
                temparr = pad_arr-np.roll(pad_arr,(0,i,j),axis=(0,1,2))
                ww = np.where(temparr==-1)
                arrout[ww] = 1
    return(arrout[:,1:-1,1:-1])

def mask_snr_spec(spec,snr=1):
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
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

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
    return(hout)

def get_coord_grids(h):
    XX,YY = np.meshgrid(np.arange(h['NAXIS1']),np.arange(h['NAXIS2']))
    if h['NAXIS'] == 3 or h['WCSAXES'] == 3 :
        htemp = strip_header(copy.copy(h),3)
    w = WCS(htemp)
    try:
        LL,BB = w.all_pix2world(XX,YY,0)
    except:
        pdb.set_trace()
    return LL,BB

def px2coord(x,y,h):
    w = WCS(h)
    l,b = w.all_pix2world(x,y,0)
    return(l,b)

def coord2px(l,b,h):
    w = WCS(h)
    x,y = w.all_world2pix(l,b,0)
    try:
        return(np.around(x).astype(int),np.around(y).astype(int))
    except:
        return(int(round(x)),int(round(y)))

def c2v(channel,h):
    velocity = ((channel - h['CRPIX3']-1)*h['CDELT3'] + h['CRVAL3'])/1000.
    return velocity

def v2c(velocity,h):
    channel = (velocity*1000. - h['CRVAL3'])/h['CDELT3'] + h['CRPIX3']-1
    try:
        return(channel.astype(int))
    except:
        return(int(channel))

def get_vax(h):
    vax = np.linspace(c2v(0,h),c2v(h['NAXIS3']-1,h),h['NAXIS3'])
    return(vax)

if __name__ == '__main__':
    main()

