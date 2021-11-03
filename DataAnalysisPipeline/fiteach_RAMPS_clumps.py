try:
    import astropy.io.fits as pyfits
except:
    import pyfits
import sys,os,getopt
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import multiprocessing, logging
from scipy.optimize import minimize
import pyspeckit
from spectral_cube import SpectralCube
from pyspeckit.spectrum.models import ammonia_constants, ammonia, ammonia_hf
from pyspeckit.spectrum.models.ammonia_constants import freq_dict
from pyspeckit.spectrum.units import SpectroscopicAxis, SpectroscopicAxes
from pyspeckit.spectrum.models.ammonia_constants import (line_names, freq_dict, aval_dict, ortho_dict, voff_lines_dict, tau_wts_dict, ckms, ccms, h, kb, Jortho, Jpara, Brot, Crot)
from pyspeckit.spectrum.models.ammonia_constants import line_name_indices, line_names as original_line_names
from skimage.morphology import remove_small_objects,closing,disk,opening
from pyspeckit.spectrum.models import ammonia
import astropy.constants as con
import astropy.units as u
import math
import random
import pdb
from uncertainties import ufloat
from uncertainties.umath import *
import uncertainties.unumpy as unp

def main():
    #Default is to use 1 processor
    numcores = 1
    try:
        opts,args = getopt.getopt(sys.argv[1:],"f:o:n:h")
    except getopt.GetoptError,err:
        print(str(err))
        print(__doc__)
        sys.exit(2)
    for o,a in opts:
        if o == "-f":
            filebase = a
        elif o == "-o":
            outputbase = a
        elif o == "-n":
            numcores = int(a)
        elif o == "-h":
            print(__doc__)
            sys.exit(1)
        else:
            assert False, "unhandled option"
            print(__doc__)
            sys.exit(2)
    try:
        print outputbase
    except:
        outputbase = filebase
    try:
        cube_11_file = filebase+'_NH3_1-1_fixed_c.fits'
        cube_22_file = filebase+'_NH3_2-2_fixed_c.fits'
        d11,h11 = pyfits.getdata(cube_11_file,header=True)
        d22,h22 = pyfits.getdata(cube_22_file,header=True)
    except:
        cube_11_file = filebase+'_NH3_1-1_fixed.fits'
        cube_22_file = filebase+'_NH3_2-2_fixed.fits'
        d11,h11 = pyfits.getdata(cube_11_file,header=True)
        d22,h22 = pyfits.getdata(cube_22_file,header=True)

    label3D_11_file = filebase+'_NH3_1-1_clump_labels_3D.fits'
    label3D_22_file = filebase+'_NH3_2-2_clump_labels_3D.fits'
    noise_11_file = filebase+'_NH3_1-1_rms.fits'
    noise_22_file = filebase+'_NH3_2-2_rms.fits'
    vel_11_file = filebase+'_NH3_1-1_hf_vel.fits'
    width_11_file = filebase+'_NH3_1-1_hf_width.fits'
    
    label3D_11 = pyfits.getdata(label3D_11_file)
    planemask22 = np.max(pyfits.getdata(label3D_22_file),axis=0)

    errmap11 = pyfits.getdata(noise_11_file)
    errmap22 = pyfits.getdata(noise_22_file)
    vel_11 = pyfits.getdata(vel_11_file)
    width_11 = pyfits.getdata(width_11_file)
    errcube11 = np.repeat(errmap11[np.newaxis,:,:],d11.shape[0], axis=0)
    errcube22 = np.repeat(errmap22[np.newaxis,:,:],d22.shape[0], axis=0)
    errcube = np.concatenate((errcube11,errcube22))

    #pdb.set_trace()
    snr = d11/errcube11
    peaksnr = np.max(snr,axis=0)
    
    #wc = np.where(planemask_22 == clumpnum)
    
    planemask = np.logical_and(planemask22 > 0.,width_11[0,:,:] > 0.)
    vmin = max(c2v(0,h11),c2v(0,h22))
    vmax = min(c2v(h11['NAXIS3']-1,h11),c2v(h22['NAXIS3']-1,h22))
    ww = np.where(np.logical_or(vel_11<vmin,vel_11>vmax))
    vel_11[ww] = np.nan

    mmom0_11 = np.sum(d11*label3D_11,axis=0)
    peakloc = np.nanargmax(mmom0_11)
    ymax,xmax = np.unravel_index(peakloc,mmom0_11.shape)

    cube11 = pyspeckit.Cube(cube_11_file,maskmap=planemask)
    cube11.unit="K"
    cube11.xarr.velocity_convention = 'radio'
    cube11.xarr.refX = pyspeckit.spectrum.models.ammonia.freq_dict['oneone']
    cube11.xarr.refX_unit = 'Hz'
    cube11.xarr = cube11.xarr.as_unit('GHz')
    cube22 = pyspeckit.Cube(cube_22_file,maskmap=planemask)
    cube22.unit="K"
    cube22.xarr.velocity_convention = 'radio'
    cube22.xarr.refX = pyspeckit.spectrum.models.ammonia.freq_dict['twotwo']
    cube22.xarr.refX_unit = 'Hz'
    cube22.xarr = cube22.xarr.as_unit('GHz')

    cubes1 = pyspeckit.CubeStack([cube11,cube22],maskmap=planemask)
    cubes1.unit="K"

    cube1 = fit_single_comp(cubes1,vel_11,width_11,vmin,vmax,xmax,ymax,
                            peaksnr,errcube,numcores)
    modelcube1 = cube1.get_modelcube()
    

    planemask = np.logical_and(planemask22 > 0.,width_11[1,:,:] > 0.)
    cubes2 = pyspeckit.CubeStack([cube11,cube22],maskmap=planemask)
    cubes2.unit="K"
    """
    cube2 = fit_two_comp(cubes2,vel_11,width_11,vmin,vmax,xmax,ymax,
                         peaksnr,errcube,numcores)
    modelcube2 = cube2.get_modelcube()
    #merge_cubes(cube1,cube2,modelcube1,modelcube2,h11,h22,errmap11,errmap22,outputbase,vmin_clump,vmax_clump)
    """
    #"""
    #try:
    cube2 = fit_two_comp(cubes2,vel_11,width_11,vmin,vmax,xmax,ymax,
                         peaksnr,errcube,numcores)
    modelcube2 = cube2.get_modelcube()
    merge_cubes(cube1,cube2,modelcube1,modelcube2,h11,h22,errcube,outputbase,label3D_11)
    #except:
    #output_results(cube1,modelcube1,h11,h22,errcube,outputbase,label3D_11)
    #"""

    
def c2v(c,h):
    v = ((c - (h['CRPIX3']-1))*h['CDELT3'] + h['CRVAL3'])*0.001
    return v

def v2c(v,h):
    c = ((v*1000. - h['CRVAL3'])/h['CDELT3'] + (h['CRPIX3']-1))
    return int(round(c))

def get_vax(h):
    vmin = c2v(0,h)
    vmax = c2v(h['NAXIS3']-1,h)
    vax = np.linspace(vmin,vmax,h['NAXIS3'])
    return vax

def get_mask_vranges(label,h):
    vrs = np.zeros((len(np.unique(label)[1:]),3))
    for i,l in enumerate(np.unique(label)[1:]):
        wcc = np.where(label==l)[0]
        vmin = c2v(wcc.min(),h)
        vmax = c2v(wcc.max(),h)
        vrs[i,:] = np.array([l,vmin,vmax])
    return(vrs)

def get_BIC(data,model,rms,pars):
    #if len(rms)>1:
    #    rms = np.mean(rms)
    loglike = -1.*(0.5*np.log(2*np.pi)+np.log(rms.mean())+get_chi2(data,model,rms)/2)
    k = len(pars)
    N = len(data)
    BIC = np.log(N)*k-2*loglike
    return(BIC)

def get_chi2(d,m,s):
    return np.nansum((d-m)**2/s**2)

def get_tkin(trot,trot_err):
    T0 = 41.18
    trot_u = unp.uarray(trot,trot_err)
    tkin_u = trot_u/(1-(trot_u/T0)*unp.log(1+1.608*unp.exp(-25.25/trot_u)))
    tkin = unp.nominal_values(tkin_u)
    tkin_err = unp.std_devs(tkin_u)
    return(tkin,tkin_err)

def get_bantot(ntot,ntot_err,ff,ff_err):
    bantot_u = unp.uarray(ntot,ntot_err)+unp.log10(unp.uarray(ff,ff_err))
    bantot = unp.nominal_values(bantot_u)
    bantot_err = unp.std_devs(bantot_u)
    return(bantot,bantot_err)

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

def edit_header(h):
    h['NAXIS3'] = 2
    h['CRPIX3'] = 0.0
    h['CDELT3'] = 1.0
    h['CUNIT3'] = 'km/s'
    h['CRVAL3'] = 0.0
    return(h)


def fit_single_comp(cubes,vel_11,width_11,vmin,vmax,xmax,ymax,
                    peaksnr,errcube,numcores):
    """
    Fit cubes using a single-component ammonia it.
    """
    F=False
    T=True
    fittype = 'ammonia'
        
    guesses = np.zeros((7,)+cubes.cube.shape[1:])
    guesses[0,:,:] = 18                # Kinetic temperature 
    guesses[1,:,:] = 18                # Excitation  Temp
    guesses[2,:,:] = 15                # log(column)
    guesses[3,:,:] = width_11[0,:,:]   # Line width               
    guesses[4,:,:] = vel_11[0,:,:]     # Line centroid              
    guesses[5,:,:] = 0.5               # F(ortho) - ortho NH3 fraction (fixed)
    guesses[6,:,:] = 0.1               # Beam filling fraction

    print('start fit')

    cubes.fiteach(fittype=fittype,  guesses=guesses,
                  integral=False, verbose_level=1, 
                  fixed=[F,F,F,F,F,T,F], signal_cut=2,
                  limitedmax=[T,T,T,T,T,T,T],
                  maxpars=[50.,50.,17.0,10.,vmax,1,1],
                  limitedmin=[T,T,T,T,T,T,T],
                  minpars=[5,2.8,12.0,0.05,vmin,0,0],
                  start_from_point=(xmax,ymax),
                  use_neighbor_as_guess=False, 
                  use_nearest_as_guess=False, 
                  position_order = 1/peaksnr,
                  errmap=errcube, multicore=numcores)
    
    return(cubes)

def fit_two_comp(cubes,vel_11,width_11,vmin,vmax,xmax,ymax,
                 peaksnr,errcube,numcores):
    """
    Fit cubes using a two-component ammonia it.
    """
    F=False
    T=True
    fittype = 'ammonia'

    guesses = np.zeros((14,)+cubes.cube.shape[1:])
    guesses[0,:,:] = 18                # Kinetic temperature 
    guesses[1,:,:] = 18                # Excitation  Temp
    guesses[2,:,:] = 15                # log(column)
    guesses[3,:,:] = width_11[0,:,:]   # Line width               
    guesses[4,:,:] = vel_11[0,:,:]     # Line centroid              
    guesses[5,:,:] = 0.5               # F(ortho) - ortho NH3 fraction (fixed)
    guesses[6,:,:] = 0.1               # Beam filling fraction
    guesses[7,:,:] = 18                # Kinetic temperature 
    guesses[8,:,:] = 18                # Excitation  Temp
    guesses[9,:,:] = 15                # log(column)
    guesses[10,:,:] = width_11[1,:,:]  # Line width                
    guesses[11,:,:] = vel_11[1,:,:]    # Line centroid              
    guesses[12,:,:] = 0.5              # F(ortho) - ortho NH3 fraction (fixed)
    guesses[13,:,:] = 0.1              # Beam filling fraction

    #pdb.set_trace()
    print('start fit')
    cubes.fiteach(fittype=fittype,  guesses=guesses,
                  integral=False, verbose_level=1, 
                  fixed=[F,F,F,F,F,T,F,F,F,F,F,F,T,F], signal_cut=1.5,
                  limitedmax=[T,T,T,T,T,T,T,T,T,T,T,T,T,T],
                  maxpars=[50.,50.,17.0,10.,vmax,1,1,50.,50.,17.0,10.,vmax,1,1],
                  limitedmin=[T,T,T,T,T,T,T,T,T,T,T,T,T,T],
                  minpars=[5,2.8,12.0,0.05,vmin,0,0,5,2.8,12.0,0.05,vmin,0,0],
                  start_from_point=(xmax,ymax),
                  use_neighbor_as_guess=False, 
                  use_nearest_as_guess=False, 
                  position_order = 1/peaksnr,
                  errmap=errcube, multicore=numcores)

    return(cubes)

def merge_cubes(cube1,cube2,modelcube1,modelcube2,h11,h22,errcube,outputbase,label):
    """
    Merge two cubes, one single-component fit and one
    two-component fit. The two-component fit takes
    precedent over the single-component fit as long as
    the BIC is smaller by 5.
    """

    trot1 = cube1.parcube[0,:,:]
    trot_err1 = cube1.errcube[0,:,:]
    ntot1 = cube1.parcube[2,:,:]
    ntot_err1 = cube1.errcube[2,:,:]
    sigma1 = cube1.parcube[3,:,:]
    sigma_err1 = cube1.errcube[3,:,:]
    vel1 = cube1.parcube[4,:,:]
    vel_err1 = cube1.errcube[4,:,:]
    fillfrac1 = cube1.parcube[6,:,:]
    fillfrac_err1 = cube1.errcube[6,:,:]
    datacube1 = cube1.cube

    trot2 = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    trot_err2 = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    ntot2 = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    ntot_err2 = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    sigma2 = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    sigma_err2 = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    vel2 = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    vel_err2 = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    fillfrac2 = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    fillfrac_err2 = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    trot2[0,:,:] = cube2.parcube[0,:,:]
    trot_err2[0,:,:] = cube2.errcube[0,:,:]
    trot2[1,:,:] = cube2.parcube[7,:,:]
    trot_err2[1,:,:] = cube2.errcube[7,:,:]
    ntot2[0,:,:] = cube2.parcube[2,:,:]
    ntot_err2[0,:,:] = cube2.errcube[2,:,:]
    ntot2[1,:,:] = cube2.parcube[9,:,:]
    ntot_err2[1,:,:] = cube2.errcube[9,:,:]
    sigma2[0,:,:] = cube2.parcube[3,:,:]
    sigma_err2[0,:,:] = cube2.errcube[3,:,:]
    sigma2[1,:,:] = cube2.parcube[10,:,:]
    sigma_err2[1,:,:] = cube2.errcube[10,:,:]
    vel2[0,:,:] = cube2.parcube[4,:,:]
    vel_err2[0,:,:] = cube2.errcube[4,:,:]
    vel2[1,:,:] = cube2.parcube[11,:,:]
    vel_err2[1,:,:] = cube2.errcube[11,:,:]
    fillfrac2[0,:,:] = cube2.parcube[6,:,:]
    fillfrac_err2[0,:,:] = cube2.errcube[6,:,:]
    fillfrac2[1,:,:] = cube2.parcube[13,:,:]
    fillfrac_err2[1,:,:] = cube2.errcube[13,:,:]
    datacube2 = cube2.cube

    vax = get_vax(h11)
    xarr22 = SpectroscopicAxis(vax[::-1]*u.km/u.s,
                               velocity_convention='radio',
                               refX=freq_dict['twotwo']).as_unit(u.GHz)
    z_max_11 = h11['NAXIS3']
    z_max_22 = h22['NAXIS3']
    mc1_11 = modelcube1[:z_max_11,:,:][::-1]
    mc1_22 = modelcube1[z_max_11:,:,:][::-1] 
    mc2_11 = modelcube2[:z_max_11,:,:][::-1] 
    mc2_22 = modelcube2[z_max_11:,:,:][::-1] 
    dc11 = datacube2[:z_max_11,:,:][::-1]
    dc22 = datacube2[z_max_11:,:,:][::-1] 
    vranges = get_mask_vranges(label,h11)
 
    #"""
    trot = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    trot_err = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    ntot = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    ntot_err = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    sigma = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    sigma_err = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    vel = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    vel_err = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    fillfrac = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    fillfrac_err = np.full(cube1.cube[0:2,:,:].shape,np.nan)
    modelcube11 = np.full(mc1_11.shape,0.)
    modelcube22 = np.full(mc1_22.shape,0.)
    rchi2 = np.full(cube1.cube[0,:,:].shape,np.nan)
    fit_reg = np.zeros(cube1.cube[0:2,:,:].shape)
    #pdb.set_trace()
    for i in np.arange(cube1.cube.shape[1]):
        for j in np.arange(cube1.cube.shape[2]):
            #i,j = 268,22
            if modelcube1[:,i,j].max()>0 or modelcube2[:,i,j].max()>0:
                res1 = datacube1[:,i,j] - modelcube1[:,i,j]
                res2 = datacube2[:,i,j] - modelcube2[:,i,j]
                chi2_1 = np.nansum((res1**2)/(errcube[:,i,j]**2))
                dof_1 = len(res1)-5
                red_chi2_1 = chi2_1/dof_1
                chi2_2 = np.nansum((res2**2)/(errcube[:,i,j]**2))
                dof_2 = len(res2)-10
                red_chi2_2 = chi2_2/dof_2
                #pdb.set_trace()
             
                pars1 = np.array([cube1.parcube[0,i,j],cube1.parcube[2,i,j],cube1.parcube[3,i,j],cube1.parcube[4,i,j],cube1.parcube[6,i,j]])
                perrs1 = np.array([cube1.errcube[0,i,j],cube1.errcube[2,i,j],cube1.errcube[3,i,j],cube1.errcube[4,i,j],cube1.errcube[6,i,j]])
                pars2 = np.array([cube2.parcube[0,i,j],cube2.parcube[2,i,j],cube2.parcube[3,i,j],cube2.parcube[4,i,j],cube2.parcube[6,i,j],cube2.parcube[7,i,j],cube2.parcube[9,i,j],cube2.parcube[10,i,j],cube2.parcube[11,i,j],cube2.parcube[13,i,j]])
                perrs2 = np.array([cube2.errcube[0,i,j],cube2.errcube[2,i,j],cube2.errcube[3,i,j],cube2.errcube[4,i,j],cube2.errcube[6,i,j],cube2.errcube[7,i,j],cube2.errcube[9,i,j],cube2.errcube[10,i,j],cube2.errcube[11,i,j],cube2.errcube[13,i,j]])

                if np.nansum(pars1)>0:
                    model22_1_1 = (ammonia.ammonia(xarr22,trot=pars1[0],ntot=pars1[1],fortho=0.5,xoff_v=0.0,width=pars1[2],fillingfraction=pars1[4]))
                else:
                    model22_1_1 = np.zeros_like(vax)
                if np.nansum(pars2)>0.:
                    model22_2_1 = (ammonia.ammonia(xarr22,trot=pars2[0],ntot=pars2[1],fortho=0.5,xoff_v=0.0,width=pars2[2],fillingfraction=pars2[4]))
                    model22_2_2 = (ammonia.ammonia(xarr22,trot=pars2[5],ntot=pars2[6],fortho=0.5,xoff_v=0.0,width=pars2[7],fillingfraction=pars2[9]))
                else:
                    model22_2_1 = np.zeros_like(vax)
                    model22_2_2 = np.zeros_like(vax)

                snr22_1_1 = model22_1_1.max()/errcube[-1,i,j]
                snr22_2_1 = model22_2_1.max()/errcube[-1,i,j]
                snr22_2_2 = model22_2_2.max()/errcube[-1,i,j]

                nsig = 2

                if snr22_1_1>nsig:
                    BIC_1 = get_BIC(datacube1[:,i,j],modelcube1[:,i,j],errcube[:,i,j],pars1)
                else:
                    BIC_1 = get_BIC(datacube1[:,i,j],np.zeros_like(datacube1[:,i,j]),errcube[:,i,j],pars1)
                if snr22_2_1>nsig and snr22_2_2>nsig:
                    BIC_2 = get_BIC(datacube2[:,i,j],modelcube2[:,i,j],errcube[:,i,j],pars2)
                else:
                    BIC_2 = get_BIC(datacube2[:,i,j],np.zeros_like(datacube2[:,i,j]),errcube[:,i,j],pars2)

                try:
                    print(snr22_1_1>nsig,snr22_2_1>nsig,snr22_2_2>nsig)
                except:
                    pdb.set_trace()

                lcomps = np.unique(label[:,i,j])[1:]
                vfits = [pars1[3],pars2[3],pars2[8]]
                crit = np.full((len(lcomps),len(vfits)),False)
                for ii,c in enumerate(lcomps):
                    for jj,v in enumerate(vfits):
                        wvrs = np.where(vranges[:,0]==c)[0][0]
                        vrmin,vrmax = vranges[wvrs,1],vranges[wvrs,2]
                        #pdb.set_trace()
                        if v>=vrmin and v<=vrmax:
                            crit[ii,jj] = True
                        else:
                            crit[ii,jj] = False
                #plt.plot(datacube2[:,i,j],color='k')
                #plt.plot(modelcube2[:,i,j],color='r')
                #plt.show()
                #pdb.set_trace()
                if (snr22_2_1>nsig and snr22_2_2>nsig) and (BIC_2<(BIC_1-5) and crit[:,1].any() and crit[:,2].any()):
                    c1 = np.where(crit[:,1])[0]
                    if len(c1)>1:
                        vdiff = []
                        for cs in c1:
                            wwcs = np.where(label[:,i,j]==lcomps[cs])[0]
                            vvcs = np.sum(vax[wwcs]*dc11[wwcs,i,j])/np.sum(dc11[wwcs,i,j])
                            vdiff.append(abs(vfits[1]-vvcs))
                        bc1 = np.where(vdiff == min(vdiff))
                        fit_reg[0,i,j] = lcomps[bc1]
                    else:
                        fit_reg[0,i,j] = lcomps[c1]
                    c2 = np.where(crit[:,2])[0]
                    if len(c2)>1:
                        vdiff = []
                        for cs in c2:
                            wwcs = np.where(label[:,i,j]==lcomps[cs])[0]
                            vvcs = np.sum(vax[wwcs]*dc11[wwcs,i,j])/np.sum(dc11[wwcs,i,j])
                            vdiff.append(abs(vfits[2]-vvcs))
                            bc2 = np.where(vdiff == min(vdiff))
                            fit_reg[1,i,j] = lcomps[bc2]
                    else:
                        fit_reg[1,i,j] = lcomps[c2]
                    fit11 = mc2_11[:,i,j]
                    fit22 = mc2_22[:,i,j]
                    pars = pars2
                    perrs = perrs2
                    red_chi2 = red_chi2_2

                elif snr22_1_1>nsig and crit[:,0].any():
                    c1 = np.where(crit[:,0])[0]
                    if len(c1)>1:
                        vdiff = []
                        for cs in c1:
                            wwcs = np.where(label[:,i,j]==lcomps[cs])[0]
                            vvcs = np.sum(vax[wwcs]*dc11[wwcs,i,j])/np.sum(dc11[wwcs,i,j])
                            vdiff.append(abs(vfits[0]-vvcs))
                            bc1 = np.where(vdiff == min(vdiff))
                            fit_reg[0,i,j] = lcomps[bc1]
                    else:
                        fit_reg[0,i,j] = lcomps[c1]
                    fit11 = mc1_11[:,i,j]
                    fit22 = mc1_22[:,i,j]
                    pars = pars1
                    perrs = perrs1
                    red_chi2 = red_chi2_1

                else:
                    fit11 = np.zeros_like(dc11[:,i,j])
                    fit22 = np.zeros_like(dc22[:,i,j])
                    pars = np.full((10,),np.nan)
                    perrs = np.full((10,),np.nan)
                    red_chi2 = np.nan


                modelcube11[:,i,j] = fit11
                modelcube22[:,i,j] = fit22
                rchi2[i,j] = red_chi2

                order = nan_argsort(pars[::5])

                for kk in order:
                    trot[kk,i,j] = pars[kk*5]
                    trot_err[kk,i,j] = perrs[kk*5]
                    ntot[kk,i,j] = pars[kk*5+1]
                    ntot_err[kk,i,j] = perrs[kk*5+1]
                    sigma[kk,i,j] = pars[kk*5+2]
                    sigma_err[kk,i,j] = perrs[kk*5+2]
                    vel[kk,i,j] = pars[kk*5+3]
                    vel_err[kk,i,j] = perrs[kk*5+3]
                    fillfrac[kk,i,j] = pars[kk*5+4]
                    fillfrac_err[kk,i,j] = perrs[kk*5+4]
            #pdb.set_trace()                

    #pdb.set_trace()
    pyfits.writeto(outputbase+'_model11.fits',modelcube11,h11,overwrite=True)
    pyfits.writeto(outputbase+'_model22.fits',modelcube22,h22,overwrite=True)
    pyfits.writeto(outputbase+"_fit_reg.fits",fit_reg,edit_header(h11),overwrite=True)
    nan_locations = np.where(trot == 0.)
    trot[nan_locations] = np.nan
    trot_err[nan_locations] = np.nan
    ntot[nan_locations] = np.nan
    ntot_err[nan_locations] = np.nan
    sigma[nan_locations] = np.nan
    sigma_err[nan_locations] = np.nan
    vel[nan_locations] = np.nan
    vel_err[nan_locations] = np.nan
    fillfrac[nan_locations] = np.nan
    fillfrac_err[nan_locations] = np.nan
    tkin,tkin_err = get_tkin(trot,trot_err)
    bantot,bantot_err = get_bantot(ntot,ntot_err,fillfrac,fillfrac_err)
    trotfile = pyfits.PrimaryHDU(data=trot,header=edit_header(cube2.header))
    trotfile.header['BUNIT'] = 'K'
    #trotfile.header = edit_header(trotfile.header[:])
    trotfile.writeto(outputbase+"_trot.fits",clobber=True)
    troterrfile = pyfits.PrimaryHDU(data=trot_err,header=trotfile.header)
    troterrfile.writeto(outputbase+"_trot_err.fits",clobber=True)
    tkinfile = pyfits.PrimaryHDU(data=tkin,header=trotfile.header)
    tkinfile.header['BUNIT'] = 'K'
    tkinfile.writeto(outputbase+"_tkin.fits",clobber=True)
    tkinerrfile = pyfits.PrimaryHDU(data=tkin_err,header=tkinfile.header)
    tkinerrfile.writeto(outputbase+"_tkin_err.fits",clobber=True)
    ntotfile = pyfits.PrimaryHDU(data=ntot,header=trotfile.header)
    ntotfile.header['BUNIT'] = 'log(cm^2)'
    ntotfile.writeto(outputbase+"_ntot.fits",clobber=True)
    ntoterrfile = pyfits.PrimaryHDU(data=ntot_err,header=ntotfile.header)
    ntoterrfile.writeto(outputbase+"_ntot_err.fits",clobber=True)
    bantotfile = pyfits.PrimaryHDU(data=bantot,header=trotfile.header)
    bantotfile.header['BUNIT'] = 'log(cm^2)'
    bantotfile.writeto(outputbase+"_bantot.fits",clobber=True)
    bantoterrfile = pyfits.PrimaryHDU(data=bantot_err,header=bantotfile.header)
    bantoterrfile.writeto(outputbase+"_bantot_err.fits",clobber=True)
    sigmafile = pyfits.PrimaryHDU(data=sigma,header=ntotfile.header)
    sigmafile.header['BUNIT'] = 'km/s'
    sigmafile.writeto(outputbase+"_sigma.fits",clobber=True)
    sigmaerrfile = pyfits.PrimaryHDU(data=sigma_err,header=sigmafile.header)
    sigmaerrfile.writeto(outputbase+"_sigma_err.fits",clobber=True)
    velfile = pyfits.PrimaryHDU(data=vel,header=sigmafile.header)
    velfile.writeto(outputbase+"_vel.fits",clobber=True)
    velerrfile = pyfits.PrimaryHDU(data=vel_err,header=velfile.header)
    velerrfile.writeto(outputbase+"_vel_err.fits",clobber=True)
    fillfracfile = pyfits.PrimaryHDU(data=fillfrac,header=velfile.header)
    fillfracfile.header['BUNIT'] = 'frac'
    fillfracfile.writeto(outputbase+"_fillfrac.fits",clobber=True)
    fillfracerrfile = pyfits.PrimaryHDU(data=fillfrac_err,
                                        header=fillfracfile.header)
    fillfracerrfile.writeto(outputbase+"_fillfrac_err.fits",clobber=True)
    red_chi2_file = pyfits.PrimaryHDU(data=rchi2,header=cube2.header)
    red_chi2_file.header['BUNIT'] = ''
    red_chi2_file.header = strip_header(red_chi2_file.header[:],3)
    red_chi2_file.writeto(outputbase+"_red_chi2.fits",clobber=True)

    #"""

def output_results(cube,modelcube,h11,h22,errcube,outputbase,label):
    trot = np.full(cube.cube[0,:,:].shape,np.nan)
    trot_err = np.full(cube.cube[0,:,:].shape,np.nan)
    ntot = np.full(cube.cube[0,:,:].shape,np.nan)
    ntot_err = np.full(cube.cube[0,:,:].shape,np.nan)
    sigma = np.full(cube.cube[0,:,:].shape,np.nan)
    sigma_err = np.full(cube.cube[0,:,:].shape,np.nan)
    vel = np.full(cube.cube[0,:,:].shape,np.nan)
    vel_err = np.full(cube.cube[0,:,:].shape,np.nan)
    fillfrac = np.full(cube.cube[0,:,:].shape,np.nan)
    fillfrac_err = np.full(cube.cube[0,:,:].shape,np.nan)
    fit_reg = np.zeros(cube.cube[0,:,:].shape)

    vax = get_vax(h11)
    vranges = get_mask_vranges(label,h11)
    rchi2 = np.full(cube.cube[0,:,:].shape,np.nan)
    res = cube.cube - modelcube
    dof = len(res[:,0,0])-5
    for i in range(rchi2.shape[0]):
        for j in range(rchi2.shape[1]):
            chi2 = np.nansum((res[:,i,j]**2)/(errcube[:,i,j]**2))
            rchi2[i,j] = chi2/dof
            lcomps = np.unique(label[:,i,j])[1:]
            crit = np.full((len(lcomps)),False)
            for k,c in enumerate(lcomps):
                wvrs = np.where(vranges[:,0]==c)[0][0]
                vrmin,vrmax = vranges[wvrs,1],vranges[wvrs,2]
                #pdb.set_trace()
                if cube.parcube[4,i,j]>=vrmin and cube.parcube[4,i,j]<=vrmax:
                    crit[k] = True
                else:
                    crit[k] = False
            if crit.any():
                trot[i,j] = cube.parcube[0,i,j]
                trot_err[i,j] = cube.errcube[0,i,j]
                ntot[i,j] = cube.parcube[2,i,j]
                ntot_err[i,j] = cube.errcube[2,i,j]
                sigma[i,j] = cube.parcube[3,i,j]
                sigma_err[i,j] = cube.errcube[3,i,j]
                vel[i,j] = cube.parcube[4,i,j]
                vel_err[i,j] = cube.errcube[4,i,j]
                fillfrac[i,j] = cube.parcube[6,i,j]
                fillfrac_err[i,j] = cube.errcube[6,i,j]
                c1 = np.where(crit[:])[0]
                if len(c1)>1:
                    vdiff = []
                    for cs in c1:
                        wwcs = np.where(label[:,i,j]==lcomps[cs])[0]
                        vvcs = np.sum(vax[wwcs]*data[wwcs,i,j])/np.sum(data[wwcs,i,j])
                        vdiff.append(abs(vel[i,j]-vvcs))
                        bc1 = np.where(vdiff == min(vdiff))
                        fit_reg[i,j] = lcomps[bc1]
                else:
                    fit_reg[i,j] = lcomps[c1]
    
    pyfits.writeto(outputbase+"_fit_reg.fits",fit_reg,strip_header(h11,3),overwrite=True)
    nan_locations = np.where(trot == 0.)
    trot[nan_locations] = np.nan
    trot_err[nan_locations] = np.nan
    ntot[nan_locations] = np.nan
    ntot_err[nan_locations] = np.nan
    sigma[nan_locations] = np.nan
    sigma_err[nan_locations] = np.nan
    vel[nan_locations] = np.nan
    vel_err[nan_locations] = np.nan
    fillfrac[nan_locations] = np.nan
    fillfrac_err[nan_locations] = np.nan
    tkin,tkin_err = get_tkin(trot,trot_err)
    bantot,bantot_err = get_bantot(ntot,ntot_err,fillfrac,fillfrac_err)
    trotfile = pyfits.PrimaryHDU(data=trot,header=cube.header)
    trotfile.header['BUNIT'] = 'K'
    trotfile.header = strip_header(trotfile.header[:],3)
    trotfile.writeto(outputbase+"_trot.fits",clobber=True)
    troterrfile = pyfits.PrimaryHDU(data=trot_err,header=trotfile.header)
    troterrfile.writeto(outputbase+"_trot_err.fits",clobber=True)
    tkinfile = pyfits.PrimaryHDU(data=tkin,header=trotfile.header)
    tkinfile.header['BUNIT'] = 'K'
    tkinfile.writeto(outputbase+"_tkin.fits",clobber=True)
    tkinerrfile = pyfits.PrimaryHDU(data=tkin_err,header=tkinfile.header)
    tkinerrfile.writeto(outputbase+"_tkin_err.fits",clobber=True)
    ntotfile = pyfits.PrimaryHDU(data=ntot,header=tkinfile.header)
    ntotfile.header['BUNIT'] = 'log(cm^2)'
    ntotfile.writeto(outputbase+"_ntot.fits",clobber=True)
    ntoterrfile = pyfits.PrimaryHDU(data=ntot_err,header=ntotfile.header)
    ntoterrfile.writeto(outputbase+"_ntot_err.fits",clobber=True)
    bantotfile = pyfits.PrimaryHDU(data=bantot,header=trotfile.header)
    bantotfile.header['BUNIT'] = 'log(cm^2)'
    bantotfile.writeto(outputbase+"_bantot.fits",clobber=True)
    bantoterrfile = pyfits.PrimaryHDU(data=bantot_err,header=bantotfile.header)
    bantoterrfile.writeto(outputbase+"_bantot_err.fits",clobber=True)
    sigmafile = pyfits.PrimaryHDU(data=sigma,header=ntotfile.header)
    sigmafile.header['BUNIT'] = 'km/s'
    sigmafile.writeto(outputbase+"_sigma.fits",clobber=True)
    sigmaerrfile = pyfits.PrimaryHDU(data=sigma_err,header=sigmafile.header)
    sigmaerrfile.writeto(outputbase+"_sigma_err.fits",clobber=True)
    velfile = pyfits.PrimaryHDU(data=vel,header=sigmafile.header)
    velfile.writeto(outputbase+"_vel.fits",clobber=True)
    velerrfile = pyfits.PrimaryHDU(data=vel_err,header=velfile.header)
    velerrfile.writeto(outputbase+"_vel_err.fits",clobber=True)
    fillfracfile = pyfits.PrimaryHDU(data=fillfrac,header=velfile.header)
    fillfracfile.header['BUNIT'] = 'frac'
    fillfracfile.writeto(outputbase+"_fillfrac.fits",clobber=True)
    fillfracerrfile = pyfits.PrimaryHDU(data=fillfrac_err,
                                        header=fillfracfile.header)
    fillfracerrfile.writeto(outputbase+"_fillfrac_err.fits",clobber=True)

    z_max_22 = h22['NAXIS3']
    modelcube = modelcube[::-1]
    modelcube11 = modelcube[z_max_22:,:,:]
    modelcube22 = modelcube[0:z_max_22,:,:]
    pyfits.writeto(outputbase+'_model11.fits',modelcube11,h11,clobber=True)
    pyfits.writeto(outputbase+'_model22.fits',modelcube22,h22,clobber=True)
    red_chi2_file = pyfits.PrimaryHDU(data=rchi2,header=cube.header)
    red_chi2_file.header['BUNIT'] = ''
    red_chi2_file.header = strip_header(red_chi2_file.header[:],3)
    red_chi2_file.writeto(outputbase+"_red_chi2.fits",clobber=True)


def nan_argsort(a):
    temp = a.copy()
    temp[np.isnan(a)] = np.inf
    return temp.argsort()

if __name__ == '__main__':
    main()

