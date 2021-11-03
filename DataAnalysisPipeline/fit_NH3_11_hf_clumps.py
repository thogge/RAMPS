#!/usr/bin/env python
# encoding: utf-8
"""
fit_NH3_11.py

Fit NH3(1,1) data cube and output the fit and fit parameters. 

This version runs in parallel, which is useful because
the process is fairly slow.

Example:
python fit_NH3_11.py 
       -i L30_Tile01-04_NH3_1-1_fixed.fits 
       -o L30_Tile01-04_NH3_1-1.fits 

-i : Input      -- Input file (reduced by pipeline)
-o : Output     -- Output file 
-n : Cores   -- Number of cores available for parallized computing
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
import math
import scipy.signal as si
import matplotlib.pyplot as plt
import multiprocessing, logging
import my_pad
import uncertainties as uu
from uncertainties import unumpy as unp
from uncertainties.umath import *
from scipy.optimize import curve_fit
from scipy import stats
from pyspeckit.spectrum.models.ammonia_constants import (line_names, freq_dict, aval_dict, ortho_dict, voff_lines_dict, tau_wts_dict, ckms, ccms, h, kb, Jortho, Jpara, Brot, Crot)
from pyspeckit.spectrum.units import SpectroscopicAxis, SpectroscopicAxes
from astropy import units as u
import pdb

def main():
    #Defaults
    output_file = "default.fits"
    numcores = 1
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:l:r:n:o:h")
    except getopt.GetoptError.err:
        print(str(err))
        print(__doc__)
        sys.exit(2)
    for o,a in opts:
        if o == "-i":
            input_file = a
        elif o == "-l":
            label_file = a
        elif o == "-r":
            rms_file = a
        elif o == "-n":
            numcores = int(a)
        elif o == "-o":
            output_filebase = a
        elif o == "-h":
            print(__doc__)
            sys.exit(1)
        else:
            assert False, "unhandled option"
            print(__doc__)
            sys.exit(2)
        

    #Read in data into array
    d,h = pyfits.getdata(input_file,header=True)
    nans = np.where(np.isnan(d))
    d[nans] = 0.
    l = pyfits.getdata(label_file)
    r = pyfits.getdata(rms_file)
    m = np.zeros(l.shape)
    wm = np.where(l>0)
    m[wm] = 1.
    main_peak = np.max(d*m,axis=0)
    wc = np.where(main_peak>0)
    XG,YG = np.meshgrid(np.arange(r.shape[1]),np.arange(r.shape[0]))
    xs,ys = XG[wc],YG[wc]
    #pdb.set_trace()
    vrs = get_mask_vranges(l,h)
    if numcores > 1:
        #Split the data
        xx = np.array_split(xs, numcores, 0)
        yy = np.array_split(ys, numcores, 0)
        ps = []
        #Fit baselines and write to temporary files
        for num in range(len(xx)):
            ps.append(multiprocessing.Process(target=do_chunk_fit,
                                              args=(num,d[:,yy[num],xx[num]],
                                                    l[:,yy[num],xx[num]],
                                                    r[yy[num],xx[num]],
                                                    vrs,h,output_filebase)))
        for p in ps:
            p.start()
        for p in ps:
            p.join()

        fitcube = recombine_txt(numcores,xx,yy,output_filebase+"_temp_fit",d.shape)
        fit_reg = recombine_txt(numcores,xx,yy,output_filebase+"_temp_fit_reg",d[:2,:,:].shape)
        vel = recombine_txt(numcores,xx,yy,output_filebase+"_temp_vel",d[:2,:,:].shape)
        vel_err = recombine_txt(numcores,xx,yy,output_filebase+"_temp_vel_err",d[:2,:,:].shape)
        width = recombine_txt(numcores,xx,yy,output_filebase+"_temp_width",d[:2,:,:].shape)
        width_err = recombine_txt(numcores,xx,yy,output_filebase+"_temp_width_err",d[:2,:,:].shape)
        tex = recombine_txt(numcores,xx,yy,output_filebase+"_temp_tex",d[:2,:,:].shape)
        tex_err = recombine_txt(numcores,xx,yy,output_filebase+"_temp_tex_err",d[:2,:,:].shape)
        tau11tot = recombine_txt(numcores,xx,yy,output_filebase+"_temp_tau11tot",d[:2,:,:].shape)
        tau11tot_err = recombine_txt(numcores,xx,yy,output_filebase+"_temp_tau11tot_err",d[:2,:,:].shape)
        tau110 = recombine_txt(numcores,xx,yy,output_filebase+"_temp_tau110",d[:2,:,:].shape)
        tau110_err = recombine_txt(numcores,xx,yy,output_filebase+"_temp_tau110_err",d[:2,:,:].shape)
    else:
        do_chunk_fit(0,d[:,ys,xs],l[:,ys,xs],r[ys,xs],vrs,h,output_filebase)
        fitcube = recombine_txt(numcores,xs,ys,output_filebase+"_temp_fit",d.shape)
        fit_reg = recombine_txt(numcores,xs,ys,output_filebase+"_temp_fit_reg",d[:2,:,:].shape)
        vel = recombine_txt(numcores,xs,ys,output_filebase+"_temp_vel",d[:2,:,:].shape)
        vel_err = recombine_txt(numcores,xs,ys,output_filebase+"_temp_vel_err",d[:2,:,:].shape)
        width = recombine_txt(numcores,xs,ys,output_filebase+"_temp_width",d[:2,:,:].shape)
        width_err = recombine_txt(numcores,xs,ys,output_filebase+"_temp_width_err",d[:2,:,:].shape)
        tex = recombine_txt(numcores,xs,ys,output_filebase+"_temp_tex",d[:2,:,:].shape)
        tex_err = recombine_txt(numcores,xs,ys,output_filebase+"_temp_tex_err",d[:2,:,:].shape)
        tau11tot = recombine_txt(numcores,xs,ys,output_filebase+"_temp_tau11tot",d[:2,:,:].shape)
        tau11tot_err = recombine_txt(numcores,xs,ys,output_filebase+"_temp_tau11tot_err",d[:2,:,:].shape)
        tau110 = recombine_txt(numcores,xs,ys,output_filebase+"_temp_tau110",d[:2,:,:].shape)
        tau110_err = recombine_txt(numcores,xs,ys,output_filebase+"_temp_tau110_err",d[:2,:,:].shape)

    os.system("rm "+output_filebase+"_temp*")
    
    fit_reg_file = output_filebase+"_hf_fit_reg.fits"
    fit_file = output_filebase+"_hf_fit.fits"
    vel_file = output_filebase+"_hf_vel.fits"
    vel_err_file = output_filebase+"_hf_vel_err.fits"
    width_file = output_filebase+"_hf_width.fits"
    width_err_file = output_filebase+"_hf_width_err.fits"
    tex_file = output_filebase+"_hf_tex.fits"
    tex_err_file = output_filebase+"_hf_tex_err.fits"
    tau11tot_file = output_filebase+"_hf_tautot.fits"
    tau11tot_err_file = output_filebase+"_hf_tautot_err.fits"
    tau110_file = output_filebase+"_hf_tau0.fits"
    tau110_err_file = output_filebase+"_hf_tau0_err.fits"

    hfit = h[:]
    hfit['DATAMIN'] = -3.
    hfit['DATAMAX'] = 3.
    pyfits.writeto(fit_file,fitcube,hfit,overwrite=True)
    hfit_reg = edit_header(h[:])
    hfit_reg['DATAMIN'] = np.nanmin(fit_reg)
    hfit_reg['DATAMAX'] = np.nanmax(fit_reg)
    #pdb.set_trace()
    pyfits.writeto(fit_reg_file,fit_reg,hfit_reg,overwrite=True)
    #htex = strip_header(hfit[:],3)
    htex = edit_header(hfit[:])
    htex['DATAMIN'] = 0.
    htex['DATAMAX'] = np.nanmax(tex)
    pyfits.writeto(tex_file,tex,htex,overwrite=True)
    pyfits.writeto(tex_err_file,tex_err,htex,overwrite=True)
    hvel = htex[:]
    hvel['DATAMIN'] = np.nanmin(vel)
    hvel['DATAMAX'] = np.nanmax(vel)
    hvel['BUNIT'] = 'km/s'
    pyfits.writeto(vel_file,vel,hvel,overwrite=True)
    pyfits.writeto(vel_err_file,vel_err,hvel,overwrite=True)
    hwidth = hvel[:]
    hwidth['DATAMIN'] = 0.
    hwidth['DATAMAX'] = np.nanmax(width)
    pyfits.writeto(width_file,width,hwidth,overwrite=True)
    pyfits.writeto(width_err_file,width_err,hwidth,overwrite=True)
    htau11tot = hvel[:]
    htau11tot['DATAMIN'] = 0.
    htau11tot['DATAMAX'] = np.nanmax(tau11tot)
    pyfits.writeto(tau11tot_file,tau11tot,htau11tot,overwrite=True)
    pyfits.writeto(tau11tot_err_file,tau11tot_err,htau11tot,overwrite=True)
    htau110 = htau11tot[:]
    htau110['DATAMIN'] = 0.
    htau110['DATAMAX'] = np.nanmax(tau110)
    htau110['BUNIT'] = 'n/a'
    pyfits.writeto(tau110_file,tau110,htau110,overwrite=True)
    pyfits.writeto(tau110_err_file,tau110_err,htau110,overwrite=True)
   
def recombine_3D(numparts,filebase):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    for n in range(numparts):
        d = pyfits.getdata(filebase+"_"+str(n)+".fits")
        indata.append(d)
    final = np.dstack(indata)
    return(final)
    
def recombine_txt(numparts,xx,yy,filebase,final_shape):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    final = np.full(final_shape,np.nan)
    for n in range(numparts):
        d = np.loadtxt(filebase+"_"+str(n)+".txt")
        for i in np.arange(len(xx[n])):
            final[:,yy[n][i],xx[n][i]] = d[:,i]
    return(final)
    

def do_chunk_fit(num,data,label,rms,vranges,header,output_filebase):
    print(num)
    vax = get_vax(header)
    xarr11 = SpectroscopicAxis(vax*u.km/u.s,
                               velocity_convention='radio',
                               refX=freq_dict['oneone']).as_unit(u.GHz)

    fitcube = np.zeros_like(data)
    fit_reg=np.zeros(data[:2,:].shape)
    vel=np.full(data[:2,:].shape,np.nan)
    width=np.full(data[:2,:].shape,np.nan)
    tex=np.full(data[:2,:].shape,np.nan)
    tau11tot=np.full(data[:2,:].shape,np.nan)
    tau110=np.full(data[:2,:].shape,np.nan)
    vel_err=np.full(data[:2,:].shape,np.nan)
    width_err=np.full(data[:2,:].shape,np.nan)
    tex_err=np.full(data[:2,:].shape,np.nan)
    tau11tot_err=np.full(data[:2,:].shape,np.nan)
    tau110_err=np.full(data[:2,:].shape,np.nan)

    vmin = -10
    vmax = 160
    sigmin = 0.1
    sigmax = 5
    texmin = 2.7315
    texmax = 50
    ttotmin = 0.01
    ttotmax = 50
    par_bounds1 = [[vmin,sigmin,texmin,ttotmin],[vmax,sigmax,texmax,ttotmax]]
    par_bounds2 = [[vmin,sigmin,texmin,ttotmin,vmin,sigmin,texmin,ttotmin],[vmax,sigmax,texmax,ttotmax,vmax,sigmax,texmax,ttotmax]]
    for i in np.arange(data.shape[1]):
        print(i,100*float(i)/data.shape[1])
        s,l = [],[]
        for c in np.unique(label[:,i])[1:]:
            s.append(np.nansum(data[np.where(label[:,i]==c)[0],i]))
            l.append(c)
        cc = np.where(s==max(s))[0][0]
        ll = l[cc]
        wc = np.where(label[:,i]==ll)[0]
        vv = np.sum(vax[wc]*data[wc,i])/np.sum(data[wc,i])
        ww = max(len(wc)*(header['CDELT3']/1e3)/(2*(2*np.log(2))**(0.5)),0.2)
        init_pars1 = [vv,ww,5,5]
        fit1,pars1,perrs1 = fit_NH3_11_hf(data[:,i],vax,rms[i],init_pars1,par_bounds1)
        BIC_1 = get_BIC(data[:,i],fit1,rms[i],pars1)
        #if np.nanmax((data[:,i]-fit1)/rms[i]) > 3:
        try:
            res_snr_sums = nh3_kernel_sum((data[:,i]-fit1)/rms[i],header)
            v2 = vax[np.where(res_snr_sums==res_snr_sums.max())[0]]
            init_pars2 = np.concatenate((pars1,np.array([v2,pars1[1],pars1[2],pars1[3]])))
            fit2,pars2,perrs2 = fit_NH3_11_hf(data[:,i],vax,rms[i],init_pars2,par_bounds2)
            BIC_2 = get_BIC(data[:,i],fit2,rms[i],pars2)
        except:
            fit2 = np.zeros_like(data[:,i])
            pars2 = np.full((8,),np.nan)
            perrs2 = np.full((8,),np.nan)
            BIC_2 = np.nan
        try:
            tau110_1,tau110_err_1 = get_tau110_from_tau11tot(xarr11.value,uu.ufloat(pars1[0],perrs1[0]),uu.ufloat(pars1[1],perrs1[1]),uu.ufloat(pars1[3],perrs1[3]))
        except:
            tau110_1 = 0.
        try:
            tau110_2_1,tau110_err_2_1 = get_tau110_from_tau11tot(xarr11.value,uu.ufloat(pars2[0],perrs2[0]),uu.ufloat(pars2[1],perrs2[1]),uu.ufloat(pars2[3],perrs2[3]))
        except:
            tau110_2_1 = 0.
        try:
            tau110_2_2,tau110_err_2_2 = get_tau110_from_tau11tot(xarr11.value,uu.ufloat(pars2[4],perrs2[4]),uu.ufloat(pars2[5],perrs2[5]),uu.ufloat(pars2[7],perrs2[7]))
        except:
            tau110_2_2 = 0.
        
        amp1 = (pars1[2]-texmin)*(1-math.exp(-tau110_1))
        amp2_1 = (pars2[2]-texmin)*(1-math.exp(-tau110_2_1))
        amp2_2 = (pars2[6]-texmin)*(1-math.exp(-tau110_2_2))

        #"""
        lcomps = np.unique(label[:,i])[1:]
        vfits = [pars1[0],pars2[0],pars2[4]]
        crit = np.full((len(lcomps),len(vfits)),False)
        for j,c in enumerate(lcomps):
            for k,v in enumerate(vfits):
                wvrs = np.where(vranges[:,0]==c)[0][0]
                vrmin,vrmax = vranges[wvrs,1],vranges[wvrs,2]
                #pdb.set_trace()
                if v>=vrmin and v<=vrmax:
                    crit[j,k] = True
                else:
                    crit[j,k] = False
        
        nsig = 2
        if (amp2_1>nsig*rms[i] and amp2_2>nsig*rms[i]) and (BIC_2<(BIC_1-5) and crit[:,1].any() and crit[:,2].any()):
            c1 = np.where(crit[:,1])[0]
            if len(c1)>1:
                vdiff = []
                for cs in c1:
                    wwcs = np.where(label[:,i]==lcomps[cs])[0]
                    vvcs = np.sum(vax[wwcs]*data[wwcs,i])/np.sum(data[wwcs,i])
                    vdiff.append(abs(vfits[1]-vvcs))
                bc1 = np.where(vdiff == min(vdiff))
                fit_reg[0,i] = lcomps[bc1]
            else:
                fit_reg[0,i] = lcomps[c1]
            c2 = np.where(crit[:,2])[0]
            if len(c2)>1:
                vdiff = []
                for cs in c2:
                    wwcs = np.where(label[:,i]==lcomps[cs])[0]
                    vvcs = np.sum(vax[wwcs]*data[wwcs,i])/np.sum(data[wwcs,i])
                    vdiff.append(abs(vfits[2]-vvcs))
                bc2 = np.where(vdiff == min(vdiff))
                fit_reg[1,i] = lcomps[bc2]
            else:
                fit_reg[1,i] = lcomps[c2]
            fit = fit2
            pars = pars2
            perrs = perrs2
        elif amp1>nsig*rms[i] and crit[:,0].any():
            c1 = np.where(crit[:,0])[0]
            if len(c1)>1:
                vdiff = []
                for cs in c1:
                    wwcs = np.where(label[:,i]==lcomps[cs])[0]
                    vvcs = np.sum(vax[wwcs]*data[wwcs,i])/np.sum(data[wwcs,i])
                    vdiff.append(abs(vfits[0]-vvcs))
                bc1 = np.where(vdiff == min(vdiff))
                fit_reg[0,i] = lcomps[bc1]
            else:
                fit_reg[0,i] = lcomps[c1]
            fit = fit1
            pars = pars1
            perrs = perrs1
        else:
            fit = np.zeros_like(data[:,i])
            pars = np.full((8,),np.nan)
            perrs = np.full((8,),np.nan)
        #pdb.set_trace()

            
        fitcube[:,i] = fit
        for j in np.arange(len(pars)/4):
            vel[j,i] = pars[4*j]
            vel_err[j,i] = perrs[4*j]
            width[j,i] = pars[4*j+1]
            width_err[j,i] = perrs[4*j+1]
            tex[j,i] = pars[4*j+2]
            tex_err[j,i] = perrs[4*j+2]
            tau11tot[j,i] = pars[4*j+3]
            tau11tot_err[j,i] = perrs[4*j+3]
            if np.isfinite(pars[4*j]):
                tau110[j,i],tau110_err[j,i] = get_tau110_from_tau11tot(xarr11.value,uu.ufloat(pars[4*j],perrs[4*j]),uu.ufloat(pars[4*j+1],perrs[4*j+1]),uu.ufloat(pars[4*j+3],perrs[4*j+3]))
        #pdb.set_trace()

    np.savetxt(output_filebase+"_temp_fit_"+str(num)+".txt",fitcube)
    np.savetxt(output_filebase+"_temp_fit_reg_"+str(num)+".txt",fit_reg)
    np.savetxt(output_filebase+"_temp_vel_"+str(num)+".txt",vel)
    np.savetxt(output_filebase+"_temp_vel_err_"+str(num)+".txt",vel_err)
    np.savetxt(output_filebase+"_temp_width_"+str(num)+".txt",width)
    np.savetxt(output_filebase+"_temp_width_err_"+str(num)+".txt",width_err)
    np.savetxt(output_filebase+"_temp_tex_"+str(num)+".txt",tex)
    np.savetxt(output_filebase+"_temp_tex_err_"+str(num)+".txt",tex_err)
    np.savetxt(output_filebase+"_temp_tau11tot_"+str(num)+".txt",tau11tot)
    np.savetxt(output_filebase+"_temp_tau11tot_err_"+str(num)+".txt",tau11tot_err)
    np.savetxt(output_filebase+"_temp_tau110_"+str(num)+".txt",tau110)
    np.savetxt(output_filebase+"_temp_tau110_err_"+str(num)+".txt",tau110_err)


def fit_NH3_11_hf(spec,vax,rms,init_pars,par_bounds):
    try:
        xarr11 = SpectroscopicAxis(vax*u.km/u.s,
                                   velocity_convention='radio',
                                   refX=freq_dict['oneone']).as_unit(u.GHz)
        noise = np.full(spec.shape,rms)
        #pdb.set_trace()
        popt, pcov = curve_fit(NH3_11_hf_func, xarr11.value, spec,
                               p0=init_pars,sigma=noise,
                               bounds=par_bounds)
        fit = NH3_11_hf_func(xarr11.value,*popt)
        perr = np.sqrt(np.diag(pcov))
        #pdb.set_trace()
    #"""
    except:
        #pdb.set_trace()
        fit = np.zeros_like(spec)
        popt = np.full((len(init_pars),),np.nan)
        perr = np.full((len(init_pars),),np.nan)
    #"""
    return(fit,popt,perr)

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

def edit_header(h,n=2):
    h['NAXIS3'] = n
    h['CRPIX3'] = 0.0
    h['CDELT3'] = 1.0
    h['CUNIT3'] = 'Components'
    h['CRVAL3'] = 0.0
    return(h)


def NH3_11_hf_func(x,*pars):
    model = np.zeros_like(x)
    linename = 'oneone'
    npars = 4
    Tbkg = 2.7315
    voff_lines = np.array(voff_lines_dict[linename])
    tau_wts = np.array(tau_wts_dict[linename])
    lines = (1-voff_lines/ckms)*freq_dict[linename]/1e9
    tau_wts = tau_wts / (tau_wts).sum()
    for i in range(len(pars)/npars):
        nuwidth = np.abs(pars[1+i*npars]/ckms*lines)
        xoff_v = pars[0+i*npars]
        nuoff = xoff_v/ckms*lines
        # tau array
        tauprof = np.zeros_like(model)
        for kk,nuo in enumerate(nuoff):
            tauprof += (pars[3+i*npars] * tau_wts[kk] *
                        np.exp(-(x+nuo-lines[kk])**2 /
                               (2.0*nuwidth[kk]**2)))
        model += (pars[2+i*npars]-Tbkg)*(1-np.exp(-1.*tauprof))
    #pdb.set_trace()
    return model

def lnlike_line(theta, x, y, yerr):
    model = NH3_11_hf_func(x,*theta)
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior_line(theta,pbs):
    ndim = len(theta)
    npars = 4
    ncomp = int(ndim/npars)
    prs = []
    for i in range(ncomp*npars):
        prs.append([pbs[0][i],pbs[1][i]])
    crits = []
    for cc,rs in enumerate(prs):
        if np.logical_and(theta[cc::ndim].min()>rs[0],
                          theta[cc::ndim].max()<rs[1]):
            crits.append(True)
        else:
            crits.append(False)
    if False in crits:
        return -np.inf
    else:
        lps = []
        for i,p in enumerate(theta):
            if i%npars != 0:
                lps.append(p*np.log(prs[i%npars][1]/prs[i%npars][0]))
            else:
                lps.append(prs[i%npars][1]-prs[i%npars][0])
        lp = -1.*np.log(np.prod(np.asarray(lps)))
        return lp

def lnprob_line(theta, x, y, yerr, pbs):
    lp = lnprior_line(theta,pbs)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_line(theta, x, y, yerr)

def get_tau110_from_tau11tot(xarr,vel,width,tau11_tot):
    linename = 'oneone'

    #pdb.set_trace()
    voff_lines = np.array(voff_lines_dict[linename])
    tau_wts = np.array(tau_wts_dict[linename])
    lines = (1-voff_lines/ckms)*freq_dict[linename]/1e9
    tau_wts = tau_wts / (tau_wts).sum()
    nuwidth = np.abs(width/ckms*lines)
    xoff_v = vel
    nuoff = xoff_v/ckms*lines
    # tau array
    tauprofn = np.zeros(len(xarr))
    tauprofs = np.zeros(len(xarr))
    for kk,nuo in enumerate(nuoff):
        #pdb.set_trace()
        #for i,x in enumerate(xarr):
        tauprofn = tauprofn + (tau11_tot.nominal_value * tau_wts[kk] *
                             unp.exp(-(xarr+nuo.nominal_value-lines[kk])**2/
                                     (2.0*nuwidth[kk].nominal_value**2)))
        tauprofs = tauprofs + (tau11_tot * tau_wts[kk] *
                             unp.exp(-(xarr+nuo-lines[kk])**2/
                                     (2.0*nuwidth[kk]**2)))

         
    freq = v2f_py(xoff_v,linename)/1e9
    peak_loc = np.where(abs(freq-xarr) == abs(freq-xarr).min())
    tau0 = tauprofn[peak_loc][0]
    tau0_err = tauprofs[peak_loc][0].std_dev
    #pdb.set_trace()
    """
    print tau0
    plt.plot(tauprofn)
    plt.show()
    pdb.set_trace()
    #"""
    return tau0,tau0_err

def get_BIC(data,model,rms,pars):
    loglike = -1.*(0.5*np.log(2*np.pi)+np.log(rms)+get_chi2(data,model,rms)/2)
    k = len(pars)
    N = len(data)
    BIC = np.log(N)*k-2*loglike
    return(BIC)

def get_chi2(d,m,s):
    return np.sum((d-m)**2/s**2)

def c2v(c,h):
    v = ((c - (h['CRPIX3']-1))*h['CDELT3'] + h['CRVAL3'])*0.001
    return v

def v2c(v,h):
    c = (v*1000. - h['CRVAL3'])/h['CDELT3'] + (h['CRPIX3']-1)
    return int(round(c))

def v2f_py(v,linename):
    f0 = freq_dict[linename]
    f = (f0)*(1-(v/ckms))
    return f

def f2v_py(f,linename):
    f0 = freq_dict[linename]
    v = ckms*(1-(f/f0))
    return v

def get_vax(h):
    vmin = c2v(0,h)
    vmax = c2v(h['NAXIS3']-1,h)
    vax = np.linspace(vmin,vmax,h['NAXIS3'])
    return vax

def nh3_kernel_sum(spec,h):
    vsats = np.array([-19.421,-7.771,7.760,19.408])
    csats = (vsats/(h['CDELT3']/1e3)).astype(int)
    ya  = rolling_window(spec,int(1e3/h['CDELT3']))
    ya_ma = mask_snr_spec(ya,snr=1)
    sums = ma.sum(ya_ma,-1)
    sum_errs = np.sqrt(ma.count(ya_ma,-1))
    sum_lo = np.roll(sums,csats[0])
    sum_li = np.roll(sums,csats[1])
    sum_ri = np.roll(sums,csats[2])
    sum_ro = np.roll(sums,csats[3])
    combined = sums+sum_lo+sum_li+sum_ri+sum_ro
    return(combined)

def get_mask_vranges(label,h):
    vrs = np.zeros((len(np.unique(label)[1:]),3))
    for i,l in enumerate(np.unique(label)[1:]):
        wcc = np.where(label==l)[0]
        vmin = c2v(wcc.min(),h)
        vmax = c2v(wcc.max(),h)
        vrs[i,:] = np.array([l,vmin,vmax])
    return(vrs)


if __name__ == '__main__':
    main()

