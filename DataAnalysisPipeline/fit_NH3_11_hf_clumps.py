"""
fit_NH3_11_hf_clump.py

Fit NH3(1,1) data cube using an ammonia model that includes
the magnetic hyperfine transitions. This module outputs the 
best-fit model, the fit parameters, and the errors on the 
fit parameters. Fit parameters include the excitation 
temperature (tex), the clump velocity (vel), the velocity
dispersion (sigma), and the total integrated optical
depth (tau11_tot). Additionally, outputs the line 
center optical depth (tau11_0).

This version runs in parallel, which is useful because
the process is fairly slow.

Example:
python fit_NH3_11_hf_clumps.py 
       -i L30_Tile01-04_NH3_1-1_fixed.fits 
       -o L30_Tile01-04_NH3_1-1.fits 

-i : Input      -- Input NH3(1,1) data cube file 
-l : Label      -- NH3(1,1) 3D label cube file 
-r : rms noise  -- NH3(1,1) rms noise map file 
-o : Outputbase -- Filebase for the output data
-n : Numcores   -- Number of cores available for parallized computing
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
import multiprocessing as mp
import uncertainties as uu
from uncertainties import unumpy as unp
from uncertainties.umath import *
from scipy.optimize import curve_fit
from scipy import stats
from pyspeckit.spectrum.models.ammonia_constants import (line_names, 
                                                         freq_dict, 
                                                         aval_dict, 
                                                         ortho_dict, 
                                                         voff_lines_dict, 
                                                         tau_wts_dict, 
                                                         ckms, ccms, 
                                                         h, kb, 
                                                         Jortho, 
                                                         Jpara, Brot, 
                                                         Crot)
from pyspeckit.spectrum.units import SpectroscopicAxis, SpectroscopicAxes
from astropy import units as u
import pdb

#Constants
T_CMB = 2.7315 #K, Temperature of the Cosmic Microwave Background
fwhm_to_sigma = (2*(2*np.log(2)))**(-0.5) #Convert Gaussian fwhm to sigma

def main():
    #Defaults
    numcores = 1
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:l:r:n:o:h")
    except getopt.GetoptError as err:
        print(err.msg)
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
        

    #Read in data into arrays
    d,h = pyfits.getdata(input_file,header=True)
    labels_3D = pyfits.getdata(label_file)
    rms_map = pyfits.getdata(rms_file)
    #Mask nonlabeled channels that are not associated with a clump
    mask_3D = np.zeros(labels_3D.shape)
    unmasked_voxs = np.where(labels_3D>0)
    mask_3D[unmasked_voxs] = 1.
    #Get the pixel values of spectra that have clump emission
    main_peak = np.max(d*mask_3D,axis=0)
    clump_pxs = np.where(main_peak>0)
    xgrid,ygrid = np.meshgrid(np.arange(rms_map.shape[1]),
                              np.arange(rms_map.shape[0]))
    xpixs,ypixs = xgrid[clump_pxs],ygrid[clump_pxs]
    #Get the velocity ranges for each labeled clump
    vranges = get_vranges_from_labels(labels_3D,h)
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
        #Split the arrays of pixel values
        xsplit = np.array_split(xpixs, numcores, 0)
        ysplit = np.array_split(ypixs, numcores, 0)
        procs = []
        """
        Fit the NH3(1,1) spectra and write best-fit 
        parameters and errors to temporary files
        """
        for num in range(len(xsplit)):
            procs.append(mp.Process(target=do_chunk_fit,
                                    args=(num,d[:,ysplit[num],xsplit[num]],
                                          labels_3D[:,ysplit[num],xsplit[num]],
                                          rms_map[ysplit[num],xsplit[num]],
                                          vranges,h,output_filebase)))
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()

        #Recombine temporary files into parameter maps
        modelcube = recombine_txt(numcores,xsplit,ysplit,
                                  output_filebase+"_fit",
                                  d.shape)
        fit_labels = recombine_txt(numcores,xsplit,ysplit,
                                   output_filebase+"_fit_labels",
                                   d[:2,:,:].shape)
        vel = recombine_txt(numcores,xsplit,ysplit,
                            output_filebase+"_vel",
                            d[:2,:,:].shape)
        vel_err = recombine_txt(numcores,xsplit,ysplit,
                                output_filebase+"_vel_err",
                                d[:2,:,:].shape)
        sigma = recombine_txt(numcores,xsplit,ysplit,
                              output_filebase+"_sigma",
                              d[:2,:,:].shape)
        sigma_err = recombine_txt(numcores,xsplit,ysplit,
                                  output_filebase+"_sigma_err",
                                  d[:2,:,:].shape)
        tex = recombine_txt(numcores,xsplit,ysplit,
                            output_filebase+"_tex",
                            d[:2,:,:].shape)
        tex_err = recombine_txt(numcores,xsplit,ysplit,
                                output_filebase+"_tex_err",
                                d[:2,:,:].shape)
        tau11_tot = recombine_txt(numcores,xsplit,ysplit,
                                  output_filebase+"_tau11_tot",
                                  d[:2,:,:].shape)
        tau11_tot_err = recombine_txt(numcores,xsplit,ysplit,
                                      output_filebase+"_tau11_tot_err",
                                      d[:2,:,:].shape)
        tau11_0 = recombine_txt(numcores,xsplit,ysplit,
                                output_filebase+"_tau11_0",
                                d[:2,:,:].shape)
        tau11_0_err = recombine_txt(numcores,xsplit,ysplit,
                                    output_filebase+"_tau11_0_err",
                                    d[:2,:,:].shape)
    else:
        """
        Fit the NH3(1,1) spectra and write best-fit 
        parameters and errors to temporary files
        """
        do_chunk_fit(0,d[:,ypixs,xpixs],
                     labels_3D[:,ypixs,xpixs],
                     rms_map[ypixs,xpixs],
                     vranges,h,output_filebase)
        #Recombine temporary files into parameter maps
        modelcube = recombine_txt(numcores,xpixs,ypixs,
                                  output_filebase+"_fit",
                                  d.shape)
        fit_labels = recombine_txt(numcores,xpixs,ypixs,
                                output_filebase+"_fit_labels",
                                d[:2,:,:].shape)
        vel = recombine_txt(numcores,xpixs,ypixs,
                            output_filebase+"_vel",
                            d[:2,:,:].shape)
        vel_err = recombine_txt(numcores,xpixs,ypixs,
                                output_filebase+"_vel_err",
                                d[:2,:,:].shape)
        sigma = recombine_txt(numcores,xpixs,ypixs,
                              output_filebase+"_sigma",
                              d[:2,:,:].shape)
        sigma_err = recombine_txt(numcores,xpixs,ypixs,
                                  output_filebase+"_sigma_err",
                                  d[:2,:,:].shape)
        tex = recombine_txt(numcores,xpixs,ypixs,
                            output_filebase+"_tex",
                            d[:2,:,:].shape)
        tex_err = recombine_txt(numcores,xpixs,ypixs,
                                output_filebase+"_tex_err",
                                d[:2,:,:].shape)
        tau11_tot = recombine_txt(numcores,xpixs,ypixs,
                                  output_filebase+"_tau11_tot",
                                  d[:2,:,:].shape)
        tau11_tot_err = recombine_txt(numcores,xpixs,ypixs,
                                      output_filebase+"_tau11_tot_err",
                                      d[:2,:,:].shape)
        tau11_0 = recombine_txt(numcores,xpixs,ypixs,
                                output_filebase+"_tau11_0",
                                d[:2,:,:].shape)
        tau11_0_err = recombine_txt(numcores,xpixs,ypixs,
                                    output_filebase+"_tau11_0_err",
                                    d[:2,:,:].shape)
    #Delete temporary files
    os.system("rm "+output_filebase+"*_temp*.txt")
    
    #Define parameter map file names
    fit_labels_file = output_filebase+"_hf_fit_labels.fits"
    fit_file = output_filebase+"_hf_fit.fits"
    vel_file = output_filebase+"_hf_vel.fits"
    vel_err_file = output_filebase+"_hf_vel_err.fits"
    sigma_file = output_filebase+"_hf_sigma.fits"
    sigma_err_file = output_filebase+"_hf_sigma_err.fits"
    tex_file = output_filebase+"_hf_tex.fits"
    tex_err_file = output_filebase+"_hf_tex_err.fits"
    tau11_tot_file = output_filebase+"_hf_tautot.fits"
    tau11_tot_err_file = output_filebase+"_hf_tautot_err.fits"
    tau11_0_file = output_filebase+"_hf_tau0.fits"
    tau11_0_err_file = output_filebase+"_hf_tau0_err.fits"

    """
    Edit headers and write parameter files
    """
    hfit = h[:]
    hfit['DATAMIN'] = -3.
    hfit['DATAMAX'] = 3.
    pyfits.writeto(fit_file,modelcube,hfit,overwrite=True)
    hfit_labels = edit_header(h[:])
    hfit_labels['DATAMIN'] = np.nanmin(fit_labels)
    hfit_labels['DATAMAX'] = np.nanmax(fit_labels)
    pyfits.writeto(fit_labels_file,fit_labels,hfit_labels,overwrite=True)
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
    hsigma = hvel[:]
    hsigma['DATAMIN'] = 0.
    hsigma['DATAMAX'] = np.nanmax(sigma)
    pyfits.writeto(sigma_file,sigma,hsigma,overwrite=True)
    pyfits.writeto(sigma_err_file,sigma_err,hsigma,overwrite=True)
    htau11_tot = hvel[:]
    htau11_tot['DATAMIN'] = 0.
    htau11_tot['DATAMAX'] = np.nanmax(tau11_tot)
    pyfits.writeto(tau11_tot_file,tau11_tot,htau11_tot,overwrite=True)
    pyfits.writeto(tau11_tot_err_file,tau11_tot_err,htau11_tot,overwrite=True)
    htau11_0 = htau11_tot[:]
    htau11_0['DATAMIN'] = 0.
    htau11_0['DATAMAX'] = np.nanmax(tau11_0)
    htau11_0['BUNIT'] = 'n/a'
    pyfits.writeto(tau11_0_file,tau11_0,htau11_0,overwrite=True)
    pyfits.writeto(tau11_0_err_file,tau11_0_err,htau11_0,overwrite=True)
   
def recombine_txt(numparts,xsplit,ysplit,filebase,final_shape):
    """
    Recombine all the individual fits files into 
    one final image
    """
    indata = []
    final = np.full(final_shape,np.nan)
    for n in range(numparts):
        d = np.loadtxt(filebase+"_temp"+str(n)+".txt")
        for i in np.arange(len(xsplit[n])):
            final[:,ysplit[n][i],xsplit[n][i]] = d[:,i]
    return(final)
    

def do_chunk_fit(num,spec_arr,label_arr,rms_arr,vranges,header,output_filebase):
    """
    Loop over M NH3(1,1) spectra of length N to fit them with an 
    NH3(1,1) model and return the best-fit parameters and errors. 
    Write these values to text files for later creation of parameter 
    maps.

    Parameters
    ----------
    num : int
        The chunk number, which corresponds to an independent core for
        parallelized computing.
    spec_arr : ndarray
        An NxM array that stores the input spectra, where N is the 
        length of a spectrum and M is the number of spectra in a chunk.
    label_arr : ndarray
        An array the same shape as spec_arr, where channel values are
        equal to the clump label index of their associated clump.
        Zero-valued channels are unassociated with a clump.
    rms_arr : ndarray
        1D array of size M that records the rms noise for each spectrum.
    vranges : ndarray
        An Lx3 array, where L is the number of clumps detected in the
        input data cube. Rows correspond to individual clumps, the 
        first column contains the clump label indices, the second 
        column contains the minimum velocity of the labeled channels, 
        and the third column contains the maximum velocity of the 
        labeled channels.
    header : string
        The FITS header of the input data cube.
    output_filebase : string
        The base string for the output files.

    Returns
    -------
    None.

    """

    print(num)
    
    #Create the velocity and frequency axes
    vax = get_vax(header)
    freq11ax = vax_to_freq11ax(vax)

    #Create nan arrays to store parameters and write to temporary files
    model_arr = np.zeros_like(spec_arr)
    fit_labels_arr = np.zeros(spec_arr[:2,:].shape)
    vel_arr = np.full(spec_arr[:2,:].shape,np.nan)
    sigma_arr = np.full(spec_arr[:2,:].shape,np.nan)
    tex_arr = np.full(spec_arr[:2,:].shape,np.nan)
    tau11_tot_arr = np.full(spec_arr[:2,:].shape,np.nan)
    tau11_0_arr = np.full(spec_arr[:2,:].shape,np.nan)
    vel_err_arr = np.full(spec_arr[:2,:].shape,np.nan)
    sigma_err_arr = np.full(spec_arr[:2,:].shape,np.nan)
    tex_err_arr = np.full(spec_arr[:2,:].shape,np.nan)
    tau11_tot_err_arr = np.full(spec_arr[:2,:].shape,np.nan)
    tau11_0_err_arr = np.full(spec_arr[:2,:].shape,np.nan)

    #Bound the parameter values to reasonable values for the fits
    vmin = -10
    vmax = 140
    sigmin = 0.1
    sigmax = 5
    texmin = T_CMB
    texmax = 50
    ttotmin = 0.01
    ttotmax = 50
    par_bounds = [[vmin,sigmin,texmin,ttotmin],[vmax,sigmax,texmax,ttotmax]]
    #Loop over each spectrum in the array
    for i in np.arange(spec_arr.shape[1]):
        print(num,i,100*float(i)/spec_arr.shape[1])
        #Fit spectra with a one- or two-component NH3(1,1) model
        fit_labels,pars,perrs = fit_wrapper(spec_arr[:,i],vax,header,
                                            label_arr[:,i],rms_arr[i],
                                            par_bounds,vranges)
        """
        Create a NH3(1,1) model from the best-fit parameters and store
        in the output model array
        """
        model_arr[:,i] = NH3_11_hf_func(freq11ax,*pars)
        """
        Loop over velocity components and store fit results in the
        output parameter arrays
        """
        for j in np.arange(int(len(pars)/4)):
            fit_labels_arr[j,i] = fit_labels[j]
            vel_arr[j,i] = pars[4*j]
            vel_err_arr[j,i] = perrs[4*j]
            sigma_arr[j,i] = pars[4*j+1]
            sigma_err_arr[j,i] = perrs[4*j+1]
            tex_arr[j,i] = pars[4*j+2]
            tex_err_arr[j,i] = perrs[4*j+2]
            tau11_tot_arr[j,i] = pars[4*j+3]
            tau11_tot_err_arr[j,i] = perrs[4*j+3]
            if np.isfinite(pars[4*j]):
                tau11_0_arr[j,i],\
                tau11_0_err_arr[j,i] = get_tau11_0(freq11ax,
                                                   pars[4*j:4*(j+1)],
                                                   perrs[4*j:4*(j+1)])
                                                           
    """
    Write parameter arrays to temporary files for later combination
    to create parameter maps
    """
    np.savetxt(output_filebase+"_model_temp"+str(num)+".txt",model_arr)
    np.savetxt(output_filebase+"_fit_labels_temp"+str(num)+".txt",fit_labels_arr)
    np.savetxt(output_filebase+"_vel_temp"+str(num)+".txt",vel_arr)
    np.savetxt(output_filebase+"_vel_err_temp"+str(num)+".txt",vel_err_arr)
    np.savetxt(output_filebase+"_sigma_temp"+str(num)+".txt",sigma_arr)
    np.savetxt(output_filebase+"_sigma_err_temp"+str(num)+".txt",sigma_err_arr)
    np.savetxt(output_filebase+"_tex_temp"+str(num)+".txt",tex_arr)
    np.savetxt(output_filebase+"_tex_err_temp"+str(num)+".txt",tex_err_arr)
    np.savetxt(output_filebase+"_tau11_tot_temp"+str(num)+".txt",tau11_tot_arr)
    np.savetxt(output_filebase+"_tau11_tot_err_temp"+str(num)+".txt",tau11_tot_err_arr)
    np.savetxt(output_filebase+"_tau11_0_temp"+str(num)+".txt",tau11_0_arr)
    np.savetxt(output_filebase+"_tau11_0_err_temp"+str(num)+".txt",tau11_0_err_arr)


def fit_wrapper(spec,vax,header,labeled_chans,rms,par_bounds,vranges):
    """
    Fit a spectrum with a one- and two-component NH3(1,1) model.
    Determine which model is the better fit to the data and has
    a velocity that is close to the velocity estimated by the
    ammonia_clumpfind module.

    Parameters
    ----------
    spec : ndarray
        The NH3(1,1) spectrum to be fit.
    vax : ndarray
        The corresponding velocity axis with size equal to spec size.
    header : string
        The FITS header of the input data cube.
    labeled_chans : ndarray
        An array the size of spec, where channels are labeled with
        integer values corresponding to clump label indices. 
        Zero-valued channels indicate the channel is not associated 
        with a detected clump.
    rms : float
        The rms noise in the spectrum.
    par_bounds : list
        A list of two lists. The first list contains the lower bounds
        for the fit parameters and the second list contains the upper 
        bounds for the fit parameters.
    vranges : ndarray
        Rows correspond to individual clumps, the first column
        contains the clump label indices, the second column contains
        the minimum velocity of the labeled channels, and the third
        column contains the maximum velocity of the labeled channels.

    Returns
    -------
    fit_labels : list
        List of clump label indices that tie each component of a fit
        to a particular labeled clump. 
    pars : ndarray
        1D array that contains the best-fit parameters. The number
        of parameters is 4X the number of components.
    perrs : ndarray
        1D array that contains the statistical errors on the 
        best-fit parameters.
    """
    
    #Only pass finite channels, since nans will cause the fit to fail
    finite_chans = np.where(np.isfinite(spec))
    spec,vax,labeled_chans = spec[finite_chans],\
                             vax[finite_chans],\
                             labeled_chans[finite_chans]
    """
    Find the clump component with the greatest main line integrated 
    intensity and make it the first component to fit.
    """
    channel_sums,labels = [],[]
    for label in np.unique(labeled_chans)[1:]:
        channel_sums.append(spec[np.where(labeled_chans==label)[0]].sum())
        labels.append(label)
    first_component_loc = np.argmax(channel_sums)
    first_component_label = labels[first_component_loc]
    """
    Initialize the parameters close to the expected values 
    for the first component using the labeled channels.
    """
    fc_chans = np.where(labeled_chans==first_component_label)[0]
    vel_init = (vax[fc_chans]*spec[fc_chans]).sum()/spec[fc_chans].sum()
    sigma_init = max(len(fc_chans)*(header['CDELT3']/1e3)*fwhm_to_sigma,0.2)
    init_pars1 = [vel_init,sigma_init,5,5]
    #Perform a single-component fit
    freq11ax = vax_to_freq11ax(vax)
    pars1,perrs1 = fit_NH3_11_hf(spec,freq11ax,rms,init_pars1,par_bounds)
    #Create a model spectrum from the best-fit parameters
    model1 = NH3_11_hf_func(freq11ax,*pars1)
    #Calculate the Bayesian Information Criterion to assess goodness of fit
    BIC_1 = get_BIC(spec,model1,rms,pars1)
    
    """    
    Attempt a two-component fit. If it fails, return nans for the
    best-fit two-component parameters, errors, model, and the BIC.
    """
    try:
        """
        Find the velocity where the residual emission most closely
        resembles the NH3 satellite line pattern
        """
        vel2 = get_peak_vel((spec-model1)/rms,vax,header)
        """
        Initialize second-component parameters to velocity found 
        from residual and best-fit parameters from the first fit.
        """
        init_pars2 = np.concatenate((pars1,
                                     np.array([vel2,
                                               pars1[1],
                                               pars1[2],
                                               pars1[3]])))
        #Use same parameter bounds for second component
        par_bounds[0] += par_bounds[0]
        par_bounds[1] += par_bounds[1]
        #Perform a two-component fit
        pars2,perrs2 = fit_NH3_11_hf(spec,freq11ax,rms,init_pars2,par_bounds)
        #Create a model spectrum from the best-fit parameters
        model2 = NH3_11_hf_func(freq11ax,*pars2)
        #Calculate the BIC to assess goodness of fit
        BIC_2 = get_BIC(spec,model2,rms,pars2)
    except:
        model2 = np.zeros_like(spec)
        pars2 = np.full((8,),np.nan)
        perrs2 = np.full((8,),np.nan)
        BIC_2 = np.nan
        
    #Calculate the BIC for a model with no emission
    BIC_0 = get_BIC(spec,np.zeros_like(spec),rms,[])
    """
    If the zero-component BIC is better than the one- or two-
    component BICs, return nans for clump labels and parameters.
    I require the one- and two-component BICs to be 5 and 10 less
    than the zero parameter model, since less parameters should
    be favored unless the fit is much improved.
    """
    BIC is better than the     if BIC_0 < (BIC_1+5) and BIC_0 < (BIC_2+10):
        pars = np.full((4,),np.nan)
        perrs = np.full((4,),np.nan)
        return([np.nan],pars,perrs)
        
    #Calculate the model main line amplitudes 
    amp1 = get_model_amp(freq11ax,pars1,perrs1)
    amp2_1 = get_model_amp(freq11ax,pars2[:4],perrs2[:4])
    amp2_2 = get_model_amp(freq11ax,pars2[4:],perrs2[4:])
    
    """
    Check that the best-fit velocities fall within the velocity
    range found by the ammonia_clumpfind routine for the clumps
    that display emission in this spectrum.
    """
    component_labels = np.unique(labeled_chans)[1:]
    vel_pars = [pars1[0],pars2[0],pars2[4]]
    within_vrange = np.full((len(component_labels),len(vel_pars)),False)
    for j,label in enumerate(component_labels):
        for k,vel_par in enumerate(vel_pars):
            label_loc = np.where(vranges[:,0]==label)[0][0]
            vrange_min,vrange_max = vranges[label_loc,1],vranges[label_loc,2]
            if vel_par>=vrange_min and vel_par<=vrange_max:
                within_vrange[j,k] = True
            else:
                within_vrange[j,k] = False

    """
    Set the significance threshold for detection. I set this lower
    than 3 sigma because the ammonia_clumpfind routine provides
    additional information about whether emission is genuine.
    """
    nsig = 2
    fit_labels = []
    """
    Accept the two-component fit as the better model if the amplitudes
    of both lines satisfy the significance threshold, the two-component
    BIC is at least 5 points lower than the one-component BIC, and both
    components fall within the velocity range of one of the clumps
    that displays emission within the spectrum.
    """
    if ((amp2_1>nsig*rms and amp2_2>nsig*rms) and 
        BIC_2<(BIC_1-5) and 
        within_vrange[:,1].any() and within_vrange[:,2].any()):
        """
        Find the clump whose labeled channels have the closest 
        intensity-weighted velocity to the best-fit velocity of
        the first component. Assign this clump label index to 
        the first component fit results.
        """
        two_comp_vel1_labels_idx = np.where(within_vrange[:,1])[0]
        two_comp_vel1_labels = component_labels[two_comp_vel1_labels_idx]
        if len(two_comp_vel1_labels)>1:
            vel_diff = []
            for label in two_comp_vel1_labels:
                clump_chans = np.where(labeled_chans==label)[0]
                int_weighted_vel = ((vax[clump_chans]*
                                     spec[clump_chans]).sum()/
                                    spec[clump_chans].sum())
                vel_diff.append(abs(vel_pars[1]-int_weighted_vel))
            two_comp_vel1_label_idx = np.argmin(vel_diff)
            fit_labels.append(two_comp_vel1_labels[two_comp_vel1_label_idx])
        else:
            fit_labels.append(two_comp_vel1_labels[0])

        """
        Find the clump whose labeled channels have the closest 
        intensity-weighted velocity to the best-fit velocity of
        the seco component. Assign this clump label index to 
        the first component fit results.
        """
        two_comp_vel2_labels_idx = np.where(within_vrange[:,2])[0]
        two_comp_vel2_labels = component_labels[two_comp_vel2_labels_idx]
        if len(two_comp_vel2_labels)>1:
            vel_diff = []
            for label in two_comp_vel2_labels:
                clump_chans = np.where(labeled_chans==label)[0]
                int_weighted_vel = ((vax[clump_chans]*
                                     spec[clump_chans]).sum()/
                                    spec[clump_chans].sum())
                vel_diff.append(abs(vel_pars[2]-int_weighted_vel))
            two_comp_vel2_label_idx = np.argmin(vel_diff)
            fit_labels.append(two_comp_vel2_labels[two_comp_vel2_label_idx])
        else:
            fit_labels.append(two_comp_vel2_labels[0])

        pars = pars2
        perrs = perrs2
        
    elif amp1>nsig*rms and within_vrange[:,0].any():
        """
        If the two-component model is not accepted, accept the one-component 
        model if the amplitude of the main line satisfies the significance 
        threshold and the best-fits velocity falls within the velocity range 
        of a clump that displays emission within the spectrum.
        """
        
        """
        Find the clump label index whose intensity-weighted velocity is
        closest to the best-fit velocity.
        """
        one_comp_vel1_labels_idx = np.where(within_vrange[:,0])[0]
        one_comp_vel1_labels = component_labels[one_comp_vel1_labels_idx]
        if len(one_comp_vel1_labels)>1:
            vel_diff = []
            for label in one_comp_vel1_labels:
                clump_chans = np.where(labeled_chans==label)[0]
                int_weighted_vel = ((vax[clump_chans]*
                                     spec[clump_chans]).sum()/
                                    spec[clump_chans].sum())
                vel_diff.append(abs(vel_pars[0]-int_weighted_vel))
            one_comp_vel1_label_idx = np.argmin(vel_diff)
            fit_labels.append(one_comp_vel1_labels[one_comp_vel1_label_idx])
        else:
            fit_labels.append(one_comp_vel1_labels[0])

        pars = pars1
        perrs = perrs1
    else:
        """
        If neither the one- or two-component fits satisfy all criteria,
        return nans for the fit labels, parameters, and parameter errors.
        """
        fit_labels.append(np.nan)
        pars = np.full((4,),np.nan)
        perrs = np.full((4,),np.nan)
    if len(fit_labels)==0: pdb.set_trace()
    return(fit_labels,pars,perrs)


def fit_NH3_11_hf(spec,freq11ax,rms,init_pars,par_bounds):
    """
    Fit an NH3(1,1) spectrum with the magnetic hyperfine
    model. If the fit is successful, return the best-fit
    parameters and errors. If the fit is unsuccessful,
    return an array of nan values.

    Parameters
    ----------
    spec : ndarray
        The NH3(1,1) spectrum to be fit.
    vax : ndarray
        The corresponding velocity axis with size equal to spec size.
    rms : float
        The rms noise in the spectrum.
    init_pars : list
        A list of floats to initialize the parameters in the fit.
    par_bounds : list
        A list of two lists. The first list contains the lower bounds
        for the fit parameters and the second list contains the upper 
        bounds for the fit parameters.

    Returns
    -------
    popt : ndarray
        The best-fit parameter values as an array of floats
    perr : ndarray
        The statistical errors on the best-fit parameter 
        values as an array of floats
    """


    try:
        noise = np.full(spec.shape,rms)

        popt, pcov = curve_fit(NH3_11_hf_func, freq11ax, spec,
                               p0=init_pars,sigma=noise,
                               bounds=par_bounds)
        perr = np.sqrt(np.diag(pcov))
    except:
        popt = np.full((len(init_pars),),np.nan)
        perr = np.full((len(init_pars),),np.nan)
    return(popt,perr)

def NH3_11_hf_func(freq11ax,*pars):
    """
    Return model NH3(1,1) line using the parameters provided.
    The model parameters are the excitation temperature (tex), 
    the velocity (vel), the velocity dispersion (sigma), 
    and the total integrated optical depth (tau11_tot). The
    output model includes the magnetic hyperfine lines.
    """
    model = np.zeros_like(freq11ax)
    #Define the NH3 transition and parameters
    linename = 'oneone'
    npars = 4
    vels = pars[::npars]
    sigmas = pars[1::npars]
    texs = pars[2::npars]
    tau11_tots = pars[3::npars]
    #Define the hyperfine line frequency offsets
    voff_lines = np.array(voff_lines_dict[linename])
    lines = (1-voff_lines/ckms)*freq_dict[linename]/1e9
    #Define the optical depth weights for the hyperfine lines
    tau_wts = np.array(tau_wts_dict[linename])
    tau_wts = tau_wts / (tau_wts).sum()
    """
    Loop over the velocity components to model emission 
    from multiple clumps along the line of sight.
    """
    for i in np.arange(len(vels)):
        #Convert velocity dispersion and offset to frequency
        nusigma = np.abs(sigmas[i]/ckms*lines)
        nuoff = vels[i]/ckms*lines
        """
        Create the optical depth profile by looping over the
        magnetic hyperfine lines and adding Gaussian lines
        with the amplitude weights given by the optical
        depth weight array.
        """
        tauprof = np.zeros_like(model)
        for kk,nuo in enumerate(nuoff):
            tauprof += (tau11_tots[i] * tau_wts[kk] *
                        np.exp(-(freq11ax+nuo-lines[kk])**2 /
                               (2.0*nusigma[kk]**2)))
        """
        Create the model line emission using the optical depth 
        profile and the excitation temperature in the equation
        of radiative transfer.
        """
        model += (texs[i]-T_CMB)*(1-np.exp(-1.*tauprof))
    return(model)

def get_model_amp(freq11ax,pars,perrs):
    """
    Get the amplitude of the line model using the optical depth
    at line center and the excitation temperature.
    """
    tau11_0,tau11_0_err = get_tau11_0(freq11ax,pars,perrs)
    amp = (pars[2]-T_CMB)*(1-math.exp(-tau11_0))
    return(amp)

def get_tau11_0(freq11ax,pars,perrs):
    """
    Get the optical depth of the NH3(1,1) emission at line center,
    """
    
    """
    Create uncertainty objects for the fit parameters to easily 
    calculate the statistical error on the optical depth.
    """
    vel = uu.ufloat(pars[0],perrs[0])
    sigma = uu.ufloat(pars[1],perrs[1])
    tau11_tot = uu.ufloat(pars[3],perrs[3])

    """
    Calculate the optical depth weights for each of the magnetic 
    hyperfine lines.
    """
    linename = 'oneone'
    tau_wts = np.array(tau_wts_dict[linename])
    tau_wts = tau_wts / (tau_wts).sum()

    """
    Convert the velocity offsets of the main line and the 
    hyperfine lines and the velocity dispersion to units
    of frequency.
    """
    voff_lines = np.array(voff_lines_dict[linename])
    lines = (1-voff_lines/ckms)*freq_dict[linename]/1e9
    nusigma = np.abs(sigma/ckms*lines)
    nuoff = vel/ckms*lines
    
    """
    Create the optical depth profile by looping over the
    magnetic hyperfine lines and adding Gaussian lines
    with the amplitude weights given by the optical
    depth weight array.
    """
    tauprof = np.zeros(len(freq11ax))
    for kk,nuo in enumerate(nuoff):
        tauprof = tauprof + (tau11_tot * tau_wts[kk] *
                             unp.exp(-(freq11ax+nuo-lines[kk])**2/
                                     (2.0*nusigma[kk]**2)))

    #Get the channel corresponding to the line center
    freq = vel_to_freq(vel,linename)/1e9
    line_center_chan = np.argmin(abs(freq-freq11ax))
    #Calculate the optical depth and error at line center
    tau0 = tauprof[line_center_chan].nominal_value
    tau0_err = tauprof[line_center_chan].std_dev
    return(tau0,tau0_err)

def get_BIC(data,model,rms,pars):
    """
    Return the Bayesian Information Criterion (BIC), which
    provides a measure of the goodness of fit of a model.
    This method is better at comparing models with different
    numbers of parameters, since it enacts a steeper 
    penaly for having a larger number of parameters
    than the reduced chi-squared.
    """
    loglike = -1.*(0.5*np.log(2*np.pi)+np.log(rms)+
                   get_chi2(data,model,rms)/2)
    k = len(pars)
    N = len(data)
    BIC = np.log(N)*k-2*loglike
    return(BIC)

def get_chi2(data,model,rms):
    #Return the chi-squared of the best-fit model
    return(np.sum((data-model)**2/rms**2))

def get_peak_vel(snr_spec,vax,header):
    """
    Returns the velocity where emission most closely 
    matches the emission pattern of the NH3(1,1) 
    satellite lines. This value is meant to be
    used to initialize the velocity of the second 
    component for a two component fit. 
    """
    res_snr_sums = nh3_kernel_sum(snr_spec,header)
    window_width = int(1e3/header['CDELT3'])
    roll_offset = int(np.ceil((window_width-1)/2))
    vel = vax[np.argmax(res_snr_sums)+roll_offset]
    return(vel)

def nh3_kernel_sum(snr_spec,h):
    """
    Return an array that features the highest values
    where emission most closely matched the emission
    pattern of the NH3(1,1) satellite lines.
    """

    #Define the velocity and channel offsets for the satellite lines
    vsats = np.array([-19.421,-7.771,7.760,19.408])
    csats = (vsats/(h['CDELT3']/1e3)).astype(int)
    #Sum the masked snr_spec values within a 1 km/s wide rolling window.
    snr_spec_roll  = rolling_window(snr_spec,int(1e3/h['CDELT3']))
    snr_spec_roll_masked = mask_snr_spec(snr_spec_roll)
    sums = ma.sum(snr_spec_roll_masked,-1)
    sum_errs = np.sqrt(ma.count(snr_spec_roll_masked,-1))
    """
    Shift this summed array by the satellite line offsets 
    and add these arrays together.
    """
    sum_lo = np.roll(sums,csats[0])
    sum_li = np.roll(sums,csats[1])
    sum_ri = np.roll(sums,csats[2])
    sum_ro = np.roll(sums,csats[3])
    combined = sums+sum_lo+sum_li+sum_ri+sum_ro
    return(combined)

def mask_snr_spec(snr_spec,snr=1):
    """
    Mask channels in a snr spectrum (spec/rms) that are below
    the snr threshold. 
    """
    masked = ma.masked_where(np.logical_or(snr_spec<snr,
                                           np.isnan(snr_spec)),snr_spec)
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

def get_vranges_from_labels(labels,h):
    """
    Get the minimum and maximum velocities associated with each labeled
    clump. Output velocity range array, where each row provides the
    label number and the corresponding minimum and maximum velocity.
    """
    vranges = np.zeros((len(np.unique(labels)[1:]),3))
    for i,label in enumerate(np.unique(labels)[1:]):
        label_chans = np.where(labels==label)[0]
        vmin = chan_to_vel(label_chans.min(),h)
        vmax = chan_to_vel(label_chans.max(),h)
        vranges[i,:] = np.array([label,vmin,vmax])
    return(vranges)

def chan_to_vel(chan,h):
    """
    Use the header information to translate from channels to velocity.
    """
    vel = ((chan - (h['CRPIX3']-1))*h['CDELT3'] + h['CRVAL3'])*0.001
    return(vel)

def vel_to_freq(vel,linename):
    """
    Use the pyspeckit frequency dict to translate 
    from channels to velocity.
    """
    freq0 = freq_dict[linename]
    freq = (freq0)*(1-(vel/ckms))
    return(freq)

def get_vax(h):
    """
    Use the header information to calculate the velocity axis of a spectrum.
    """
    vel_min = chan_to_vel(0,h)
    vel_max = chan_to_vel(h['NAXIS3']-1,h)
    vax = np.linspace(vel_min,vel_max,h['NAXIS3'])
    return(vax)

def vax_to_freq11ax(vax):
    """
    Use pyspeckit SpectroscopicAxis object to get frequency axis
    for the NH3(1,1) spectra
    """
    freq11ax = SpectroscopicAxis(vax*u.km/u.s,
                                 velocity_convention='radio',
                                 refX=freq_dict['oneone']).as_unit(u.GHz)
    return(freq11ax.value)

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

def edit_header(h,n=2):
    """
    Edit the FITS header for an n component parameter map
    """
    h['NAXIS3'] = n
    h['CRPIX3'] = 0.0
    h['CDELT3'] = 1.0
    h['CUNIT3'] = 'Components'
    h['CRVAL3'] = 0.0
    return(h)

if __name__ == '__main__':
    main()

