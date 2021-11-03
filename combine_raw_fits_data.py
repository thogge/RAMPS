"""
combine_raw_fits_data.py

Script to combine the raw RAMPS data cubes to create a larger map. 
Regrids maps to a common grid if desired. Other optional arguments 
include spatially and/or spectrally cropping the input data cubes 
prior to combination, weighting overlapping regions using the rms
noise maps, and fitting the 
Uses MIRIAD version 4.3.8 to regrid and combine the data cubes. 

This code also utilizes the squeeze_cube.py, auto_crop_cube.py,
crop_cube_spectra.py, prefit_raw_data_baselines.py, 
and make_rms_map.py scripts for data processing prior to combination.

Example:
python combine_raw_fits_data.py -i map1.fits,map2.fits,map3.fits
                            -o big_map.fits -r template.fits -cpd

-i : Input     -- Comma-separated list of input files
-o : Output    -- Output file
-r : Regrid    -- Regrid template file
-n : Numcores  -- Number of cores for parallel processing
-c : Crop      -- Spatially crop nan values from input cubes
-s : Spec Crop -- Spectrally crop edge channels from input cubes
-p : Prefit    -- Fit cubes with 0th order baseline prior to combination
-d : Delete    -- Delete intermediate files
-h : Help      -- Flag to display this help 
"""

import sys,os,getopt
import numpy as np
try:
    import astropy.io.fits as fits
except:
    import fits
import pdb

base_dir = "/projectnb2/jjgroup/thogge/ramps/"
scripts_dir = base_dir+"scripts/python/"

def main():
    regrid = False
    weight = False
    crop = False
    spec_crop = False
    prefit = False
    delete = False
    overwrite = False
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:o:r:n:wcsvpdeh")
    except getopt.GetoptError:
        print("Invalid arguments")
        print(__doc__)
        sys.exit(2)
    for o,a in opts:
        if o == "-i":
            input_files = a
        elif o == "-o":
            output_file = a
        elif o == "-r":
            regrid = True
            template_file = a
        elif o == "-n":
            numcores = int(a)
        elif o == "-w":
            weight = True
        elif o == "-c":
            crop = True
        elif o == "-s":
            spec_crop = True
        elif o == "-p":
            prefit = True
        elif o == "-d":
            delete = True
        elif o == "-e":
            overwrite = True
        elif o == "-h":
            print(__doc__)
            sys.exit(1)
        else:
            assert False, "unhandled option"
            print(__doc__)
            sys.exit(2)

    #Define input file array, initialize file strings
    input_files_arr = input_files.replace(" ","").split(",")
    num_files = len(input_files_arr)
    miriad_infile_arr = []
    miriad_output_file = output_file[:-4]+"cm"
    miriad_infile_string = ""
    weights = ""

    #Create MIRIAD template file
    if regrid and template_file not in input_files_arr:
        miriad_template_file = template_file[:-4]+"cm"
        if os.path.exists(miriad_template_file) and overwrite:
            print("rm -r "+miriad_template_file)
            os.system("rm -r "+miriad_template_file)
        h = fits.getheader(template_file)
        if h['NAXIS'] > 3:
            execute_string = "python "+scripts_dir+\
                             "squeeze_cube.py -i "+template_file+\
                             " -o "+template_file
            print(execute_string)
            os.system(execute_string)
        execute_string = "fits in="+template_file+\
                         " out="+miriad_template_file+" op=xyin"
        print(execute_string)
        os.system(execute_string)
    elif regrid:
        miriad_template_file = template_file[:-4]+"cm"

    #Loop over input files and convert them from FITS to MIRIAD format
    for i in range(num_files):
        f = input_files_arr[i]
        tile_name = f[:-20]
        miriad_infile_arr.append(f[:-4]+"cm")
        
        #Delete MIRIAD file if it exists
        if os.path.exists(miriad_infile_arr[i]) and overwrite:
            print("rm -r "+miriad_infile_arr[i])
            os.system("rm -r "+miriad_infile_arr[i])
        else:
            pass
        
        #Remove extra dimensions in input data cubes
        h = fits.getheader(f)
        if h['NAXIS'] > 3:
            execute_string = "python "+scripts_dir+\
                             "squeeze_cube.py -i "+f+" -o "+f
            print(execute_string)
            os.system(execute_string)

        #Crop data cubes
        if crop and spec_crop:
            crop_file = f[:-5]+"_crop.fits"
            execute_string = "python "+scripts_dir+\
                             "auto_crop_cube.py -i "+f+" -o "+crop_file+\
                             " -s 500 -e 15884"
            print(execute_string)
            os.system(execute_string)
            f = crop_file
        elif crop:
            crop_file = f[:-5]+"_crop.fits"
            execute_string = "python "+scripts_dir+\
                             "auto_crop_cube.py -i "+f+" -o "+crop_file
            print(execute_string)
            os.system(execute_string)
            f = crop_file
        elif spec_crop:
            crop_file = f[:-5]+"_crop.fits"
            execute_string = "python "+scripts_dir+\
                             "crop_cube_spectra.py -i "+f+" -o "+crop_file+\
                             " -s 500 -e 15884"
            print(execute_string)
            os.system(execute_string)
            f = crop_file
        
        #Perform initial zeroth order baseline fit 
        if prefit:
            f_fit = f[:-5]+"_prefit.fits"
            execute_string = "python "+scripts_dir+\
                             "prefit_raw_data_baselines.py -i "+f+\
                             " -o "+f_fit+" -f -n "+str(numcores)
            print(execute_string)
            os.system(execute_string)
            f = f_fit

        #Calculate weights from the rms maps
        if weight:
            # Can weight by the rms noise in the overlap regions
            rms_file = f[:-5]+"_rms.fits"
            if not os.path.isfile(rms_file):
                execute_string = "python "+scripts_dir+\
                                 "make_rms_map.py -i "+f+" -o "+rms_file+\
                                 " -n "+str(numcores)
                print(execute_string)
                os.system(execute_string)
            d = fits.getdata(rms_file)
            if i == num_files - 1:
                weights += str(np.nanmedian(d))
            else: 
                weights += str(np.nanmedian(d))+","

        #Convert the input files from FITS to MIRIAD format
        if not os.path.exists(miriad_infile_arr[i]):
            execute_string = "fits in="+f+\
                             " out="+miriad_infile_arr[i]+" op=xyin"
            print(execute_string)
            os.system(execute_string)
            
        #Regrid the input files to the template file grid
        if regrid:
            regrid_file = str(miriad_infile_arr[i])[:-3]+"_regrid.cm"
            if os.path.exists(regrid_file) and delete:
                print("rm -r "+regrid_file)
                os.system("rm -r "+regrid_file)
            execute_string = "regrid in="+miriad_infile_arr[i]+\
                             " out="+regrid_file+\
                             " tin="+miriad_template_file+" options=offset"
            print(execute_string)
            os.system(execute_string)
            miriad_infile_string += regrid_file+","
        else:
            miriad_infile_string += miriad_infile_arr[i]+","

    #Delete the MIRIAD output file if it exists
    if os.path.exists(miriad_output_file) and overwrite:
        print("rm -r "+miriad_output_file)
        os.system("rm -r "+miriad_output_file)

    #Combine the MIRIAD input files
    miriad_infile_string = miriad_infile_string[:-1]
    if weight:
        execute_string = "imcomb in="+miriad_infile_string+\
                         " out="+miriad_output_file+" rms="+weights
    else:
        execute_string = "imcomb in="+miriad_infile_string+\
                         " out="+miriad_output_file
    print(execute_string)
    os.system(execute_string)

    #Convert the combined data from MIRIAD to FITS format
    execute_string = "fits in="+miriad_output_file+\
                     " out="+output_file+" op=xyout"
    print(execute_string)
    os.system(execute_string)

    #Delete the intermediate files
    if delete:
        for files in miriad_infile_arr:
            execute_string = "rm -r " + files
            print(execute_string)
            os.system(execute_string)
            if regrid:
                execute_string = "rm -r " + files[:-3]+"_regrid.cm"
                print(execute_string)
                os.system(execute_string)                
        execute_string = "rm -r " + miriad_output_file
        print(execute_string)
        os.system(execute_string)
        if regrid:
            execute_string = "rm -r " + miriad_template_file
            print(execute_string)
            os.system(execute_string) 
        if crop or spec_crop:
            for files in input_files_arr:
                execute_string = "rm " + files[:-5]+"_crop.fits"
                print(execute_string)
                os.system(execute_string) 
        if prefit:
            execute_string = "rm *_prefit.fits"
            print(execute_string)
            os.system(execute_string)

if __name__ == '__main__':
    main()

