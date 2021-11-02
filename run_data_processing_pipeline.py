"""
run_data_processing_pipeline.py

Pipeline to combine and baseline subtract data cubes for all lines 
observed by the RAMPS project. Tiles (individual observations) are
combined to form fields (1 degree wide regions). Fields are 
baseline subtracted to prepare the data cubes for analysis.
This pipeline utilizes the combine_raw_fits_data.py, 
fix_header_restfreq.py, crop_cube_spectra.py, 
output_baselined_data.py, and make_rms_map.py scripts.

Example:
python run_data_processing_pipeline.py -n 28 -o

-n : Numcores   -- Number of cores used for parallel processing
-o : Overwrite  -- Flag to overwrite previously fit data
-h : Help       -- Display this help

"""

import sys,os,getopt
import numpy as np
import pdb

def check_files_exist(base_dir,field,trans,file_suffixes):
    """
    Parameters
    ----------
    base_dir : str
        The directory that will be searched for the files.
    field : str
        The associated field of the files.
    trans : str
        The name of the transition defined by freq_to_trans_dict.
    file_suffixes : list
        A list of the suffixes of the files.

    Returns
    -------
    all_files_exist : bool
        If all of the files that were searched for exist, then True,
        if not, then False.
    """
    file_exists = []
    for suff in file_suffixes:
        file_exists.append(os.path.exists(base_dir+"L"+field+"_"+
                                          trans+"_"+suff+".fits")
    all_files_exist = file_exists.all()
    return(all_files_exist)

numcores = 1
overwrite = False
try:
    opts,args = getopt.getopt(sys.argv[1:],"n:oh")
except getopt.GetoptError:
    print("Invalid key")
    print(__doc__)
    sys.exit(2)
for o,a in opts:
    if o == "-n":
        numcores = int(a)
    elif o == "-o":
        overwrite = True
    elif o == "-h":
        print(__doc__)
        sys.exit(1)
    else:
        assert False, "unhandled option"
        print(__doc__)
        sys.exit(2)

#Define directory structure
rootdir = "/projectnb2/jjgroup/thogge/ramps/"
cubedir = rootdir+"cubes/"
template_files_dir = cubedir+"/L23/field/tiles/"
scripts_dir = rootdir+"scripts/python/"

#List to record fields that have already been processed
completed_fields = []

"""
Define all of the line frequencies observed by RAMPS. Create dict to
connect the frequencies to the transition names. Define the minimum
and maximum velocities used for spectral cropping. Define the 
smoothing factors used in output_baselined_data.py
"""
freq_list = ["23694","23722","23870","24139","24532","22235","21301",
             "21431","21550","21981","22344","23444","23963"]
freq_to_trans_dict = {"23694":"NH3_1-1","23722":"NH3_2-2","23870":"NH3_3-3",
                      "24139":"NH3_4-4","24532":"NH3_5-5","22235":"H2O",
                      "21301":"HC5N_8-7","21431":"HC7N_19-18",
                      "21550":"CH3OH_12-11","21981":"HNCO_1-0",
                      "22344":"CCS_2-1","23444":"CH3OH_10-9",
                      "23963":"HC5N_9-8"}
vmins = np.array(["-30","-30","-30","-30","-30","-140","-30","-30",
                  "-140","-30","-30","-140","-30"])
vmaxs = np.array(["140","140","140","140","140","140","140","140",
                  "140","140","140","140","140"])
downsamples = np.array(["11","11","11","11","11","7","11",
                        "11","7","11","11","7","11"])

#Define the integer-spaced fields observed by RAMPS
fields = np.concatenate((np.arange(32)+10,np.array([43,45,47]))).astype(str)

for field in fields:
    #Define the subdirectory structure
    field_basedir = cubedir+ "L" + field
    final_dir = field_basedir + "/field/"
    combine_dir = field_basedir + "/field/tiles/"
    
    #Check whether the data in this field has been processed
    if "L"+field in completed_fields:
        continue
    else: 
        files_exist = []
        for freq in freq_list:
            if (check_files_exist(combine_dir,field,freq_to_trans_dict[freq],
                                  ["cube","mom0","mom1","rms"]) 
                or check_files_exist(final_dir,field,freq_to_trans_dict[freq],
                                     ["cube","mom0","mom1","rms"])):
                files_exist.append(True)
            else:
                files_exist.append(False)
        files_exist = np.asarray(files_exist)
        if files_exist.all():
            completed_fields.append("L"+field)
            continue
        
    """
    Move all of the raw data from the tile directories to the 
    combine directory for concatenation.
    """
    print(field_basedir)
    for subdir in os.listdir(field_basedir):
        tiledir = field_basedir + "/" + subdir
        if os.path.isfile(tiledir) or "field" in subdir:
            continue
        else:
            print(tiledir)
            all_files = os.listdir(tiledir)
            file_list = [s for s in all_files if "MHz_line.fits" in s]
            for files in file_list:
                fitsfile = tiledir + "/" + files
                if os.path.isdir(fitsfile):
                    print("There is an extra directory here called " + 
                          files + " . Skipping.")
                else:
                    if os.path.exists(combine_dir+files):
                        execute_string = "mv "+fitsfile+" "+\
                                         combine_dir+files[:-5]+"2.fits"
                    else:
                        execute_string = "mv "+fitsfile+" "+combine_dir+files
                    print(execute_string)
                    os.system(execute_string)
    """
    Combine all of the raw tile data for each transition into a single 
    field. Spectrally crop the resulting data cubes, subtract baselines,
    create moment maps, and create rms noise maps.
    """
    os.chdir(combine_dir)
    all_files = os.listdir(combine_dir)
    for i,freq in enumerate(freq_list):
        #Get the tile files corresponding to individual observations
        freq_file_list = [s for s in all_files if freq+"_MHz_line.fits" in s]
        combine_input_string = ""
        for ff in freq_file_list:
            if ff == freq_file_list[-1]:
                combine_input_string += ff
            else:
                combine_input_string += ff+","
        combine_file = "L"+field+"_"+freq+"_MHz_line.fits"
        regrid_file = "regrid_"+freq+"_template_map.fits"
        
        #Check that the template file for regridding is present
        if not os.path.exists(regrid_file):
            execute_string = "cp "+template_files_dir+regrid_file+" ."
            print(execute_string)
            os.system(execute_string)
            
        #Combine the tile files to create a field file
        if not os.path.exists(combine_file):
            execute_string = "python "+scripts_dir+\
                             "combine_raw_fits_data.py -i "+\
                             combine_input_string+" -o "+combine_file+\
                             " -r "+regrid_file+" -n "+str(numcores)+" -csd"
            print(execute_string)
            os.system(execute_string)

        #Check the rest frequency in the header and change if necessary
        execute_string = "python "+scripts_dir+\
                         "fix_header_restfreq.py -i "+combine_file
        print(execute_string)
        os.system(execute_string)

        #Spectrally crop out the velocities where there is no signal
        crop_file = combine_file[:-5]+"_crop.fits"
        if not os.path.exists(crop_file):
            execute_string = "python "+scripts_dir+\
                             "crop_cube_spectra.py -i "+combine_file+\
                             " -o "+crop_file+" -s "+vmaxs[i]+\
                             " -e "+vmins[i]+" -v"
            print(execute_string)
            os.system(execute_string)

        #Baseline subtract the field data cube and perform moment analysis 
        if not check_files_exist(final_dir,field,freq_to_trans_dict[freq],
                                 ["cube","mom0","mom1"]):
            execute_string = "python "+scripts_dir+\
                             "output_baselined_data.py -i "+crop_file+\
                             " -o L"+field+"_"+freq_to_trans_dict[freq]+\
                             " -s "+downsamples[i]+\
                             " -w 20 -f01 -p 3 -n "+str(numcores)
            print(execute_string)
            os.system(execute_string)

        #Make the rms noise map
        cube_file = "L"+field+"_"+freq_to_trans_dict[freq]+"_cube.fits"
        rms_file = "L"+field+"_"+freq_to_trans_dict[freq]+"_rms.fits"
        if not os.path.exists(rms_file):
            execute_string = "python "+scripts_dir+\
                             "make_rms_map.py -i "+cube_file+\
                             " -o "+rms_file+" -n "+str(numcores)
            print(execute_string)
            os.system(execute_string)

    #If all of the files were created successfully, add to completed fields
    files_exist = []
    for freq in freq_list:
        if (check_files_exist(combine_dir,field,freq_to_trans_dict[freq],
                              ["cube","mom0","mom1","rms"]) 
            or check_files_exist(final_dir,field,freq_to_trans_dict[freq],
                                 ["cube","mom0","mom1","rms"])):
            files_exist.append(True)
        else:
            files_exist.append(False)
    files_exist = np.asarray(files_exist)
    if files_exist.all():
        completed_fields.append("L"+field)
