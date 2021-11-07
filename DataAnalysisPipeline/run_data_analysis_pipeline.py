"""
run_data_analysis_pipeline.py

Pipeline to find clumps and analyze their NH3(1,1) and (2,2)
emission to determine their gas properties. Must be run after
processing the data using run_data_processing_pipeline.py.

This pipeline utilizes the ammonia_clumpfind.py,
fit_NH3_11_hf_clumps.py, and fiteach_RAMPS_clumps.py scripts.


Example:
python run_data_analysis_pipeline.py -n 28 -o

-n : Numcores   -- Number of cores used for parallel processing
-o : Overwrite  -- Flag to overwrite previously fit data
-h : Help       -- Display this help

"""

import sys,os,getopt
import numpy as np
import astropy.io.fits as fits
import pdb

def main():

    #Defaults
    numcores = 1
    overwrite = False
    try:
        opts,args = getopt.getopt(sys.argv[1:],"n:oh")
    except getopt.GetoptError as err:
        print(err.msg)
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

    files_dir = os.getcwd()
    scripts_dir = "/projectnb/jjgroup/thogge/ramps/scripts/python/"
    
    fields = np.concatenate((np.arange(63)/2.+10,np.array([43,45,47])))
    trans = ["1-1","2-2"]
        
    for field in fields:
        fieldbase = "L"+str(field).replace(".","_").replace("_0","")

        for t in trans:
            filebase = fieldbase+"NH3_"+t
            label_3D_file = filebase+"_clump_labels_3D.fits"
            if (not os.path.isfile(label_3D_file)) or overwrite:
                if t=="1-1":
                    executestring = "python "+scripts_dir+\
                                    "ammonia_clumpfind.py -i "+files+\
                                    " -r "+filebase+"_rms.fits"+\
                                    " -o "+filebase+" -n "+str(numcores)+\
                                    " -t "+t[::2]+" -s 100" 
                elif t=="2-2" and "L47" not in filebase:
                    executestring = "python "+scripts_dir+\
                                    "ammonia_clumpfind.py -i "+files+\
                                    " -r "+filebase+"_rms.fits"+\
                                    " -o "+filebase+" -n "+str(numcores)+\
                                    " -t "+t[::2]+" -s 50" 
                print(executestring)
                os.system(executestring)

        
        label_3D_11_file = fieldbase+"_NH3_1-1_clump_labels_3D.fits"
        vel11_file = fieldbase+"_NH3_1-1_hf_vel.fits"
        sigma11_file = fieldbase+"_NH3_1-1_hf_sigma.fits"
        file11 = fieldbase+"_NH3_1-1_cube.fits"
        fit11_files = [vel11_file,sigma11_file]
        if (os.path.isfile(label_3D_11_file) and 
            (not check_files_exist(fit11_files) or overwrite)):
            executestring = "python "+scripts_dir+\
                            "fit_NH3_11_hf_clumps.py -i "+file11+\
                            " -o "+fieldbase+"_NH3_1-1 -l "+label_3D_11_file+\
                            " -r "+fieldbase+"_NH3_1-1_rms.fits"+\
                            " -n "+str(numcores)
            print(executestring)
            os.system(executestring)

        tkin_file = fieldbase+"_tkin.fits"
        label_3D_22_file = fieldbase+"_NH3_2-2_clump_labels_3D.fits"
        label_files = [label_3D_11_file,label_3D_22_file]
        if (check_files_exist(fit11_files+label_files) and 
            (not os.path.isfile(tkin_file) or overwrite)):
            executestring = "python "+scripts_dir+\
                            "pyspeckit_scripts/fiteach_RAMPS_clumps.py -f "+\
                            fieldbase+" -n "+str(numcores)
            print(executestring)
            os.system(executestring)

def check_files_exist(file_list):
    """
    Check that necessary files exist before moving on to the 
    next step of the pipeline

    Parameters
    ----------
    file_list : list
        A list of the files to search for.

    Returns
    -------
    all_files_exist : bool
        If all of the files that were searched for exist, then True,
        if not, then False.
    """
    file_exists = []
    for files in file_list:
        file_exists.append(os.path.exists(files))
    all_files_exist = file_exists.all()
    return(all_files_exist)

if __name__ == '__main__':
    main()



