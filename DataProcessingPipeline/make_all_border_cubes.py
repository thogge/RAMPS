import sys,os,getopt
import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS

base_dir = "/projectnb/jjgroup/thogge/ramps/"
files_dir = base_dir+"cubes/multi-field/"
scripts_dir = base_dir+"scripts/python/"

overwrite = False
try:
    opts,args = getopt.getopt(sys.argv[1:],":oh")
except getopt.GetoptError,err:
    print(str(err))
    print(__doc__)
    sys.exit(2)
for o,a in opts:
    if o == "-o":
        overwrite = True
    elif o == "-h":
        print(__doc__)
        sys.exit(1)
    else:
        assert False, "unhandled option"
        print(__doc__)
        sys.exit(2)

trans_list = ["1-1","2-2"]
nfiles = 32
fields = (np.arange(nfiles)+10).astype(str)
for trans in trans_list:
    template_file = "L"+fields[0]+"_NH3_"+trans+"_cube.fits"
    for i in np.arange(nfiles-1):
        field_file1 = "L"+fields[i]+"_NH3_"+trans+"_cube.fits"
        field_file2 = "L"+fields[i+1]+"_NH3_"+trans+"_cube.fits"
        combined_file = "L"+fields[i]+"-"+fields[i+1]+\
                        "_NH3_"+trans+"_cube.fits"
        if not os.path.exists(combined_file) or overwrite:
            execute_string = "python "+scripts_dir+\
                             "combine_fits_data.py -i "+field_file1+","+\
                             field_file2+" -o "+combined_file+\
                             " -r "+template_file +" -d"
            print(execute_string)
            os.system(execute_string)
        crop_file = "L"+fields[i]+"_5_NH3_"+trans+"_cube.fits"
        if os.path.exists(combined_file) and (not os.path.exists(crop_file) 
                                              or overwrite):
            os.system("rm "+crop_file)
            execute_string = "python "+scripts_dir+\
                             "crop_cube.py -i "+combined_file+\
                             " -o "+crop_file+
                             " -x "+fields[i]+".5 -c 0.5 -w"
            print(execute_string)
            os.system(execute_string)
        if os.path.exists(crop_file) and os.path.exists(combined_file):
            os.system("rm "+combined_file)
