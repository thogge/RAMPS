"""
make_all_border_cubes.py

Make field data cubes that are centered at half-integer Galactic 
longitudes. The purpose of this step is to allow the 
ammonia_clumpfind.py script to accurately find molecular clumps 
that overlap the borders of the field data cubes that are 
centered at integer Galactic longitudes. 

This module utilizes the combine_raw_fits_data.py and 
crop_cube.py scripts.

Example:
python make_all_border_cubes.py

-o : Overwrite  -- Flag to overwrite previously processed data
-h : Help       -- Display this help
"""


import sys,os,getopt
import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS

base_dir = "/projectnb/jjgroup/thogge/ramps/"
multifield_dir = base_dir+"cubes/multi-field/"
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

"""
Change directory to the multifield directory where the processed data
are stored.
"""
os.chdir(multifield_dir)

"""
Loop over the NH3(1,1) and NH3(2,2) transition data and the fields centered
at integer Galactic longitudes between 10 degrees and 41 degrees. Create 
1 degree wide fields centered at half-integer Galactic longitudes between 
10.5 degrees and 40.5 degrees.
"""
trans_list = ["1-1","2-2"]
nfiles = 32
fields = (np.arange(nfiles)+10).astype(str)
for trans in trans_list:
    #Define the template file for regridding
    template_file = "L"+fields[0]+"_NH3_"+trans+"_cube.fits"
    for i in np.arange(nfiles-1):
        """
        Define the filenames for the fields to be combined, the
        combined field, and the field centered at half-integer 
        Galactic longitude
        """
        field_file1 = "L"+fields[i]+"_NH3_"+trans+"_cube.fits"
        field_file2 = "L"+fields[i+1]+"_NH3_"+trans+"_cube.fits"
        combined_file = "L"+fields[i]+"-"+fields[i+1]+\
                        "_NH3_"+trans+"_cube.fits"
        crop_file = "L"+fields[i]+"_5_NH3_"+trans+"_cube.fits"
        
        #Combine pairs of neighboring fields
        if not os.path.exists(combined_file) or overwrite:
            execute_string = "python "+scripts_dir+\
                             "combine_raw_fits_data.py -i "+field_file1+","+\
                             field_file2+" -o "+combined_file+\
                             " -r "+template_file +" -d"
            print(execute_string)
            os.system(execute_string)
        """    
        Crop the two-field data cubes to create a single 1 degree wide
        field centered at half-integer Galactic longitudes
        """
        if os.path.exists(combined_file) and (not os.path.exists(crop_file) 
                                              or overwrite):
            os.system("rm "+crop_file)
            execute_string = "python "+scripts_dir+\
                             "crop_cube.py -i "+combined_file+\
                             " -o "+crop_file+
                             " -x "+fields[i]+".5 -c 0.5 -w"
            print(execute_string)
            os.system(execute_string)
        
        #Remove the two-field data cubes if the crop was successful
        if os.path.exists(crop_file) and os.path.exists(combined_file):
            os.system("rm "+combined_file)
