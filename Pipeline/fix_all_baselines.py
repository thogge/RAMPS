"""
fix_all_baselines.py

Script to run fix_ramps_parallel_NH3.py and fix_ramps_parallel_H2O.py
on all RAMPS data cubes that have not been baseline subtracted. 
Additionally, there is an option to fit all cubes, even those that have 
been fit previously. This option will overwrite the previously fit cubes. 

Example:
python fix_all_baselines.py -o

-o : Overwrite  -- Flag to overwrite previously fit data
-h : Help       -- Display this help

"""

import sys,os,getopt
import pdb

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

rootdir = "/net/scc-df4/data/ramps/"
rootcube = "/net/scc-df4/data/ramps/cubes"

for directory in os.listdir(rootcube):
    fielddir = rootcube+ "/" + directory
    if os.path.isfile(fielddir):
        continue
    print fielddir
    for subdir in os.listdir(fielddir):
        tiledir = fielddir + "/" + subdir
        if os.path.isfile(tiledir):
            continue
        else:
            print tiledir
            for files in os.listdir(tiledir):
                fitsfile = tiledir + "/" + files
                if os.path.isdir(fitsfile):
                    print "There is an extra directory here called " + files + " . Skipping"
                else:
                    if ("line.fits" in files) or ("cube.fits" in files):
                        print fitsfile
                        #this is pretty hardcored for now. better search algorithm would be smart
                        if "23694" in files:
                            fixedfile = fitsfile[:-19]+"NH3_1-1_fixed.fits"
                        elif "23722" in files:
                            fixedfile = fitsfile[:-19]+"NH3_2-2_fixed.fits"
                        elif "23870" in files:
                            fixedfile = fitsfile[:-19]+"NH3_3-3_fixed.fits"
                        elif "24139" in files:
                            fixedfile = fitsfile[:-19]+"NH3_4-4_fixed.fits"
                        elif "24533" in files:
                            fixedfile = fitsfile[:-19]+"NH3_5-5_fixed.fits"
                        elif "22235" in files:
                            fixedfile = fitsfile[:-19]+"H2O_fixed.fits"
                        elif "23445" in files:
                            fixedfile = fitsfile[:-19]+"CH3OH_1_fixed.fits"
                        elif "21550" in files:
                            fixedfile = fitsfile[:-19]+"CH3OH_2_fixed.fits"
                        croppedfitsfile = fitsfile[:-5] + "_cropped.fits"
                        print fixedfile
                        finalfile = fixedfile[:-11] + "_final.fits"
                        print finalfile
                        noisefile = fixedfile[:-11] + "_noise.fits"
                        if (os.path.exists(finalfile) == False) or overwrite:
                            print 'Fixing cube'
                            if 'NH3' in fixedfile:
                                end_channel = '13884'
                                smooth = '11'
                            elif 'H2O' in fixedfile:
                                end_channel = '15564'
                                smooth = '6'
                            executestring = "python " + rootdir + "scripts/python/crop_cube_spectra.py -i " + fitsfile + " -o " + croppedfitsfile + " -s 820 -e "+end_channel
                            print(executestring)
                            os.system(executestring)
                            executestring = "python " + rootdir + "scripts/python/fix_ramps_parallel.py -i " + croppedfitsfile + " -o " + fixedfile + " -s " + smooth + " -fv01"
                            print(executestring)
                            os.system(executestring)
                            executestring = "rm " + croppedfitsfile
                            print(executestring)
                            os.system(executestring)
                            executestring = "python " + rootdir + "scripts/python/auto_crop_cube.py -i " + fixedfile + " -o " + finalfile
                            print(executestring)
                            os.system(executestring)
                            executestring = "python " + rootdir + "scripts/python/make_noise_map.py -i " + finalfile + " -o " + noisefile
                            print(executestring)
                            os.system(executestring)
                        else:
                            print "Skipping this cube, it was already fixed."
