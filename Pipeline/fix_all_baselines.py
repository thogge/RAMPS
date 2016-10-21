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
                    if "line.fits" in files:
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
                        print fixedfile
                        if overwrite:
                            if os.path.exists(fixedfile):
                                print 'Fixing cube and overwriting old cube'
                            else:
                                print 'Fixing cube'
                            if 'NH3' in fixedfile:
                                croppedfitsfile = fitsfile[:-5] + "_cropped.fits"
                                executestring = "python " + rootdir + "scripts/python/crop_cube_spectra.py -i " + fitsfile + " -o " + croppedfitsfile + " -s 0 -e 13884"
                                print(executestring)
                                os.system(executestring)
                                executestring = "python " + rootdir + "scripts/python/fix_ramps_parallel_NH3.py -i " + croppedfitsfile + " -o " + fixedfile + " -fv01"
                                print(executestring)
                                os.system(executestring)
                                finalfile = fixedfile[:-5] + "_final.fits"
                                executestring = "python " + rootdir + "scripts/python/auto_crop_cube.py -i " + fixedfile + " -o " + finalfile
                                print(executestring)
                                os.system(executestring)
                                noisefile = fixedfile[:-11] + "_noise.fits"
                                executestring = "python " + rootdir + "scripts/python/make_noise_map.py -i " + finalfile + " -o " + noisefile
                                print(executestring)
                                os.system(executestring)
                            elif 'H2O' in fixedfile:
                                executestring = "python " + rootdir + "scripts/python/fix_ramps_parallel_H2O.py -i " + fitsfile + " -o " + fixedfile + " -fv01"	
                                print(executestring)
                                os.system(executestring)
       	       	       	       	finalfile = fixedfile[:-5] + "_final.fits"
                                executestring = "python " + rootdir + "scripts/python/auto_crop_cube.py -i " + fixedfile + " -o " + finalfile
                                print(executestring)
                                os.system(executestring)
                                noisefile = fixedfile[:-11] + "_noise.fits"
                                executestring = "python " + rootdir + "scripts/python/make_noise_map.py -i " + finalfile + " -o " + noisefile
                                print(executestring)
                                os.system(executestring)
                        else:
                            if os.path.exists(fixedfile):
                                print "Cube already fixed."
                            else:
                                print "Not yet fixed! Doing fitting now."
                                if 'NH3' in fixedfile:
                                    croppedfitsfile = fitsfile[:-5] + "_cropped.fits"
                                    executestring = "python " + rootdir + "scripts/python/crop_cube_spectra.py -i " + fitsfile + " -o " + croppedfitsfile + " -s 0 -e 13884"
                                    print(executestring)
                                    os.system(executestring)
                                    executestring = "python " + rootdir + "scripts/python/fix_ramps_parallel_NH3.py -i " + croppedfitsfile + " -o " + fixedfile + " -fv01"
                                    print(executestring)
                                    os.system(executestring)
                                    finalfile = fixedfile[:-5] + "_final.fits"
                                    executestring = "python " + rootdir + "scripts/python/auto_crop_cube.py -i " + fixedfile + " -o " + finalfile
                                    print(executestring)
                                    os.system(executestring)
                                    noisefile = fixedfile[:-11] + "_noise.fits"
                                    executestring = "python " + rootdir + "scripts/python/make_noise_map.py -i " + finalfile + " -o " + noisefile
                                    print(executestring)
                                    os.system(executestring)
                                elif 'H2O' in fixedfile:
                                    executestring = "python " + rootdir + "scripts/python/fix_ramps_parallel_H2O.py -i " + fitsfile + " -o " + fixedfile + " -fv01"
                                    print(executestring)
                                    os.system(executestring)
                                    finalfile = fixedfile[:-5] + "_final.fits"
                                    executestring = "python " + rootdir + "scripts/python/auto_crop_cube.py -i " + fixedfile + " -o " + finalfile
                                    print(executestring)
                                    os.system(executestring)
                                    noisefile = fixedfile[:-11] + "_noise.fits"
                                    executestring = "python " + rootdir + "scripts/python/make_noise_map.py -i " + finalfile + " -o " + noisefile
                                    print(executestring)
                                    os.system(executestring)
