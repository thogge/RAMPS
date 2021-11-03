import sys,os,getopt
import numpy as np
import astropy.io.fits as fits
import pdb

overwrite = False
numcores = 28
try:
    opts,args = getopt.getopt(sys.argv[1:],"n:oh")
except getopt.GetoptError,err:
    print(str(err))
    print(__doc__)
    sys.exit(2)
for o,a in opts:
    if o == "-o":
        overwrite = True
    elif o == "-n":
        numcores = int(a)
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

for field in fields:
    filebase = "L"+str(field).replace(".","_").replace("_0","")
    label3D_11_file = filebase+"_NH3_1-1_clump_labels_3D.fits"
    tex_file = filebase + "_NH3_1-1_hf_tex.fits"
    if os.path.isfile(filebase+"_NH3_1-1_fixed_c.fits"):
        file11 = filebase+"_NH3_1-1_fixed_c.fits"
    else:
        file11 = filebase+"_NH3_1-1_fixed.fits"
    if overwrite or (not os.path.isfile(tex_file)) and os.path.isfile(label3D_11_file):
        executestring = "python " + scripts_dir + "fit_NH3_11_hf_clumps.py -i " + file11 + " -o " + filebase + "_NH3_1-1 -l " + label3D_11_file + " -r " + filebase + "_NH3_1-1_rms.fits" + " -n " + str(numcores)
        print(executestring)
        os.system(executestring)

    tkin_file = filebase + "_tkin.fits"
    label2D_22_file = filebase+"_NH3_2-2_clump_labels_2D.fits"
    if overwrite or (not os.path.isfile(tkin_file)) and os.path.isfile(label2D_22_file):
        executestring = "python " + scripts_dir + "pyspeckit_scripts/test_pyspeckit/fiteach_RAMPS_clumps.py -f " + filebase + " -n " + str(numcores)
        print(executestring)
        os.system(executestring)
    #"""


