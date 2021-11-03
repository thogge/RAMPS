import sys,os,getopt
import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS
import pdb

def c2v(channel,h):
    velocity = ((channel - h['CRPIX3'])*h['CDELT3'] + h['CRVAL3'])/1000.
    return velocity

def v2c(velocity,h):
    channel = (velocity*1000. - h['CRVAL3'])/h['CDELT3'] + h['CRPIX3']
    return int(round(channel))

def c2f(channel,h):
    frequency = ((channel - h['CRPIX3'])*h['CDELT3'] + h['CRVAL3'])
    return frequency

def f2c(frequency,h):
    channel = (frequency - h['CRVAL3'])/h['CDELT3'] + h['CRPIX3']
    return int(round(channel))

def f2v(frequency,h):
    if 'RESTFREQ' in h:
        f0 = h['RESTFREQ']
    elif 'RESTFRQ' in h:
        f0 = h['RESTFRQ']
    velocity = 3e5*(1-(frequency/f0))
    return velocity

def v2f(velocity,h):
    if 'RESTFREQ' in h:
        f0 = h['RESTFREQ']
    elif 'RESTFRQ' in h:
        f0 = h['RESTFRQ']
    frequency = (f0)*(1-(velocity/3e5))
    return frequency

def get_coord_grids(h):
    XX,YY = np.meshgrid(np.arange(h['NAXIS1']),np.arange(h['NAXIS2']))
    w = WCS(h)
    LL,BB = w.all_pix2world(XX,YY,0)
    return LL,BB

def px2coord(x,y,h):
    w = WCS(h)
    l,b = w.all_pix2world(x,y,0)
    return(l,b)

def coord2px(l,b,h):
    w = WCS(h)
    x,y = w.all_world2pix(l,b,0)
    return(int(round(x)),int(round(y)))


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

base_dir = "/projectnb2/jjgroup/thogge/ramps/"
files_dir = os.getcwd()
scripts_dir = base_dir+"scripts/python/clumpfind_scripts/"

all_files = os.listdir(files_dir)
trans = ['1-1','2-2']

for t in trans:
    file_list = [s for s in all_files if "NH3_"+t+"_fixed" in s]
    #if len(file_list)==0:
        #file_list = [s for s in all_files if "NH3_"+t+"_fixed.fits" in s]

    for files in file_list:
        if "_c.fits" in files:
            filebase = files[:-13]
        else:
            filebase = files[:-11]
        mask_file = filebase+"_clump_mask.fits"
        label_cube_file = filebase+"_clump_labels_3D.fits"
        label_plane_file = filebase+"_clump_labels_2D.fits"
        #if overwrite or (not os.path.isfile(mask_file)) or (not os.path.isfile(label_cube_file)) or (not os.path.isfile(label_plane_file)):
        if overwrite or (not os.path.isfile(label_cube_file)) or (not os.path.isfile(label_plane_file)):
            if t=='1-1':
                executestring = "python "+scripts_dir+"satellite_clumpfind.py -i "+files+" -r "+filebase+"_rms.fits"+" -o "+filebase+" -n "+str(numcores)+" -t "+t[::2]+" -s 100" 
            elif t=='2-2' and "L47" not in filebase :
                executestring = "python "+scripts_dir+"satellite_clumpfind.py -i "+files+" -r "+filebase+"_rms.fits"+" -o "+filebase+" -n "+str(numcores)+" -t "+t[::2]+" -s 50" 
            print(executestring)
            os.system(executestring)
            
    #pdb.set_trace()

