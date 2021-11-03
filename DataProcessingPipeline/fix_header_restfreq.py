"""
fix_header_restfreq.py

Check that the rest frequency in the data cube header is
accurate. If it is not, change the rest frequency to the
value reported by Splatalogue.

Example:
python fix_header_restfreq.py -i L30_Tile01_23694_MHz_line.fits

-i : Input file   -- Input file
-o : Output file  -- Output file. If unset, overwrite input file
-h : Help         -- Display this help

"""

import astropy.io.fits as fits 
import os,sys,getopt
import numpy as np

def main():
    #Defaults
    overwrite = True
    try:
        opts,args = getopt.getopt(sys.argv[1:],"i:o:h")
    except getopt.GetoptError:
        print("Invalid arguments")
        print(__doc__)
        sys.exit(2)
    for o,a in opts:
        if o == "-i":
            input_file = a
        elif o == "-o":
            output_file = a
            overwrite = False
        elif o == "-h":
            print(__doc__)
            sys.exit(1)
        else:
            assert False, "unhandled option"
            print(__doc__)
            sys.exit(2)
            
    #If output_file is not set, overwrite input_file
    if overwrite: 
        output_file = input_file

    #Frequencies in the filenames are in MHz
    filename_freqs = ['21431', '21550', '21981', '22235', '22344', 
                      '23444', '21301', '23694', '23722', '23870', 
                      '23963', '24139', '24532']

    #Accurate rest frequencies (in KHz) come from https://splatalogue.online
    accurate_freqs = {'21431':21431931700., '21550':21550300000., 
                      '21981':21981572600., '22235':22235079800., 
                      '22344':22344030800., '23444':23444778000., 
                      '21301':21301256800., '23694':23694495500.,
                      '23722':23722633300., '23870':23870129200., 
                      '23963':23963896800., '24139':24139416300., 
                      '24532':24532988700.}

    #Find the truncated rest frequency of the line data from the filename
    filename_chunks = input_file.split("_")
    freq_chunk = np.empty(0,dtype=bool)
    for chunk in filename_chunks:
        freq_chunk = np.concatenate((freq_chunk,
                                     np.array([chunk in filename_freqs])))
    if freq_chunk.any():
        freq_ind = np.where(freq_chunk)[0]
        if freq_ind.shape[0] == 1:
            freq = filename_chunks[freq_ind[0]]
        else:
            print("More than one frequency found in the input filename.")
            sys.exit(2)
    else:
        print("Unrecognized frequency in the input filename.")
        sys.exit(2)

    #Read in data and fix rest frequency naming in header if necessary
    d,h = fits.getdata(input_file,header=True)
    if not ('RESTFREQ' in h):
        if ('RESTFRQ' in h)
            h.rename_keyword('RESTFRQ','RESTFREQ')
        else:
            h.set('RESTFREQ',0.)
    print(freq,h['RESTFREQ'])

    """
    If the rest frequency in the header is not accurate, 
    fix it and write the data.
    """
    if h['RESTFREQ'] != accurate_freqs[freq]:
        h['RESTFREQ'] = accurate_freqs[freq]
        fits.writeto(output_file,d,h,overwrite=True)
    else:
        print("Rest frequency is correct.")

if __name__ == '__main__':
    main()
