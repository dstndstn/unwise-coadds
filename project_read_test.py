#! /usr/bin/env python

import fitsio
import optparse
import cPickle as pickle
import time
import numpy as np
import sys
from astrometry.util.util import Sip

def main():
    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('--scratch', dest='scratch', default=False, action='store_true',
                      help='read files from scratch?')
    parser.add_option('--sip', dest='only_sip', default=False, action='store_true',
                      help='only extract astrometry?')
    parser.add_option('--nmax', dest='nmax', default=None, type=int,
                      help='maximum number of files to access')

    opt,args = parser.parse_args()

    # load in the list of file names from pkl file
    flist = pickle.load( open( "/global/cscratch1/sd/ameisner/code/unwise-coadds/w1_l1b_filenames.pkl", "rb" ) )

    nmax = (opt.nmax if (opt.nmax is not None) else len(flist))

    if opt.scratch:
        flist = [f.replace('/project/projectdirs/cosmo/data/wise/allsky','/global/cscratch1/sd/ameisner') for f in flist]

    nfile = len(flist)
    times = np.zeros(nmax)
    for i in range(nmax):
        fname = flist[i]
        if not opt.only_sip:
            t0 = time.time()
            image = fitsio.read(fname)
            dt = time.time() - t0
        else:
            t0 = time.time()
            wcs = Sip(fname)
            dt = time.time()-t0

        print i, '  ', dt, '    ', fname
        times[i] = dt

    outname = 'times_' + ('scratch' if opt.scratch else 'project') + '_' + ('wcs' if opt.only_sip else 'image') + '.fits'
    fitsio.write(outname, times)

if __name__ == '__main__':
    sys.exit(main())
