#! /usr/bin/env python

import fitsio
import optparse
import cPickle as pickle
import time
import numpy as np
import sys
from astrometry.util.util import Sip

# test performance of Cori scratch when many
# processes are simultaneously reading
# L1b files in rapid succession

def read_files(indstart, nproc, delay=None):
    flist = pickle.load( open( "/global/cscratch1/sd/ameisner/code/unwise-coadds/w1_l1b_filenames.pkl", "rb" ) )
    flist = [f.replace('/project/projectdirs/cosmo/data/wise/allsky','/global/cscratch1/sd/ameisner') for f in flist]

    # assume for now that indstart and nproc are such that there won't be any attempt to 
    # access invalid indices within flist

    times = np.zeros(nproc)
    for i in range(indstart, indstart+nproc):
        fname = flist[i]
        t0 = time.time()
        image = fitsio.read(fname)
        dt = time.time() - t0
        print i, '  ', dt, '    ', fname
        times[i-indstart] = dt
        if delay is not None:
            time.sleep(float(delay))

    outname = 'times_scratch_' + str(indstart).zfill(5) + '.fits'
    fitsio.write(outname, times)


def main():
    parser = optparse.OptionParser('%prog [options]')

    parser.add_option('--indstart', dest='indstart', default=0, type=int,
                      help='starting index in list of L1b file names')
    parser.add_option('--nproc', dest='nproc', default=1000, type=int,
                      help='number of L1b files to read')
    parser.add_option('--delay', dest='delay', default=None, 
                      help='add delay (seconds) between reads')
    opt,args = parser.parse_args()
    read_files(opt.indstart,opt.nproc, delay=opt.delay)

if __name__ == '__main__':
    sys.exit(main())
