#! /usr/bin/env python
import sys

import numpy as np

from astrometry.util.fits import *
from astrometry.libkd.spherematch import *

def main():
    import optparse
    parser = optparse.OptionParser('%prog [options] <ra> <dec>')

    opt,args = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        return -1

    ra = float(args[0])
    dec = float(args[1])

    from unwise_coadd import get_atlas_tiles

    T = get_atlas_tiles(ra, ra, dec, dec)
    print 'Found', len(T), 'atlas tiles touching RA,Dec', ra, dec
    print T.coadd_id

    return 0

if __name__ == '__main__':
    sys.exit(main())
        
