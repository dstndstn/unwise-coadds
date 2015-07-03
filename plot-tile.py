import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os

from astrometry.util.fits import *
from astrometry.util.plotutils import *

from unwise_coadd import *

if __name__ == '__main__':

    tile = '1336p666'
    T = fits_table('unwise-%s-w3-frames.fits' % tile)

    ps = PlotSequence('tile')

    # approx
    ra,dec = tile_to_radec(tile)
    wcs = get_coadd_tile_wcs(ra, dec)
    W,H = wcs.imagew, wcs.imageh
    
    rr,dd = wcs.pixelxy2radec(np.array([1,W,W,1,1]),
                              np.array([1,1,H,H,1]))

    print 'included:', np.sum(T.included)

    T.use = (T.use == 1)
    T.included = (T.included == 1)

    print 'included:', len(np.flatnonzero(T.included))
    
    plt.clf()
    plt.plot(T.ra, T.dec, 'r.')
    I = (T.qual_frame == 0)
    p1 = plt.plot(T.ra[I], T.dec[I], 'mx')
    p2 = plt.plot(T.ra[T.use], T.dec[T.use], 'b.')

    I = (T.npixrchi > (T.npixoverlap * 0.01))
    p3 = plt.plot(T.ra[I], T.dec[I], 'r+')

    I = T.included
    p4 = plt.plot(T.ra[I], T.dec[I], 'go')
    plt.plot(rr, dd, 'k-')

    plt.figlegend((p[0] for p in (p1,p2,p3,p4)),
                  ('Bad qual_frame', 'Used', 'Bad rchi', 'Included'),
        loc='upper right')

    ps.savefig()
    
