import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys

import fitsio

from scipy.ndimage.filters import gaussian_filter

from astrometry.util.plotutils import *
from astrometry.util.util import *
from astrometry.util.resample import *

from unwise_coadd import estimate_sky_2

def main():
    import optparse
    parser = optparse.OptionParser('%prog <unWISE input> <WISE input> <Output>')
    parser.add_option('--plots', dest='plots', action='store_true',
                      help='Write plots?')
    opt,args = parser.parse_args()
    if len(args) != 3:
        parser.print_help()
        return -1

    ufn = args[0]
    wfn = args[1]
    outfn = args[2]

    if opt.plots:
        ps = PlotSequence('fix')
    else:
        ps = None
    print 'Reading', ufn
    unwise,uhdr = fitsio.read(ufn, header=True)
    uwcs = Tan(ufn)
    print 'unwise', unwise.shape, unwise.dtype
    print 'Reading', wfn
    wise,whdr = fitsio.read(wfn, header=True)
    print 'wise', wise.shape, wise.dtype
    wcs = anwcs(wfn)

    assert(wise.shape == (4095,4095))
    assert(unwise.shape == (2048,2048))

    zp = whdr.get('MAGZP')
    uzp = uhdr.get('MAGZP')
    scale = 10.**((zp - uzp) / -2.5)
    wise *= scale

    print 'binning...'
    # we keep the odd last pixel, so the resampling fully covers the unwise img
    wH,wW = wise.shape
    whalf = np.zeros((wH/2 + 1, wW/2 + 1), np.float32)
    Heven,Weven = 2*(wH/2),2*(wW/2)
    H,W = whalf.shape
    whalf[:Heven/2,:Weven/2] = (wise[ :Heven:2,  :Weven:2] +
                                wise[1:Heven:2,  :Weven:2] +
                                wise[ :Heven:2, 1:Weven:2] +
                                wise[1:Heven:2, 1:Weven:2]) / 4.
    # last row & col
    whalf[-1,:Heven/2] = (wise[-1, :Weven:2] + wise[-1, 1:Weven:2]) / 2.
    whalf[:Weven/2,-1] = (wise[:Heven:2, -1] + wise[1:Heven:2, -1]) / 2.
    whalf[-1,-1] = wise[-1,-1]
    # photocal adjustment from binning
    whalf *= 4.

    # subtract constant background level from WISE
    print 'estimating sky...'
    wbg = estimate_sky_2(whalf)
    whalf -= wbg

    # resample WISE image to unWISE image coords -- SIN vs TAN projection
    print 'resampling...'
    uh,uw = unwise.shape
    xx,yy = np.meshgrid(np.arange(uw), np.arange(uh))
    rr,dd = uwcs.pixelxy2radec(xx+1, yy+1)
    ok,xx,yy = wcs.radec2pixelxy(rr, dd)
    xx = np.clip(np.round((xx - 1.) / 2.).astype(int), 0, W-1)
    yy = np.clip(np.round((yy - 1.) / 2.).astype(int), 0, H-1)
    wre = whalf[yy,xx]

    print 'median smoothing...'
    wsmooth = np.zeros_like(wre)
    median_smooth(wre, None, 50, wsmooth)

    unew = unwise + wsmooth

    uhdr.add_record(dict(name='UNWFIXBG', value=True,
                         comment='Background patched by fix-background.py'))
    fitsio.write(outfn, unew, header=uhdr, clobber=True)
    print 'Wrote', outfn
    
    if ps:
        lo,hi = [np.percentile(whalf, p) for p in [1, 99]]
        ima = dict(interpolation='nearest', origin='lower',
                   cmap='gray', vmin=lo, vmax=hi)
    
        plt.clf()
        plt.imshow(unwise, **ima)
        plt.colorbar()
        plt.title('unWISE')
        ps.savefig()    

        plt.clf()
        plt.imshow(wre, **ima)
        plt.colorbar()
        plt.title('WISE')
        ps.savefig()
        
        plt.clf()
        plt.imshow(wsmooth, **ima)
        plt.colorbar()
        plt.title('Median-smoothed WISE')
        ps.savefig()    

        plt.clf()
        plt.imshow(unew, **ima)
        plt.colorbar()
        plt.title('unWISE + median-smoothed WISE')
        ps.savefig()    

        x0,x1, y0,y1 = 600,800, 1000,1200
        slc = (slice(y0,y1), slice(x0,x1))
        ima.update(extent=(x0,x1,y0,y1))
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(wre[slc], **ima)
        plt.title('WISE')
        plt.subplot(1,2,2)
        plt.imshow(unew[slc], **ima)
        plt.title('unWISE + median(WISE)')
        ps.savefig()    
        
if __name__ == '__main__':
    sys.exit(main())
    
