import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import numpy as np
import pylab as plt
import sys

import fitsio

from astrometry.util.plotutils import *
from astrometry.util.fits import *

ps = PlotSequence('rchi')

def rchi_sim():
    # simulations -- what is the false positive rate for the rchi outlier
    # detection method given N observations?
    
    for nobs in range(2, 16):
        rchis = []
        rchis2 = []
        rchis3 = []
        rchis4 = []
        for sami in range(10000):
            xx = np.random.normal(size=nobs)
            mn = np.mean(xx)
            imsq = np.sum(xx**2)
    
            subw = nobs - 1
            subco = (mn*nobs - xx) / subw
            subsq = (imsq - xx**2) / subw
            subpp = np.sqrt(np.maximum(0, subsq - subco**2))
    
            #subpp2 = np.sqrt(np.maximum(0, subsq - subco**2) * (nobs / float(nobs-1)))
            subpp2 = np.sqrt(np.maximum(0, subsq - subco**2) * (subw / np.maximum(1, float(subw-1))))

            priorpp = np.sqrt(1. + (0.03 * subco)**2)
            nprior = 5
            subpp3 = ((subpp * (nobs-1)) + (priorpp * nprior)) / ((nobs-1) + nprior)

            subpp4 = np.sqrt(((subpp**2 * (nobs-1)) + (priorpp**2 * nprior)) / ((nobs-1) + nprior))

            # subv = np.maximum(0, subsq - subco**2)
            # w = 1.
            # priorv = 1./w + (0.03 * np.maximum(subco, 0))**2
            # priorw = nprior * w
            # subpp5 = np.sqrt((subv * subw + priorv * priorw) / (subw + priorw))

            
            rchi = (xx - subco) / subpp
            rchis.append(rchi)
    
            rchis2.append((xx - subco) / subpp2)

            rchis3.append((xx - subco) / subpp3)

            rchis4.append((xx - subco) / subpp4)
            
        rchis = np.array(rchis).ravel()
        rchis2 = np.array(rchis2).ravel()
        rchis3 = np.array(rchis3).ravel()
        rchis4 = np.array(rchis4).ravel()
    
        plt.clf()
        plt.hist(np.clip(rchis, -6, 6), 100, range=(-6,6), histtype='step', color='b')
        plt.hist(np.clip(rchis2, -6, 6), 100, range=(-6,6), histtype='step', color='g')
        plt.hist(np.clip(rchis3, -6, 6), 100, range=(-6,6), histtype='step', color='r')
        plt.hist(np.clip(rchis4, -6, 6), 100, range=(-6,6), histtype='step', color='m')
        plt.title('N=%i: %% outliers = %.2f | %.2f | %.2f | %.2f' % (nobs,
                                                        100.*(np.sum(np.abs(rchis ) >= 5.) / float(len(rchis))),
                                                        100.*(np.sum(np.abs(rchis2) >= 5.) / float(len(rchis2))),
                                                        100.*(np.sum(np.abs(rchis3) >= 5.) / float(len(rchis3))),
                                                        100.*(np.sum(np.abs(rchis3) >= 5.) / float(len(rchis4))),
                                                        ))
        ps.savefig()
    


def error_stats(band):

    tile = '1384p454'

    unwdir = 'data/unwise-comp'
    basename = unwdir + '/%s/%s/unwise-%s-w%i' % (tile[:3], tile, tile, band)
    iv = fitsio.read(basename + '-invvar-m.fits.gz')
    pp = fitsio.read(basename + '-std-m.fits.gz')
    ff = fitsio.read(basename + '-img-m.fits')

    print 'median invvar:', np.median(iv)
    sig1 = 1./np.sqrt(np.median(iv))
    print '-> sig1', sig1
    
    plt.clf()
    loghist(np.log10(np.maximum(10, ff.ravel())),
            np.log10(np.maximum(10, pp.ravel())), 100)
    plt.xlabel('flux')
    plt.ylabel('pp std')

    ax = plt.axis()
    xx = np.linspace(ax[0], ax[1], 100)
    err = np.zeros_like(xx) + sig1
    flux = 10.**xx
    err = np.hypot(err, 0.03 * flux)

    plt.plot(xx, np.log10(err), 'b-')
    plt.axis(ax)
    
    ps.savefig()
    



#error_stats(3)
#error_stats(4)

rchi_sim()

#sys.exit(0)


tile = '1336p666'
band = 3

#unwdir = 'data/unwise-comp'
unwdir = 'xxx'
basename = unwdir + '/%s/%s/unwise-%s-w%i' % (tile[:3], tile, tile, band)
iv = fitsio.read(basename + '-invvar-m.fits.gz')
plt.clf()
plt.imshow(iv, interpolation='nearest', origin='lower')
plt.colorbar()
plt.title('round 2 invvar')
ps.savefig()

nm2 = fitsio.read(basename + '-n-m.fits.gz')

plt.clf()
plt.imshow(nm2, interpolation='nearest', origin='lower')
plt.colorbar()
plt.title('round 2 n exposures')
ps.savefig()

T = fits_table(basename + '-frames.fits')
meanw = np.mean(T.weight[T.weight > 0])
print 'Mean weight:', meanw

print 'use', np.sum(T.use)
print 'incl', np.sum(T.included)

plt.clf()
plt.plot(T.npixoverlap, T.npixrchi, 'b.')
plt.xlabel('pixels overlapping')
plt.ylabel('pixels with bad rchi')
#nmx = max(T.npixoverlap)
ax = plt.axis()
nmx = ax[1]
plt.plot([0,nmx], [0, nmx*0.01], 'r-')
plt.axis(ax)
ps.savefig()



base1 = '%s-w%i-' % (tile, band)
ims = []
for fn in ['coimg1', 'cow1', 'coppstd1']:
    im = fitsio.read(base1 + fn + '.fits')
    ims.append(im)
    plo,phi = [np.percentile(im, p) for p in 25,99.8]

    args = dict(vmin=plo, vmax=phi, interpolation='nearest', origin='lower')
    if fn == 'coimg1':
        ima = args
    plt.clf()
    plt.imshow(im, **args)
    plt.colorbar()
    plt.title(fn)
    ps.savefig()

mean1, cow1, std1 = ims

plt.clf()
plt.imshow(cow1 / meanw, interpolation='nearest', origin='lower', vmin=0)
plt.colorbar()
plt.title('cow1 / meanw')
ps.savefig()

cube = fitsio.FITS(base1 + 'cube1.fits')[0]
print 'Cube', cube.get_info()['dims']
N,H,W = cube.get_info()['dims']

n1 = np.round(cow1 / meanw).astype(int)
print 'n1', n1.min(), n1.max(), n1.mean()
#lowpix = (n1 == n1.min())
lowpix = (n1 == 3)

U = np.flatnonzero(T.use > 0)
print 'Weights of used fields:', T.weight[U]

lo,hi = -10000,10000
xx = np.linspace(lo, hi, 500)
for yi,xi in zip(*np.unravel_index(np.flatnonzero(lowpix), n1.shape))[:50]:
    pix = cube[:, yi:yi+1, xi:xi+1]

    pix = pix.ravel()
    I = np.flatnonzero(pix != 0)
    
    #pix = pix[pix != 0]
    std = std1[yi,xi]
    mn  = mean1[yi,xi]
    ww  = cow1[yi,xi]

    print 'pix', pix[I]
    print 'ww %g, mn %g, std %g' % (ww, mn, std)

    dy = 0.2

    plt.clf()
    plt.plot(pix[I], np.zeros_like(pix[I]), 'bo')
    plt.plot(xx, dy*np.exp(-(xx - mn)**2 / std**2), 'r-')
    #plt.axvline(mn - 5.*std, color='r')
    #plt.axvline(mn + 5.*std, color='r')
    plt.plot(np.array([[mn-5.*std]*2, [mn+5.*std]*2]).T,
             np.array([[0, dy]]*2).T, 'r-')

    # coppstd = sqrt( coimgsq - coimg**2 )
    # coimgsq = coppstd**2 + coimg**2
    imsq = std**2 + mn**2

    k = 0
    for j in I:
        k += 1

        plt.axhline(dy*k, color='k', alpha=0.5)
        plt.plot(pix[I], np.zeros_like(pix[I]) + dy*k, 'bo')
        plt.plot(pix[j], dy*k, 'ro')
        pj = pix[j]

        wj = T.weight[U[j]]
        subw  = ww - wj
        subco = ((mn * ww) - pj * wj) / subw
        subsq = ((imsq * ww) - (pj**2 * wj)) / subw

        subpp = np.sqrt(np.maximum(0, subsq - subco**2))

        subpp2 = np.sqrt( np.maximum(0, subsq - subco**2) * (subw / (subw - meanw)) )

        # print 'wj', wj
        # print 'subw', subw
        # print 'subco', subco
        # print 'subpp', subpp
        print 'wj', wj, 'subw', subw, 'submn', subco, 'subsq', subsq, 'subpp', subpp

        plt.plot(xx, dy*k + dy*np.exp(-(xx - subco)**2 / subpp**2), 'r-')
        plt.plot(xx, dy*k + dy*np.exp(-(xx - subco)**2 / subpp2**2), 'm-')

        plt.plot(np.array([[subco-5.*subpp]*2, [subco+5.*subpp]*2]).T,
                 np.array([[dy*k, dy*(k+1)]]*2).T, 'r-')

        plt.plot(np.array([[subco-5.*subpp2]*2, [subco+5.*subpp2]*2]).T,
                 np.array([[dy*k, dy*(k+1)]]*2).T, 'm-')

    plt.title('pixel %i,%i: mean %g, std %g, N round 2: %i' % (xi,yi, mn, std, nm2[yi,xi]))
    plt.xlim(lo, hi)
    ps.savefig()

sys.exit(0)

for i in range(N):
    plane = cube[i:i+1,:,:]
    #print 'plane', plane.shape
    plane = plane[0,:,:]
    plt.clf()
    plt.imshow(plane, **ima)
    plt.colorbar()
    ps.savefig()
    
    
    


