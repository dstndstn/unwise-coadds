import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys
import fitsio

from astrometry.util.fits import *
from astrometry.util.plotutils import *

bands = [1,2,3,4]
MU = ['m','u']


def collect_stats():
    unwdir = 'data/unwise-comp'
    T = fits_table('allsky-atlas.fits')
    mx = 50

    allcounts = init_stats()
    update_stats(T, allcounts, unwdir, mx)
    
    for band in bands:
        for mu in MU:
            fitsio.write('counts-%s-%i.fits' % (mu, band), allcounts[(band,mu)],
                         clobber=True)

def init_stats():
    allcounts = {}
    for band in bands:
        for mu in MU:
            allcounts[(band,mu)] = np.zeros((len(T), mx+1), np.int32)
    return allcounts

def update_stats(T, allcounts, unwdir, mx, gz=True):
        
    for ti,tile in enumerate(T.coadd_id):
        for bi,band in enumerate(bands):
            for mui,mu in enumerate(MU):
                try:
                    fn = os.path.join(unwdir, tile[:3], tile,
                                      'unwise-%s-w%i-n-%s.fits' % (tile, band, mu))
                    if gz:
                        fn += '.gz'
                    print (ti+1), 'of', len(T), ':', fn
                    if not os.path.exists(fn):
                        print 'No such file'
                        continue
                    N = fitsio.read(fn)
                    allcounts[(band,mu)][ti,:] = np.bincount(np.minimum(N, mx).ravel(), minlength=mx+1)
                except:
                    import traceback
                    traceback.print_exc()
    



#collect_stats()

allcounts = {}
for band in bands:
    for mu in MU:
        fn = 'counts-%s-%i.fits' % (mu, band)
        #fn = 'counts4-%s-%i.fits' % (mu, band)
        c = fitsio.read(fn)
        allcounts[(band,mu)] = c

T = fits_table('allsky-atlas.fits')
mx = 50

if True:
    unwdir = 'data/unwise-4'
    update_stats(T, allcounts, unwdir, mx, gz=False)
    for band in bands:
        for mu in MU:
            fn = 'counts4-%s-%i.fits' % (mu, band)
            fitsio.write(fn, allcounts[(band,mu)], clobber=True)
            print 'Wrote', fn

ps = PlotSequence('nstats4')

order = dict([(b, []) for b in bands])

cc = ['b','g','r','m']
#for mu in MU:
for mu in ['u']:
    for n in range(11):
        plt.clf()
        for iband,band in enumerate(bands):
            #C = fitsio.read('counts-%s-%i.fits' % (mu,band))
            C = allcounts[(band,mu)]
            #print mu,band, C.shape

            nz = C[:,n].ravel()
            I = np.flatnonzero(nz > 0)
            order[band].append(I)
            if len(I) == 0:
                print mu,'band',band, ': no images with >0 pixels with coverage', n
                continue
            nz = nz[I]

            nbad = np.sum(nz > 1e3)
            print mu, 'band', band, ':', nbad, 'images with >1000 pixels with coverage', n

            J = np.argsort(-nz)
            print 'worst:', ' '.join(T.coadd_id[I[J[:20]]])
            print 'IDs:', band*20000 + I[J[:20]]
            
            #plt.hist(C[:,n].ravel(), 100, histtype='step', color=cc[iband])
            plt.hist(np.log10(nz), 100, range=(0,7), histtype='step', color=cc[iband])
            plt.xlabel('log10 n pixels')
            plt.ylabel('Number of images')
            
        plt.title('Number of pixels with coverage = %i, %s' % (n, mu))
        ps.savefig()


#sys.exit(0)        

if False:
    T = fits_table('allsky-atlas.fits')
    listed = np.zeros(4*20000 + len(T), bool)
    for n in range(8):
        oo = []
        for b in bands:
            o = order[b][n] + b*20000
            #print 'Band', b, ':', len(o)
            onew = o[listed[o] == False]
            oo.append(onew)
            listed[o] = True
            #oo.append(o + b*20000)
        oo = np.hstack(oo)
        fn = 'jobs-%02i.txt' % n
        f = open(fn, 'w')
        f.write('\n'.join(['%i'%i for i in oo]))
        f.write('\n')
        f.close()
        print 'qdo load unwise4 %s --priority %i' % (fn, 20-n)

for mu in ['u']:
    for n in range(21):
        plt.figure(1)
        plt.clf()
        plt.figure(2)
        plt.clf()
        for iband,band in enumerate(bands):
            #C = fitsio.read('counts-%s-%i.fits' % (mu,band))
            C = allcounts[(band,mu)]
            #print mu,band, C.shape

            nz = C[:,:n+1].sum(axis=1)
            I = np.flatnonzero(nz > 0)
            if len(I) == 0:
                print mu,'band',band, ': no images with >0 pixels with coverage <=', n
                continue
            nz = nz[I]

            #nbad = np.sum(nz > 1e3)
            #print mu, 'band', band, ':', nbad, 'images with >1000 pixels with coverage <=', n
            nbad = np.sum(nz >= 1)
            print mu, 'band', band, ':', nbad, 'images with >=1 pixels with coverage <=', n
            
            #plt.hist(C[:,n].ravel(), 100, histtype='step', color=cc[iband])
            plt.figure(1)
            plt.hist(np.log10(nz), 100, range=(0,7), histtype='step', color=cc[iband])
            plt.xlabel('log10 n pixels')
            plt.ylabel('Number of images')

            plt.figure(2)
            plt.plot(T.ra[I], T.dec[I], '.', color=cc[iband])
            
        plt.figure(1)
        plt.title('Number of pixels with coverage <= %i, %s' % (n, mu))
        ps.savefig()

        plt.figure(2)
        plt.title('Pixels with coverage <= %i, %s' % (n, mu))
        ps.savefig()


