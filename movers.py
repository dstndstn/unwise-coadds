import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.measurements import label, find_objects

import fitsio

from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.util.run_command import *
from astrometry.util.starutil_numpy import *
from astrometry.util.ttime import *
from astrometry.libkd.spherematch import *
from astrometry.util.stages import *

import logging
lvl = logging.INFO
logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

from unwise_coadd import *

# GLOBALS:
# Location of WISE Level 1b inputs
wisedir = 'wise-frames'
mask_gz = True
unc_gz = True

def stage0(ps=None, **kwargs):
    #ra, dec = 0., 0.
    mjd = 55370

    #coadd_id = '0000p000'
    coadd_id = '0015p000'
    band = 1
    nsigma_co = 3.

    basefn = 'data/unwise/%s/%s/unwise-%s-w%i-' % (coadd_id[:3], coadd_id, coadd_id, band)

    T = fits_table('%sframes.fits' % basefn)
    print 'Read', len(T), 'frames'
    T.cut((T.use != 0))
    print 'Cut to', len(T), 'on "use"'
    T.cut((T.included != 0))
    print 'Cut to', len(T), 'on "included"'
    T.cut(np.abs(T.mjd - mjd) < 10)
    print 'Cut to', len(T), 'on MJD'
    print 'MJD range', T.mjd.min(), T.mjd.max()
    mjd = np.mean(T.mjd)
    print 'MJD span', T.mjd.max() - T.mjd.min()

    cofn = '%simg-m.fits' % basefn
    coadd = fitsio.read(cofn)
    coiv  = fitsio.read('%sinvvar-m.fits' % basefn)
    cowcs = Tan(cofn)
    sig1 = 1./np.sqrt(np.median(coiv.ravel()))
    print 'Sig1', sig1

    xx = np.linspace(-5, 5, 500)
    plt.clf()
    n,b,p = plt.hist(coadd.ravel() / sig1, 100, range=(-5., 5.))
    plt.plot(xx, np.exp(-0.5*xx**2) * n.max(), 'r-')
    plt.title('coadd')
    ps.savefig()

    fwhm = 6. / 2.75
    psfsig = fwhm / 2.35
    print 'PSF sigma:', psfsig

    # ????!!!!
    psfnorm = np.sqrt(1./(2. * np.sqrt(np.pi) * psfsig))
    print 'PSF norm', psfnorm

    detmap = gaussian_filter(coadd, psfsig, mode='constant')
    detmap /= psfnorm**2
    detnoise = sig1 / psfnorm

    plt.clf()
    n,b,p = plt.hist(detmap.ravel() / detnoise, 100, range=(-5., 5.))
    plt.plot(xx, np.exp(-0.5*xx**2) * n.max(), 'r-')
    plt.title('detmap')
    ps.savefig()

    thresh = nsigma_co * detnoise

    hot = (detmap > thresh)
    print 'Pixels above threshold:', sum(hot)
    hot = binary_dilation(hot, iterations=3)
    print 'Grown pixels above threshold:', sum(hot)

    plt.clf()
    plt.imshow(coadd, interpolation='nearest', origin='lower',
               vmin=-2.*sig1, vmax=3.*sig1)
    plt.colorbar()
    ps.savefig()

    plt.clf()
    plt.imshow(coadd * np.logical_not(hot), interpolation='nearest', origin='lower',
               vmin=-2.*sig1, vmax=3.*sig1)
    plt.colorbar()
    ps.savefig()

    h,w = hot.shape

    # W = fits_table('data/sdss-phot-temp/wise-sources-0000p000.fits')
    # print len(W), 'WISE sources'
    # ok,x,y = cowcs.radec2pixelxy(W.ra, W.dec)
    # W.x = x - 1.
    # W.y = y - 1.
    # print 'Brightest:', W.w1mpro[np.argsort(W.w1mpro)[:20]]
    # W.cut(W.w1mpro < 9.)
    # flux = 10.**((W.w1mpro - 22.5)/-2.5)
    # mrad = np.sqrt(flux) * 0.02
    # for x,y,r in zip(W.x.astype(int), W.y.astype(int), mrad):
    #     hot[max(0, y-r) : min(h, y+r), max(0, x-r) : min(w, x+r)] = 1
    # 
    # plt.clf()
    # plt.imshow(coadd * np.logical_not(hot), interpolation='nearest', origin='lower',
    #            vmin=-2.*sig1, vmax=3.*sig1)
    # plt.colorbar()
    # ps.savefig()

    return dict(T=T, detmap=detmap, detnoise=detnoise,
                psfsig=psfsig, coadd=coadd, hot=hot, sig1=sig1, psfnorm=psfnorm,
                band=band, basefn=basefn, coadd_id=coadd_id, cowcs=cowcs)

def stage1(ps=None, T=None, detmap=None, detnoise=None,
           band=None, coadd_id=None, basefn=None, psfnorm=None, cowcs=None,
           psfsig=None, coadd=None, hot=None, sig1=None, **kwargs):

    T.pix = np.array([None] * len(T))
    T.ox = np.array([None] * len(T))
    T.oy = np.array([None] * len(T))
    T.sig1 = np.zeros(len(T))
    for i,wise in enumerate(T):
        intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, band)
        print 'intfn:', intfn
        maskfn = intfn.replace('-int-', '-msk-')
        if mask_gz:
            maskfn = maskfn + '.gz'
        print 'maskfn', maskfn

        fn = os.path.basename(intfn).replace('-int', '')
        comaskfn = os.path.join('%smask' % basefn,
                              'unwise-mask-' + coadd_id + '-' + fn + '.gz')
        print 'comaskfn', comaskfn

        if not (os.path.exists(intfn) and os.path.exists(maskfn)):
            print 'file not found; skipping'
            continue
        
        #print 'Reading...'
        img = fitsio.read(intfn)
        mask = fitsio.read(maskfn)
        comask = fitsio.read(comaskfn)

        #print 'Filtering...'
        img -= wise.sky1
        img[mask > 0] = 0.
        img[comask > 0] = 0.
        zpscale = 1. / zeropointToScale(wise.zeropoint)
        img *= zpscale

        detmapi = gaussian_filter(img, psfsig, mode='constant')
        detmapi /= psfnorm**2
        #print 'Detmap range', img.min(), img.max()

        #print 'Resampling...'
        wcs = Tan(intfn)
        try:
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(cowcs, wcs, [], None)
        except OverlapError:
            print 'No overlap; skipping'
            continue
        T.pix[i] = detmapi[Yi,Xi]
        T.ox [i] = Xo
        T.oy [i] = Yo
        T.sig1[i] = np.sqrt(1./wise.weight)
        print 'Saved', len(T.pix[i]), 'pixels'

    return dict(T=T)

def stage2(ps=None, T=None, detmap=None, detnoise=None, band=None,
           psfsig=None, coadd=None, hot=None, sig1=None, psfnorm=None, **kwargs):

    mjd = np.mean(T.mjd)
    xx = np.linspace(-5, 5, 500)

    for vx,vy in [ (2,0), (1,1), (0,2), (-1,1), (-2,0), (-1,-1), (0,-2), (1,-1),
                   (4,0), (3,3), (0,4), (-3,3), (-4,0), (-3,-3), (-4,0), (3,-3) ]:
        H,W = coadd.shape
        shadd = np.zeros((H,W), np.float32)
        shw   = np.zeros((H,W), np.float32)
    
        for i,wise in enumerate(T):
            if wise.pix is None:
                continue
            dx = int(np.round((wise.mjd - mjd) * vx))
            dy = int(np.round((wise.mjd - mjd) * vy))
            print 'dMJD', wise.mjd - mjd, '-> dx,dy', dx,dy
            ox = wise.ox + dx
            oy = wise.oy + dy
            I = (ox > 0) * (ox < W) * (oy > 0) * (oy < H)
            oy = oy[I]
            ox = ox[I]
            shadd[oy,ox] += wise.pix[I] * wise.weight
            shw  [oy,ox] += wise.weight
    
        shadd /= np.maximum(shw, 1e-12)
        shsig1 = 1./np.sqrt(np.median(shw))
    
        # plt.clf()
        # plt.imshow(shadd, interpolation='nearest', origin='lower', vmin=0)
        # plt.colorbar()
        # ps.savefig()
    
        cold = np.logical_not(hot)
    
        # plt.clf()
        # plt.imshow(shadd * cold, interpolation='nearest', origin='lower', vmin=0)
        # plt.colorbar()
        # ps.savefig()
    
        #thresh2 = 4. * detnoise
        #plt.clf()
        #plt.imshow(shadd > thresh2, interpolation='nearest', origin='lower', vmin=0)
        #plt.colorbar()
        #ps.savefig()
    
        # plt.clf()
        # plt.hist(coadd.ravel() / sig1, 100, range=(-5., 5.))
        # plt.title('coadd')
        # ps.savefig()
    
        # lo,hi = np.percentile(shadd, 5), np.percentile(shadd, 95)
        # 
        # plt.clf()
        # plt.hist(shadd.ravel(), 100, range=(lo,hi))
        # plt.title('shadd')
        # ps.savefig()
    
        print 'shsig1', shsig1
        shnoise = shsig1 / psfnorm
        print 'shnoise', shnoise
    
        plt.clf()
        n,b,p = plt.hist(shadd.ravel() / shnoise, 100, range=(-5,5))
        plt.plot(xx, np.exp(-0.5*xx**2) * n.max(), 'r-')
        plt.title('shadd')
        ps.savefig()
    
        shsig = shadd * cold / shnoise
        cosig = coadd * cold / sig1
        cosig2 = coadd / sig1
        detsig = detmap * cold / detnoise
    
    
        det = (shsig > 4.5)
    	det = binary_dilation(det, structure=np.ones((3,3)), iterations=2)
    	blobs,nblobs = label(det, np.ones((3,3), int))
        blobslices = find_objects(blobs)
        print 'found', nblobs, 'blobs'
        if nblobs == 0:
            continue
    
    	# Find maximum pixel within each blob.
    	BX,BY = [],[]
    	for b,slc in enumerate(blobslices):
    		sy,sx = slc
    		y0,y1 = sy.start, sy.stop
    		x0,x1 = sx.start, sx.stop
    		bl = blobs[slc]
    		# find highest-S/N pixel within this blob
    		i = np.argmax((bl == (b+1)) * shsig[slc])
    		iy,ix = np.unravel_index(i, dims=bl.shape)
    		by = iy + y0
    		bx = ix + x0
    		BX.append(bx)
    		BY.append(by)
    	BX = np.array(BX)
    	BY = np.array(BY)
        bval = shsig[BY,BX]
        I = np.argsort(-bval)

        iy,ix = BY[I],BX[I]
        bval = bval[I]
        margin = 5
        I = np.flatnonzero((iy > margin) * (ix > margin) * (iy < (H-margin)) * (ix < (W-margin)))
        print len(I), 'pass cuts'
        if len(I) == 0:
            continue
        I = I[:10]

        iy,ix = iy[I], ix[I]
        bval = bval[I]

        plt.clf()
        plt.imshow(shadd * cold / shnoise, interpolation='nearest', origin='lower',
                   vmin=-1, vmax=5, cmap='gray')
        plt.colorbar()
        ax = plt.axis()
        plt.plot(ix, iy, 'o', mec='r', mfc='none')
        for i,(x,y,v) in enumerate(zip(ix,iy,bval)):
            plt.text(x, y, '%i' % (i+1), color='r', ha='center', va='bottom')
        plt.axis(ax)
        plt.title('Shift-n-add: vx,vy = %.1f,%.1f' % (vx,vy))
        ps.savefig()
        
    
        for x,y in zip(ix,iy):
            S = 25
            slc = slice(max(0, y-S), min(H, y+S)), slice(max(0, x-S), min(W, x+S))
            plt.clf()
            plt.subplot(2,2,1)
            plt.imshow(shsig[slc], interpolation='nearest', origin='lower',
                       vmin=-1, vmax=5)
            plt.title('shifted')
            plt.subplot(2,2,2)
            plt.imshow(detsig[slc], interpolation='nearest', origin='lower',
                       vmin=-1, vmax=5)
            plt.title('stationary')
            plt.subplot(2,2,3)
            plt.imshow(cosig[slc], interpolation='nearest', origin='lower',
                       vmin=-1, vmax=5)
            plt.title('coadd')
            plt.subplot(2,2,4)

            plt.imshow(cosig2[slc], interpolation='nearest', origin='lower',
                       vmin=-1, vmax=5)
            plt.title('coadd')
            plt.suptitle('S/N %.1f' % shsig[y,x])
            ps.savefig()
    
            # Plot individual exposures in range.
            stamps = []
            for j,wise in enumerate(T):
                if wise.pix is None:
                    continue
                x0,x1,y0,y1 = wise.coextent
                if x < x0 or x > x1 or y < y0 or y > y1:
                    continue
                S = 10
                stamp = np.zeros((S*2+1, S*2+1), np.float32)
                J = (wise.ox >= x-S) * (wise.ox <= x+S) * (wise.oy >= y-S) * (wise.oy <= y+S)
                stamp[wise.oy[J] - (y-S), wise.ox[J] - (x-S)] = wise.pix[J]
                if stamp[S,S] != 0:
                    stamps.append(stamp / wise.sig1)
    
            N = len(stamps)
            cols = int(np.ceil(np.sqrt(N)))
            rows = int(np.ceil(N / float(cols)))
            plt.clf()
            for i,stamp in enumerate(stamps):
                plt.subplot(rows, cols, i+1)
                plt.imshow(stamp, interpolation='nearest', origin='lower',
                           vmin=-1, vmax=5)
            ps.savefig()




class Duck(object):
    pass

def main():
    opt = Duck()
    opt.ps = 'movers'
    opt.picklepat = 'movers-s%02i.pickle'
    opt.force = []
    opt.write = True

    stage = 2

    class MyCaller(CallGlobal):
        def getkwargs(self, stage, **kwargs):
            kwa = self.kwargs.copy()
            kwa.update(kwargs)
            if opt.ps is not None:
                kwa.update(ps = PlotSequence(opt.ps + '-s%i' % stage, format='%03i'))
            return kwa

    prereqs = {}
    #    100: None,
    #    204: 103,

    runner = MyCaller('stage%i', globals(), opt=opt)

    R = runstage(stage, opt.picklepat, runner, force=opt.force, prereqs=prereqs,
                 write=opt.write)
    return R

if __name__ == '__main__':
    main()
    
