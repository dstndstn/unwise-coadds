import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', serif='computer modern roman')
matplotlib.rc('font', **{'sans-serif': 'computer modern sans serif'})
import matplotlib.cm
from matplotlib.ticker import FixedFormatter
import matplotlib.patches as patches
import numpy as np
import pylab as plt
import os
import sys

import fitsio

from astrometry.util.stages import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.fits import *
from astrometry.util.util import *

#wisel3 = 'wise-L3'
wisel3 = 'allwise-L3'
wise_tag = 'ac51'
coadds = 'wise-coadds'

from unwise_coadd import estimate_sky, estimate_sky_2, get_l1b_file
from tractor import GaussianMixturePSF, NanoMaggies

def read_wise_coadd(coadd_id, band, unc=False, cov=False, wget=True, get_imgfn=False,
                    basedir=None):
    if basedir is None:
        basedir = wisel3
    dir1 = os.path.join(basedir, coadd_id[:2], coadd_id[:4],
                        coadd_id + '_' + wise_tag)
    fn = '%s_%s-w%i-int-3.fits' % (coadd_id, wise_tag, band)
    rtn = read(dir1, fn, header=True, wget=wget)
    if get_imgfn:
        rtn = rtn + (os.path.join(dir1, fn),)
    if unc:
        uncim = read(dir1, '%s_%s-w%i-unc-3.fits.gz' % (coadd_id, wise_tag, band), wget=wget)
        rtn = rtn + (uncim,)
    if cov:
        covim = read(dir1, '%s_%s-w%i-cov-3.fits.gz' % (coadd_id, wise_tag, band), wget=wget)
        rtn = rtn + (covim,)
    return rtn

def read(dirnm, fn, header=False, wget=False):
    pth = os.path.join(dirnm, fn)
    print pth
    if wget and not os.path.exists(pth):
        coadd_id = os.path.basename(os.path.dirname(pth)).replace('_'+wise_tag,'')
        print 'coadd_id', coadd_id
        if '-3.fits' in pth:
            if not os.path.exists(dirnm):
                os.makedirs(dirnm)
            cmd = 'wget -nv -O %s http://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd/%s/%s/%s/%s' % (pth, coadd_id[:2], coadd_id[:4],
                                                                                                          coadd_id + '_' + wise_tag, fn)
            print cmd
            os.system(cmd)

    data = fitsio.read(pth, header=header)
    return data


#def get_wise_dir(coadd_id, band):
#    dir1 = os.path.join(wisel3, coadd_id[:2], coadd_id[:4], coadd_id + '_ac51')
#    return dir1

def plot_exposures():

    plt.subplots_adjust(bottom=0.01, top=0.9, left=0., right=1., wspace=0.05, hspace=0.2)
    for coadd_id,band in [('1384p454', 3)]:
        print coadd_id, band
    
        plt.clf()
        plt.subplot(1,2,1)
        fn2 = os.path.join(coadds, 'unwise-%s-w%i-img-w.fits' % (coadd_id, band))
        J = fitsio.read(fn2)
        binJ = reduce(np.add, [J[i/4::4, i%4::4] for i in range(16)])
        plo,phi = [np.percentile(binJ, p) for p in [25,99]]
        plt.imshow(binJ, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.xticks([]); plt.yticks([])
        plt.subplot(1,2,2)
        #fn3 = os.path.join(coadds, 'unwise-%s-w%i-invvar-w.fits' % (coadd_id, band))
        fn3 = os.path.join(coadds, 'unwise-%s-w%i-ppstd.fits' % (coadd_id, band))
        J = fitsio.read(fn3)
        binJ = reduce(np.add, [J[i/4::4, i%4::4] for i in range(16)])
        phi = np.percentile(binJ, 99)
        plt.imshow(binJ, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=0, vmax=phi)
        plt.xticks([]); plt.yticks([])
        ps.savefig()
    
    
        fn = os.path.join(coadds, 'unwise-%s-w%i-frames.fits' % (coadd_id, band))
        T = fits_table(fn)
        print len(T), 'frames'
        T.cut(np.lexsort((T.frame_num, T.scan_id)))
    
        plt.clf()
        n,b,p = plt.hist(np.log10(np.maximum(0.1, T.npixrchi)), bins=100, range=(-1,6),
                         log=True)
        plt.xlabel('log10( N pix with bad rchi )')
        plt.ylabel('Number of images')
        plt.ylim(0.1, np.max(n) + 5)
        ps.savefig()
    
        J = np.argsort(-T.npixrchi)
        print 'Largest npixrchi:'
        for n,s,f in zip(T.npixrchi[J], T.scan_id[J], T.frame_num[J[:20]]):
            print '  n', n, 'scan', s, 'frame', f
    
        i0 = 0
        while i0 <= len(T):
            plt.clf()
            R,C = 4,6
            for i in range(i0, i0+(R*C)):
                if i >= len(T):
                    break
                t = T[i]
                fn = get_l1b_file('wise-frames', t.scan_id, t.frame_num, band)
                print fn
                I = fitsio.read(fn)
                bad = np.flatnonzero(np.logical_not(np.isfinite(I)))
                I.flat[bad] = 0.
                print I.shape
                binI = reduce(np.add, [I[j/4::4, j%4::4] for j in range(16)])
                print binI.shape
                plt.subplot(R,C,i-i0+1)
                plo,phi = [np.percentile(binI, p) for p in [25,99]]
                print 'p', plo,phi
                plt.imshow(binI, interpolation='nearest', origin='lower',
                           vmin=plo, vmax=phi, cmap='gray')
                plt.xticks([]); plt.yticks([])
                plt.title('%s %i' % (t.scan_id, t.frame_num))
            plt.suptitle('%s W%i' % (coadd_id, band))
            ps.savefig()
            i0 += R*C


# T = fits_table('tab.fits')
# T.cut(T.band == 3)
# print len(T), 'in WISE coadd'
# F = fits_table('wise-coadds/unwise-1384p454-w3-frames.fits')
# print len(F), 'in unWISE coadd'
# 
# for s,f in zip(T.scan_id, T.frame_num):
#     I = np.flatnonzero((F.scan_id == s) * (F.frame_num == f))
#     if len(I) == 1:
#         continue
#     print 'scan/frame', s,f, ': not found'
#     #W = fits_table('sequels-frames.fits')
# sys.exit(0)

def pixel_area():
    for wcs in [Sip('05127a136-w1-int-1b.fits'),
        #wise-frames/2a/03242a/215/03242a215-w1-int-1b.fits'),
                Tan('data/unwise/001/0015p000/unwise-0015p000-w1-img-m.fits'),
        #wise-coadds/unwise-1384p454-w1-img.fits'),
                ]:
        W,H = wcs.get_width(), wcs.get_height()
        print 'W,H', W,H
        #xx,yy = np.meshgrid(np.arange(0, W, 10), np.arange(0, H, 10))
        xx,yy = np.meshgrid(np.arange(W), np.arange(H))
        rr,dd = wcs.pixelxy2radec(xx, yy)
        rr -= wcs.crval[0]
        rr *= np.cos(np.deg2rad(dd))
        dd -= wcs.crval[1]
        
        # (zero,zero) r,d
        zzr = rr[:-1, :-1]
        zzd = dd[:-1, :-1]
        ozr = rr[:-1, 1:]
        ozd = dd[:-1, 1:]
        zor = rr[1:, :-1]
        zod = dd[1:, :-1]
        oor = rr[1:, 1:]
        ood = dd[1:, 1:]
        
        a = np.hypot(zor - zzr, zod - zzd)
        A = np.hypot(oor - ozr, ood - ozd)
        b = np.hypot(ozr - zzr, ozd - zzd)
        B = np.hypot(oor - zor, ood - zod)
        C = np.hypot(ozr - zor, ozd - zod)
        c = C
        
        s = (a + b + c)/2.
        S = (A + B + C)/2.
        
        area = np.sqrt(s * (s-a) * (s-b) * (s-c)) + np.sqrt(S * (S-A) * (S-B) * (S-C))

        print 'Pixel area:'
        print ' min', area.min()
        print ' max', area.max()

        plt.clf()
        plt.imshow(area, interpolation='nearest', origin='lower')
        plt.title('Pixel area')
        plt.colorbar()
        ps.savefig()

# plt.clf()
# plt.plot(rr.ravel(), dd.ravel(), 'r.')
# plt.axis('scaled')
# ps.savefig()

def binimg(img, b):
    hh,ww = img.shape
    hh = int(hh / b) * b
    ww = int(ww / b) * b
    return (reduce(np.add, [img[i/b:hh:b, i%b:ww:b] for i in range(b*b)]) /
            float(b*b))

def paper_plots(coadd_id, band, ps, dir2='e',
                part1=True, part2=True, part3=True, part4=True,
                cmap_nims='jet', bw=False):
    figsize = (4,4)
    spa = dict(left=0.01, right=0.99, bottom=0.02, top=0.99,
               wspace=0.05, hspace=0.05)

    medfigsize = (5,4)
    medspa = dict(left=0.12, right=0.96, bottom=0.14, top=0.96)

    bigfigsize = (8,6)
    bigspa = dict(left=0.1, right=0.98, bottom=0.1, top=0.97)

    plt.figure(figsize=figsize)
    plt.subplots_adjust(**spa)

    dir1 = os.path.join(wisel3, coadd_id[:2], coadd_id[:4],
                        coadd_id + '_' + wise_tag)
    
    wiseim,wisehdr,unc,wisen = read_wise_coadd(coadd_id, band,
                                               unc=True, cov=True)

    imm    = read(dir2, 'unwise-%s-w%i-img-m.fits' % (coadd_id, band))
    imu    = read(dir2, 'unwise-%s-w%i-img-u.fits' % (coadd_id, band))

    ivm    = read(dir2, 'unwise-%s-w%i-invvar-m.fits' % (coadd_id, band))
    ivu    = read(dir2, 'unwise-%s-w%i-invvar-u.fits' % (coadd_id, band))

    ppstdm = read(dir2, 'unwise-%s-w%i-std-m.fits' % (coadd_id, band))

    numm   = read(dir2, 'unwise-%s-w%i-n-m.fits' % (coadd_id, band))
    numu   = read(dir2, 'unwise-%s-w%i-n-u.fits' % (coadd_id, band))

    binwise = binimg(wiseim, 25)
    binimm = binimg(imm, 16)
    binimu = binimg(imu, 16)

    sigm = 1./np.sqrt(np.maximum(ivm, 1e-16))
    sigm1 = np.median(sigm)
    print 'sigm1:', sigm1

    wisemed = np.median(wiseim[::4,::4])
    wisesig = np.median(unc[::4,::4])
    #wisesky = estimate_sky(wiseim, wisemed-2.*wisesig, wisemed+1.*wisesig)
    wisesky = estimate_sky_2(wiseim)
    print 'WISE sky estimate:', wisesky

    zp = wisehdr['MAGZP']
    print 'WISE image zeropoint:', zp
    zpscale = 1. / NanoMaggies.zeropointToScale(zp)
    print 'zpscale', zpscale

    P = fits_table('wise-psf-avg.fits', hdu=band)
    psf = GaussianMixturePSF(P.amp, P.mean, P.var)
    R = 100
    psf.radius = R
    pat = psf.getPointSourcePatch(0., 0.)
    pat = pat.patch
    pat /= pat.sum()
    psfnorm = np.sqrt(np.sum(pat**2))
    print 'PSF norm (native pixel scale):', psfnorm

    wise_unc_fudge = 2.4
    
    ima = dict(interpolation='nearest', origin='lower', cmap='gray')

    def myimshow(img, pp=[25,95]):
        plo,phi = [np.percentile(img, p) for p in [25,95]]
        imai = ima.copy()
        imai.update(vmin=plo, vmax=phi)
        plt.clf()
        plt.imshow(img, **imai)
        plt.xticks([]); plt.yticks([])

    if not part1:
        ps.skip(9)
    else:
        for img in [binwise, binimm, binimu]:
            myimshow(img)
            ps.savefig()
            
        hi,wi = wiseim.shape
        hj,wj = imm.shape
        #flo,fhi = 0.45, 0.55
        flo,fhi = 0.45, 0.50
        slcW = slice(int(hi*flo), int(hi*fhi)+1), slice(int(wi*flo), int(wi*fhi)+1)
        slcU = slice(int(hj*flo), int(hj*fhi)+1), slice(int(wj*flo), int(wj*fhi)+1)
    
        subwise = wiseim[slcW]
        subimm  = imm[slcU]
        subimu  = imu[slcU]
    
        for img in [subwise, subimm, subimu]:
            myimshow(img)
            ps.savefig()

        print 'Median coverage: WISE:', np.median(wisen)
        print 'Median coverage: unWISE w:', np.median(numm)
        print 'Median coverage: unWISE:', np.median(numu)

        #mx = max(wisen.max(), un.max(), unw.max())
        mx = 62.
        na = ima.copy()
        na.update(vmin=0, vmax=mx, cmap=cmap_nims)
        plt.clf()
        plt.imshow(wisen, **na)
        plt.xticks([]); plt.yticks([])
        ps.savefig()

        plt.clf()
        plt.imshow(numu, **na)
        plt.xticks([]); plt.yticks([])
        ps.savefig()
    
        w,h = figsize
        plt.figure(figsize=(w+1,h))
        plt.subplots_adjust(**spa)
    
        plt.clf()
        plt.imshow(numm, **na)
        plt.xticks([]); plt.yticks([])
    
        parent = plt.gca()
        pb = parent.get_position(original=True).frozen()
        #print 'pb', pb
        # new parent box, padding, child box
        frac = 0.15
        pad  = 0.05
        (pbnew, padbox, cbox) = pb.splitx(1.0-(frac+pad), 1.0-frac)
        # print 'pbnew', pbnew
        # print 'padbox', padbox
        # print 'cbox', cbox
        cbox = cbox.anchored('C', cbox)
        parent.set_position(pbnew)
        parent.set_anchor((1.0, 0.5))
        cax = parent.get_figure().add_axes(cbox)
        aspect = 20
        cax.set_aspect(aspect, anchor=((0.0, 0.5)), adjustable='box')
        parent.get_figure().sca(parent)
        plt.colorbar(cax=cax, ticks=[0,15,30,45,60])
        ps.savefig()
    

    if not part2:
        ps.skip(1)
    else:
        # Sky / Error properties

        # plt.figure(figsize=figsize)
        # plt.subplots_adjust(**spa)
        # 
        # dskyim = fitsio.read('g/138/1384p454/unwise-1384p454-w1-img-m.fits')
        # b = 15
        # xbinwise = reduce(np.add, [wiseim[i/b::b, i%b::b] for i in range(b*b)]) / float(b*b)
        # b = 8
        # #xbinim   = reduce(np.add, [im    [i/b::b, i%b::b] for i in range(b*b)]) / float(b*b)
        # xbinimw  = reduce(np.add, [imw   [i/b::b, i%b::b] for i in range(b*b)]) / float(b*b)
        # xbindsky = reduce(np.add, [dskyim [i/b::b, i%b::b] for i in range(b*b)]) / float(b*b)
        # 
        # #for img in [(binwise - wisesky) * zpscale / psfnorm, binim, binimw]:
        # for img in [(xbinwise - wisesky) * zpscale / psfnorm, xbinimw, xbindsky]:
        #     plt.clf()
        #     plt.imshow(img, *sigw1, vmax=3.*sigw1, **ima)
        #     #plt.imshow(img, vmin=-2.*sigw1, vmax=2.*sigw1, **ima)
        #     plt.xticks([]); plt.yticks([])
        #     ps.savefig()
        
        plt.figure(figsize=bigfigsize)
        plt.subplots_adjust(**bigspa)
        wisechi = ((wiseim-wisesky) / unc).ravel()
        #wisechi2 = 2. * ((wiseim-wisesky) / (unc/psfnorm)).ravel()
        #wisechi2 = (2.*psfnorm * (wiseim-wisesky) / unc).ravel()
        #wisechi2 = 0.5 * ((wiseim-wisesky) / unc).ravel()
        wisechi2 = ((wiseim-wisesky) / (wise_unc_fudge * unc)).ravel()

        #galpha = 0.3
        gsty = dict(linestyle='-', alpha=0.3)
        
        chiw = (imm / sigm).ravel()
        lo,hi = -6,12
        ha = dict(range=(lo,hi), bins=100, log=True, histtype='step')
        ha1 = dict(range=(lo,hi), bins=100)
        plt.clf()
        h1,e = np.histogram(wisechi, **ha1)
        h3,e = np.histogram(chiw, **ha1)
        nw = h3
        nwise = h1
        ee = e.repeat(2)[1:-1]
        p1 = plt.plot(ee, (h1/1.).repeat(2), zorder=25, color='r', lw=3, alpha=0.5)
        p3 = plt.plot(ee, h3.repeat(2), zorder=25, color='b', lw=2, alpha=0.75)
        plt.yscale('log')
        xx = np.linspace(lo, hi, 300)
        plt.plot(xx, max(nwise)*np.exp(-0.5*(xx**2)/(2.**2)), color='r', **gsty)
        plt.plot(xx, max(nwise)*np.exp(-0.5*(xx**2)/(2.5**2)), color='r', **gsty)
        plt.plot(xx, max(nw)*np.exp(-0.5*(xx**2)), color='b', **gsty)
        plt.xlabel('Pixel / Uncertainty ($\sigma$)')
        plt.ylabel('Number of pixels')

        wc = (wiseim-wisesky) / unc
        print 'wc', wc.shape
        pp = []
        for ii,cc in [
            (np.linspace(0, wc.shape[0],  6), 'm'),
            #(np.linspace(0, wc.shape[0], 11), 'r'),
            #(np.linspace(0, wc.shape[0], 21), 'g'),
            ]:
            nmx = []
            for ilo,ihi in zip(ii, ii[1:]):
                for jlo,jhi in zip(ii, ii[1:]):
                    wsub = wiseim[ilo:ihi, jlo:jhi]
                    usub = unc[ilo:ihi, jlo:jhi]
                    ssky = wisesky
                    #ssky = estimate_sky(wsub, wisemed-2.*wisesig, wisemed+1.*wisesig)
                    h,e = np.histogram(((wsub - ssky)/usub).ravel(), **ha1)
                    imax = np.argmax(h)
                    ew = (e[1]-e[0])/2.
                    de = -(e[imax] + ew)
                    plt.plot(ee + de, h.repeat(2), color=cc, lw=1, alpha=0.25)
                    #plt.plot(e[:-1] + de + ew, h, color=cc, lw=1, alpha=0.5)
                    nmx.append(max(h))
            # for legend only
            p4 = plt.plot([0],[1e10], color=cc)
            pp.append(p4[0])
            for s in [np.sqrt(2.), 2.]:
                plt.plot(xx, np.median(nmx)*np.exp(-0.5*(xx**2)/s**2),
                         zorder=20, color='k', **gsty)
        plt.legend([p1[0],p3[0]]+pp, ('WISE', 'unWISE', '5x5 sub WISE'))
        plt.ylim(3, 1e6)
        plt.xlim(lo,hi)
        plt.axvline(0, color='k', alpha=0.1)
        ps.savefig()
    
    if not part3:
        ps.skip(2)
    else:
        plt.figure(figsize=medfigsize)
        plt.subplots_adjust(**medspa)
        
        wiseflux = (wiseim - wisesky) * zpscale
        wiseerr  = unc * zpscale

        wiseflux /= psfnorm
        wiseerr *= wise_unc_fudge / psfnorm

        wiseerr1 = np.median(wiseerr)

        print 'WISE err1:', wiseerr1
        
        unflux = imm.ravel()
        unerr = ppstdm.ravel()

        print 'unWISE err1:', np.median(unerr)
        
        logflo,logfhi = -2, 5.
        logelo,logehi = 0., 3.
        #logelo,logehi = -0.5, 3.

        flo,fhi = 10.**logflo, 10.**logfhi
        elo,ehi = 10.**logelo, 10.**logehi

        # wf = wiseflux[::2, ::2].ravel()
        # plt.clf()
        # loghist(np.log10(wf), np.log10(unflux), range=[[np.log10(flo),np.log10(fhi)]]*2,
        #         nbins=200, hot=False, doclf=False,
        #         docolorbar=False, imshowargs=dict(cmap=antigray))
        # plt.xlabel('log WISE flux')
        # plt.ylabel('log unWISE flux')
        # ps.savefig()

        wiseflux = wiseflux.ravel()
        wiseerr  = wiseerr.ravel()

        ha = dict(hot=False, doclf=False, nbins=200,
                  range=((np.log10(flo),np.log10(fhi)),
                         (np.log10(elo),np.log10(ehi))),
                  docolorbar=False, imshowargs=dict(cmap=antigray))

        plt.clf()
        loghist(np.log10(np.clip(wiseflux, flo,fhi)),
                np.log10(np.clip(wiseerr,  elo,ehi)), **ha)
        plt.xlabel('WISE flux')
        plt.ylabel('WISE flux uncertainty')
        ax = plt.axis()
        xx = np.linspace(np.log10(flo), np.log10(fhi), 500)

        # yy = np.log10(np.hypot(wiseerr1, np.sqrt(0.01 * 10.**xx)))
        # plt.plot(xx, yy, 'm-', lw=2)
        yy = np.log10(np.hypot(wiseerr1, np.sqrt(0.02 * 10.**xx)))

        c = 'r'
        if bw:
            c = 'k'
        plt.plot(xx, yy, '-', lw=2, color=c)

        plt.axis(ax)
        logf = np.arange(logflo,logfhi+1)
        plt.xticks(logf, ['$10^{%i}$' % x for x in logf])
        loge = np.arange(int(np.ceil(logelo)),logehi+1)
        plt.yticks(loge, ['$10^{%i}$' % x for x in loge])
        ps.savefig()

        plt.clf()
        loghist(np.log10(np.clip(unflux, flo,fhi)),
                np.log10(np.clip(unerr,  elo,ehi)), **ha)
        plt.xlabel('unWISE flux')
        plt.ylabel('unWISE sample standard deviation')
        yy = np.log10(np.hypot(np.hypot(wiseerr1, np.sqrt(0.02 * 10.**xx)),
                               3e-2*(10.**xx)))
        ax = plt.axis()
        plt.plot(xx, yy, '-', lw=2, color=c)
        plt.axis(ax)
        logf = np.arange(logflo,logfhi+1)
        plt.xticks(logf, ['$10^{%i}$' % x for x in logf])
        loge = np.arange(int(np.ceil(logelo)),logehi+1)
        plt.yticks(loge, ['$10^{%i}$' % x for x in loge])
        ps.savefig()


        # plt.clf()
        # plt.hist(wiseflux / wiseerr, range=(-6,10), log=True, bins=100,
        #          histtype='step', color='r')
        # plt.hist(unflux / unerr, range=(-6,10), log=True, bins=100,
        #          histtype='step', color='b')
        # yl,yh = plt.ylim()
        # plt.ylim(0.1, yh)
        # ps.savefig()

    if part4:
        plt.figure(figsize=figsize)
        plt.subplots_adjust(**spa)

        hi,wi = wiseim.shape
        hj,wj = imm.shape

        # franges = [ (0.0,0.05), (0.45,0.5), (0.94,0.99) ]
        franges = [ (0.0,0.1), (0.45,0.55), (0.89,0.99) ]
        imargs = ima.copy()
        imargs.update(vmin=-2.*sigm1, vmax=2.*sigm1,
                      cmap='jet')
        if bw:
            imargs.update(cmap='gray')
        plt.clf()
        k = 1
        for yflo,yfhi in reversed(franges):
            for xflo,xfhi in franges:
                plt.subplot(len(franges),len(franges), k)
                k += 1
                slcW = (slice(int(hi*yflo), int(hi*yfhi)+1),
                        slice(int(wi*xflo), int(wi*xfhi)+1))
                subwise = wiseim[slcW]
                # bin
                # subwise = binimg(subwise, 2)
                subwise = binimg(subwise, 4)
                plt.imshow((subwise - wisesky) * zpscale / psfnorm, **imargs)
                plt.xticks([]); plt.yticks([])
        ps.savefig()
        plt.clf()
        k = 1
        for yflo,yfhi in reversed(franges):
            for xflo,xfhi in franges:
                plt.subplot(len(franges),len(franges), k)
                k += 1
                slcU = (slice(int(hj*yflo), int(hj*yfhi)+1),
                        slice(int(wj*xflo), int(wj*xfhi)+1))
                subimm = imm[slcU]
                subimm = binimg(subimm, 2)
                plt.imshow(subimm, **imargs)
                plt.xticks([]); plt.yticks([])
        ps.savefig()
            

class CompositeStage(object):
    def __init__(self):
        pass
    def __call__(self, stage, **kwargs):
        f = { 0:self.stage0, 1:self.stage1 }[stage]
        return f(**kwargs)
    def stage0(self, bands=None, coadd_id=None, medpct=None, dir2=None,
               fxlo=None, fxhi=None, fylo=None, fyhi=None, official=None,
               unmasked = True, fit_sky=False,
               sky_plo=5, sky_phi=60, bin=None, sky_args={},
               skyplots=True,
               **kwargs):
        wiseims = []
        imws = []
        ims = []
        #hi,wi = wiseims[0].shape
        #hj,wj = imws[0].shape
        hi,wi = 4095,4095
        slcI = (slice(int(hi*fylo), int(hi*fyhi)+1),
                slice(int(wi*fxlo), int(wi*fxhi)+1))
        # print 'slices:', slcI, slcJ

        if official:
            for band in bands:

                wiseim,wisehdr,unc = read_wise_coadd(coadd_id, band, unc=True)
                wisemed = np.percentile(wiseim[::4,::4], medpct)
                wisesig = np.median(unc[::4,::4])
                x,c,fc,wisesky = estimate_sky(wiseim, wisemed-2.*wisesig, wisemed+1.*wisesig,
                                              return_fit=True)
                print 'WISE sky', wisesky
    
                wiseim = wiseim[slcI]
                wiseim -= wisesky
                # adjust zeropoints
                zp = wisehdr['MAGZP']
                zpscale = 1. / NanoMaggies.zeropointToScale(zp)
                wiseim *= zpscale
                wisesig *= zpscale
                
                # plt.clf()
                # plt.plot(x, c, 'ro', alpha=0.5)
                # plt.plot(x, fc, 'b-', alpha=0.5)
                # plt.title('WISE W%i' % band)
                # ps.savefig()
        
                sky = estimate_sky(wiseim, -2.*wisesig, 1.*wisesig)
                print 'wise sky 2:', sky
                wiseim -= sky
                wiseims.append(wiseim)

                
        for band in bands:
            im = read(dir2, 'unwise-%s-w%i-img-m.fits' % (coadd_id, band))

            hj,wj = im.shape #2048,2048
            slcJ = (slice(int(hj*fylo), int(hj*fyhi)+1),
                    slice(int(wj*fxlo), int(wj*fxhi)+1))

            imws.append(im[slcJ])
            if unmasked:
                im = read(dir2, 'unwise-%s-w%i-img-u.fits'   % (coadd_id, band))
                ims.append(im[slcJ])
    
            # std = read(dir2, 'unwise-%s-w%i-std-m.fits' % (coadd_id, band))
            # sig = np.median(std[::4,::4])
            # print 'median std:', sig
            
            if fit_sky:
                imlist = [imws[-1]]
                if unmasked:
                    imlist.append(ims[-1])
                
                for im in imlist:

                    # med = np.percentile(im, medpct)
                    # # percentile ranges to include in sky fit
                    # plo,phi = sky_plo,sky_phi
                    # rlo,rhi = [np.percentile(im, p) for p in (plo,phi)]
                    # #rlo,rhi = (med-2.*sig, med+1.*sig)

                    x,c,fc,sky,warn,be1,c1 = estimate_sky_2(im, return_fit=True, **sky_args)
                    
                    if skyplots:
                        plt.clf()
                        plt.hist(im.ravel(), range=(np.percentile(im, 5),
                                                    np.percentile(im, 90)),
                                bins=100, histtype='step', color='b', log=True)
                        #plt.axvline(rlo, color='g')
                        #plt.axvline(rhi, color='g')
                        plt.axvline(sky, color='r')
                        plt.title('Sky estimate: W%i' % band)
                        ps.savefig()
    
                        plt.clf()
                        plt.plot(x, c, 'b-', alpha=0.5)
                        plt.plot(x, fc, 'r-', alpha=0.5)
                        plt.xlabel('image')
                        plt.ylabel('sky hist vs fit')
                        plt.title('Sky estimate: W%i' % band)
                        ps.savefig()
                        
                    # print 'med', med, 'sig', sig
                    print 'estimated sky', sky
                    im -= sky
                    # plt.clf()
                    # plt.plot(x, c, 'ro', alpha=0.5)
                    # plt.plot(x, fc, 'b-', alpha=0.5)
                    # plt.title('unWISE W%i' % band)
                    # ps.savefig()

        if bin is not None:
            wiseims = [binimg(im,bin) for im in wiseims]
            imws = [binimg(im,bin) for im in imws]
            ims = [binimg(im,bin) for im in ims]
                    
        return dict(wiseims=wiseims, imws=imws, ims=ims)
                    
    def stage1(self, wiseims=None, imws=None, ims=None, bands=None,
               compoffset=None, inset=None, official=None, 
               QQ = [20],
               SS = [100],
               unmasked = True,
               mix=None,
               r0=0., b0=0., g0=0.,
               w3scale=10.,
               w4scale=50., subsample=False,
               **kwargs):
        # soften W2
        # for im in [wiseims, imws, ims]:
        #     #im[1] /= 3.
        #     #im[1] /= 2
        #     #im[1] /= 1.5
        #     pass

        # soften W3,W4
        for imlist in [wiseims, imws, ims]:
            for band,im in zip(bands, imlist):
                if band == 3:
                    im /= w3scale
                if band == 4:
                    im /= w4scale
    
        # compensate for WISE psf norm
        if official:        
            for im in wiseims:
                im *= 4.
    
        # histograms
        if False:
            medfigsize = (5,3.5)
            medspa = dict(left=0.12, right=0.96, bottom=0.12, top=0.96)
            plt.figure(figsize=medfigsize)
            plt.subplots_adjust(**medspa)
            for imlist in [wiseims, imws, ims]:
                plt.clf()
                for im,cc,scale in zip(imlist, 'bgr', [1.,1.,0.2]):
                    plt.hist((im*scale).ravel(), range=(-5,20), histtype='step',
                             bins=100)
                ps.savefig()

        # plt.figure(figsize=(8,8))
        # #spa = dict(left=0.01, right=0.99, bottom=0.01, top=0.99)
        # spa = dict(left=0.005, right=0.995, bottom=0.005, top=0.995)
        # plt.subplots_adjust(**spa)

        # for im in [wiseims, imws, ims]:
        #     plt.clf()
        #     for i,Q in enumerate([3, 10, 30, 100]):
        #         plt.subplot(2,2, i+1)
        #         L = _lupton_comp([i/100 for i in im], Q=Q)
        #         plt.imshow(L, interpolation='nearest', origin='lower')
        #         plt.xticks([]); plt.yticks([])
        #     ps.savefig()

        # plt.figure(figsize=(4,4))
        # #spa = dict(left=0.01, right=0.99, bottom=0.01, top=0.99)
        # spa = dict(left=0.005, right=0.995, bottom=0.005, top=0.995)
        # plt.subplots_adjust(**spa)

        imslist,scales = [],[]
        if official:
            imslist.append(wiseims)
            scales.append(2.)
        if unmasked:
            imslist.append(ims)
            scales.append(1.)

        imslist.append(imws)
        scales.append(1.)
            
        for im,sc in zip(imslist, scales):

            plt.clf()

            k = 1

            # QQ = [10,20]
            # SS = [100,200]
            #QQ = [15,20,25]
            #SS = [50,100,200]
            for Q in QQ:
                for S in SS:

                    kwa = dict(Q=Q, clip=False)
                    if len(im) != 3:
                        L = _lupton_comp([i/S for i in im], **kwa)
                    else:
                        b,g,r = im
                        # R = g * 0.4 + r * 0.6
                        # G = b * 0.2 + g * 0.8
                        # B = b

                        if mix is None:
                            R = g * 0.8 + r * 0.5
                            G = b * 0.4 + g * 0.6
                            B = b * 1.0
                        else:
                            RGB = []
                            r += r0
                            g += g0
                            b += b0
                            for ar,ag,ab in mix:
                                print 'Mix:', ar,ag,ab
                                RGB.append(ar * r + ag * g + ab * b)
                            R,G,B = RGB
                        L = _lupton_comp([i/S for i in [B,G,R]], **kwa)

                    plt.subplot(len(QQ), len(SS), k)
                    plt.title('Q=%.0f, S=%.0f' % (Q, S))
                    k += 1

                    H,W,nil = L.shape
                    mn = min(H,W)
                    L = L[:mn,:mn]

                    if sc == 1 and subsample:
                        # Lanczos sub-sample so it has the same pixel resolution
                        # as the WISE image.
                        sh,sw,planes = L.shape
                        xx,yy = np.meshgrid(np.linspace(-0.5, sw-0.5, 2*sw),
                                            np.linspace(-0.5, sh-0.5, 2*sh))
                        xx = xx.ravel()
                        yy = yy.ravel()
                        ix = np.round(xx).astype(np.int32)
                        iy = np.round(yy).astype(np.int32)
                        dx = (xx - ix).astype(np.float32)
                        dy = (yy - iy).astype(np.float32)
                        RR = [np.zeros(sh*2*sw*2, np.float32) for i in range(planes)]
                        LL = [L[:,:,i] for i in range(planes)]
                        lanczos3_interpolate(ix, iy, dx, dy, RR, LL)
                        L = np.dstack([R.reshape((sh*2,sw*2)) for R in RR])
                        

                    
                    plt.imshow(np.clip(L, 0, 1),
                               interpolation='nearest', origin='lower')
                    plt.xticks([]); plt.yticks([])

                    print 'Inset:', inset
                    if inset is not None:
                        h,w,planes = L.shape
                        print 'w,h', w,h
                        xi = [int(np.round(i*w)) for i in inset[:2]]
                        yi = [int(np.round(i*h)) for i in inset[2:]]
                        dx = xi[1]-xi[0]
                        dy = yi[1]-yi[0]
                        dd = max(dx,dy)
                        Lsub = L[yi[0]:yi[0]+dd+1,xi[0]:xi[0]+dd+1]
                        
                        ax = plt.axis()
                        xl,xh = xi[0], xi[0]+dd
                        yl,yh = yi[0], yi[0]+dd
                        plt.plot([xl,xl,xh,xh,xl], [yl,yh,yh,yl,yl], 'w-')
                        plt.axis(ax)

                        ax = plt.axes([0.69, 0.01, 0.3, 0.3])
                        plt.sca(ax)
                        plt.setp(ax.spines.values(), color='w')
                        
                        if sc == 1:
                            # Lanczos sub-sample so it has the same pixel resolution
                            # as the WISE image.
                            sh,sw,planes = Lsub.shape
                            xx,yy = np.meshgrid(np.linspace(-0.5, sw-0.5, 2*sw),
                                                np.linspace(-0.5, sh-0.5, 2*sh))
                            xx = xx.ravel()
                            yy = yy.ravel()
                            ix = np.round(xx).astype(np.int32)
                            iy = np.round(yy).astype(np.int32)
                            dx = (xx - ix).astype(np.float32)
                            dy = (yy - iy).astype(np.float32)
                            RR = [np.zeros(sh*2*sw*2, np.float32) for i in range(planes)]
                            LL = [L[:,:,i] for i in range(planes)]
                            lanczos3_interpolate(xi[0]+ix, yi[0]+iy, dx, dy, RR, LL)
                            Lsub = np.dstack([R.reshape((sh*2,sw*2)) for R in RR])
                            
                        plt.imshow(np.clip(Lsub, 0, 1),
                                   interpolation='nearest', origin='lower')
                        plt.xticks([]); plt.yticks([])

            ps.savefig()
            

        # for im in [wiseims, imws, ims]:
        #     comp = _comp(im)
        #     plt.clf()
        # 
        #     comp += compoffset
        #     #comp = (comp/200.)**0.3
        #     #comp = (comp/100.)**0.4
        #     comp = (comp/200.)**0.4
        #     #comp = (comp/300.)**0.5
        #     #comp = (comp/300.)
        #     #comp = np.sqrt(comp/25.)
        # 
        #     plt.imshow(np.clip(comp, 0., 1.),
        #                interpolation='nearest', origin='lower')
        #     plt.xticks([]); plt.yticks([])
        #     ps.savefig()
        

def _comp(imlist):
    s = imlist[0]
    HI,WI = s.shape
    rgb = np.zeros((HI, WI, 3))
    if len(imlist) == 2:
        rgb[:,:,0] = imlist[1]
        rgb[:,:,2] = imlist[0]
        rgb[:,:,1] = (rgb[:,:,0] + rgb[:,:,2])/2.
    elif len(imlist) == 3:
        # rgb[:,:,0] = imlist[2]
        # rgb[:,:,1] = imlist[1]
        # rgb[:,:,2] = imlist[0]
        r,g,b = imlist[2], imlist[1], imlist[0]
        rgb[:,:,0] = g * 0.4 + r * 0.6
        rgb[:,:,1] = b * 0.2 + g * 0.8
        rgb[:,:,2] = b
    return rgb

def _lupton_comp(imlist, alpha=1.5, Q=30, m=-2e-2, clip=True):
    s = imlist[0]
    HI,WI = s.shape
    rgb = np.zeros((HI, WI, 3))
    if len(imlist) == 2:
        r = imlist[1]
        b = imlist[0]
        g = (r+b)/2.
    elif len(imlist) == 3:
        r,g,b = imlist[2], imlist[1], imlist[0]
    else:
        print len(imlist), 'images'
        assert(False)
        
    r = np.maximum(0, r - m)
    g = np.maximum(0, g - m)
    b = np.maximum(0, b - m)
    I = (r+g+b)/3.
    m2 = 0.
    fI = np.arcsinh(alpha * Q * (I - m2)) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    R = fI * r / I
    G = fI * g / I
    B = fI * b / I
    maxrgb = reduce(np.maximum, [R,G,B])
    J = (maxrgb > 1.)
    # R[J] = R[J]/maxrgb[J]
    # G[J] = G[J]/maxrgb[J]
    # B[J] = B[J]/maxrgb[J]
    RGB = np.dstack([R,G,B])
    if clip:
        RGB = np.clip(RGB, 0., 1.)
    return RGB

def composite(coadd_id, dir2='e', medpct=50, offset=0., bands=[1,2],
              cname='comp',
              df = 0.07,
              fxlo = 0.43, fylo = 0.51,
              fxhi = None, fyhi = None,
              inset=None,
            official=True,
            **kwargs):

    if fxhi is None:
        fxhi = fxlo + df
    if fyhi is None:
        fyhi = fylo + df

    print 'Composites for tile', coadd_id

    iargs = dict(coadd_id=coadd_id, dir2=dir2, bands=bands, medpct=medpct,
                compoffset=offset)
    args = dict(fxlo=fxlo, fxhi=fxhi, fylo=fylo, fyhi=fyhi,
                inset=inset, official=official)
    args.update(kwargs)
    
    runstage(1, 'comp-%s-stage%%02i.pickle' % cname, CompositeStage(),
             force=[1], initial_args=iargs, **args)
    return

    # for imlist in [wiseims, imws, ims]:
    #     plt.clf()
    #     for im,cc in zip(imlist, ['b','r']):
    #         plt.hist(im.ravel(), bins=100, histtype='step', color=cc,
    #                  range=(-5,30))
    #     plt.xlim(-5,30)
    #     ps.savefig()




def northpole_plots():
    for dirpat in ['n%i', 'nr%i',]:
        for n in range(0, 23):
            dir1 = dirpat % n
            fn = os.path.join(dir1, 'unwise-2709p666-w1-img-w.fits')
            if not os.path.exists(fn):
                print 'Skipping', fn
                continue
            print 'Reading', fn
            I = fitsio.read(fn)
    
            fn = os.path.join(dir1, 'unwise-2709p666-w1-n-w.fits')
            N = fitsio.read(fn)
            print 'Median N:', np.median(N)
            print 'Median non-zero N:', np.median(N[N > 0])
    
            plo,phi = [np.percentile(I, p) for p in [25,95]]
            print 'Percentiles', plo,phi
            ima = dict(interpolation='nearest', origin='lower', vmin=plo, vmax=phi, cmap='gray')
    
            plt.clf()
            plt.imshow(I, **ima)
            ps.savefig()
    
            plt.clf()
            plt.imshow(I[1000:1201,1000:1201], **ima)
            ps.savefig()



def medfilt_bg_plots(dirs=['e','f'], coadd_id='1384p454', bands=[3,4],
                     plo = 5, phi = 95, vlo=None, vhi=None,
                     cut=None, binning=[10,5]):
    figsize = (4,4)
    spa = dict(left=0.01, right=0.99, bottom=0.01, top=0.99)

    medfigsize = (5,4)
    medspa = dict(left=0.12, right=0.96, bottom=0.12, top=0.96)

    print 'bg plots'
    for band in bands:
        ims = []

        print 'reading WISE'
        dir1 = os.path.join(wisel3, coadd_id[:2], coadd_id[:4], coadd_id + '_ac51')
        wiseim,wisehdr = read(dir1, '%s_%s-w%i-int-3.fits' % (coadd_id, wise_tag, band), header=True, wget=True)
        unc    = read(dir1, '%s_%s-w%i-unc-3.fits.gz' % (coadd_id, wise_tag, band), wget=True)

        print 'Estimating WISE bg...'
        wisemed = np.median(wiseim[::4,::4])
        wisesig = np.median(unc[::4,::4])
        wisesky = estimate_sky(wiseim, wisemed-2.*wisesig, wisemed+1.*wisesig)
        zp = wisehdr['MAGZP']
        print 'WISE image zeropoint:', zp
        zpscale = 1. / NanoMaggies.zeropointToScale(zp)
        print 'zpscale', zpscale
        wiseflux = (wiseim - wisesky) * zpscale
        binwise = binimg(wiseflux, binning[0])
        ims.append(binwise)
        # approximate correction for PSF norm
        binwise *= 4.

        fullims = []
        for dir2 in dirs:
            imw    = read(dir2, 'unwise-%s-w%i-img-m.fits' % (coadd_id, band))
            ims.append(binimg(imw, binning[1]))
            fullims.append(imw)

        img = ims[0]
        if vlo is None:
            vlo = np.percentile(img, plo)
        if vhi is None:
            vhi = np.percentile(img, phi)

        print 'Plot range:', vlo, vhi
        
        ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                   vmin=vlo, vmax=vhi)

        plt.figure(figsize=figsize)
        plt.subplots_adjust(**spa)

        for img in ims:

            print 'Percentiles:', [np.percentile(img, p) for p in [plo,phi]]

            if cut is not None:
                h,w = img.shape
                xi = [int(np.round(i*w)) for i in cut[:2]]
                yi = [int(np.round(i*h)) for i in cut[2:]]
                img = img[yi[0]:yi[1]+1,xi[0]:xi[1]+1]
            
            plt.clf()
            plt.imshow(img, **ima)
            plt.xticks([]); plt.yticks([])
            ps.savefig()


        # nofilt,nofiltiv = fullims[0]
        # filt,filtiv = fullims[1]
        # sig1 = 1./np.sqrt(np.median(nofiltiv))
        # print 'No-filt sig1:', sig1
        # sig1 = 1./np.sqrt(np.median(filtiv))
        # print 'Filt sig1:', sig1

        if False:
            nofilt,filt = fullims
            #print 'lo,hi', lo,hi
            #lo = hi / 1e6
            hi = max(nofilt.max(), filt.max())
            lo = hi / 1e6
    
            plt.figure(figsize=medfigsize)
            plt.subplots_adjust(**medspa)
    
            plt.clf()
            rr = [np.log10(lo), np.log10(hi)]
            loghist(np.log10(np.maximum(lo, nofilt)).ravel(), np.log10(np.maximum(lo, filt.ravel())), 200,
                    range=[rr,rr], hot=False, imshowargs=dict(cmap=antigray))
            ax = plt.axis()
            #plt.plot(rr, rr, '--', color=(0,1,0))
            plt.plot(rr, rr, '--', color='r')
            plt.axis(ax)
            plt.xlabel('W%i: Pixel value' % band)
            plt.ylabel('W%i: Median filtered pixel value' % band)
            tt = np.arange(1,7)
            plt.xticks(tt, ['$10^{%i}$' % t for t in tt])
            plt.yticks(tt, ['$10^{%i}$' % t for t in tt])
            ps.savefig()

        # lo,hi = -6,8
        # 
        # ha = dict(range=(lo,hi), bins=100)
        # plt.clf()
        # h1,e = np.histogram((nofilt/sig1).ravel(), **ha)
        # h2,e = np.histogram((filt/sig1).ravel(), **ha)
        # ee = e.repeat(2)[1:-1]
        # p1 = plt.plot(ee, (h1).repeat(2), color='r', lw=2, alpha=0.75)
        # p2 = plt.plot(ee, (h2).repeat(2), color='b', lw=2, alpha=0.75)
        # plt.yscale('log')
        # xx = np.linspace(lo, hi, 300)
        # plt.plot(xx, max(h1) * np.exp(-(xx**2)/(2.)), 'b-', alpha=0.5)
        # plt.plot(xx, max(h2) * np.exp(-(xx**2)/(2.)), 'r-', alpha=0.5)
        # plt.xlabel('Pixel / Uncertainty ($\sigma$)')
        # plt.ylabel('Number of pixels')
        # plt.legend((p1,p2), ('No filter', 'Median filter'))
        # yl,yh = plt.ylim()
        # plt.ylim(3, yh)
        # plt.xlim(lo,hi)
        # plt.axvline(0, color='k', alpha=0.1)
        # ps.savefig()

#def medfilt_bad_bg_plots(dirs=['e','f']):
        

def download_tiles(T):
    # Download from IRSA:
    for coadd_id in T.coadd_id:
        print 'Coadd id', coadd_id
        #cmd = 'wget -r -N -nH -np -nv --cut-dirs=4 -A "*int-3.fits" "http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p3am_cdd/%s/%s/%s/"' % (coadd_id[:2], coadd_id[:4], coadd_id + '_ab41')
        cmd = 'wget -r -N -nH -np -nv --cut-dirs=4 -A "*unc-3.fits.gz" "http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p3am_cdd/%s/%s/%s/"' % (coadd_id[:2], coadd_id[:4], coadd_id + '_ab41')
        print 'Cmd:', cmd
        os.system(cmd)

def coverage_plots(cmap = matplotlib.cm.spectral):
    mn,mx = 0.,102.
    
    # Plot just the colorbar
    plt.figure(figsize=(0.5,4))
    plt.clf()
    ax = plt.gca()
    print 'ax pos', ax.get_position()
    ax.set_position([0.01, 0.02, 0.35, 0.95])
    cb = matplotlib.colorbar.ColorbarBase(
        ax, cmap=cmap,
        norm=matplotlib.colors.Normalize(vmin=mn, vmax=mx),
        )
    tt = [0,25,50,75,100]
    cb.set_ticks(tt, update_ticks=False)
    cb.update_ticks()
    ps.savefig()
    
    W,H = 800,400
    wcs = anwcs_create_allsky_hammer_aitoff2(180., 0., W, H)
    xx,yy = np.meshgrid(np.arange(W), np.arange(H))
    print 'xx,yy', xx.shape, yy.shape
    ok,rr,dd = wcs.pixelxy2radec(xx+1., yy+1.)
    print 'rr,dd', rr.shape, dd.shape, rr.dtype, dd.dtype
    print 'ok', ok.shape, ok.dtype
    rr = rr[ok==0]
    dd = dd[ok==0]

    Nside = 200
    hps = np.array([radecdegtohealpix(r, d, Nside) for r,d in zip(rr, dd)])

    counts = np.zeros(xx.shape, int)

    for band in [1,2,3,4]:
        hpcounts = fitsio.read('coverage-hp-w%i.fits' % band)
        assert(len(hpcounts) == 12*Nside**2)
        counts[ok==0] = hpcounts[hps]

        print 'W%i min:' % band, np.min(hpcounts)
        for p in [1,5,10,25,50]:
            print 'W%i: percentile %i coverage: %i' % (band, p, np.percentile(hpcounts, p))
        # continue
            
        rgb = cmap(np.clip( (counts - mn) / (mx - mn), 0, 1))
        for i in range(4):
            rgb[:,:,i][ok != 0] = 1
        print 'rgb', rgb.shape, rgb.dtype, rgb.min(), rgb.max()    
        # Trim off all-white parts
        for x0 in range(W):
            if not np.all(rgb[:,x0,0] == 1):
                break
        for y0 in range(H):
            if not np.all(rgb[y0,:,0] == 1):
                break
        # assume symmetry
        rgb = rgb[y0-1:-(y0-1),x0-1:-(x0-1),:]
        print 'rgb', rgb.shape, rgb.dtype, rgb.min(), rgb.max()    
            
        dpi=100.
        WW = W
        plt.figure(figsize=(WW/dpi, H/dpi), dpi=dpi)
        spa = dict(left=0, right=1, bottom=0.05, top=0.995)
        plt.subplots_adjust(**spa)

        # hack -- clip image outside the grid lines.
        dec = np.linspace(-90, 90, 200)
        yok,x1,y1 = wcs.radec2pixelxy(0., dec)
        yok,x2,y2 = wcs.radec2pixelxy(360., list(reversed(dec)))
        xy = np.vstack((np.hstack((x1-x0,x2-x0)), np.hstack((y1-y0,y2-y0)))).T
        poly = patches.Polygon(xy, closed=True, ec='none', fc='none')
        
        plt.clf()
        im = plt.imshow(rgb, interpolation='nearest', origin='lower')
        ax = plt.gca()
        ax.add_patch(poly)
        im.set_clip_path(poly)
        plt.gca().set_frame_on(False)
        plt.xticks([]); plt.yticks([])

        ax = plt.axis()
        cc = 'k'
        #if band in [3,4]:
        #    cc = '0.5'
        # grid lines
        for ra in np.arange(0, 361, 60):
            dec = np.linspace(-90, 90, 200)
            yok,x,y = wcs.radec2pixelxy(ra, dec)
            plt.plot(x-x0, y-y0, '-', color=cc)

            if band == 1 and ra < 360:
                lok,x,y = wcs.radec2pixelxy(ra, 0.)
                # bbox=dict(facecolor='1', alpha=0.5),
                plt.text(x-x0-5., y-y0, '%i' % ra, fontsize=20,
                         ha='right', va='bottom', color='k')
            
        for dec in np.arange(-90, 91, 30):
            ra = np.linspace(0, 360, 200)
            yok,x,y = wcs.radec2pixelxy(ra, dec)
            plt.plot(x-x0, y-y0, '-', color=cc)

        plt.axis(ax)
        ps.savefig()

    return
    
    for band,cc in zip([1,2,3,4], 'bgrm'):
        counts = fitsio.read('coverage-hp-w%i.fits' % band)

        print 'W',band
        print 'min:', counts.min()
        for p in [1,2,5,10,50,90,95,98,99]:
            print 'percentile', p, ':', np.percentile(counts, p), 'exposures'
        print 'max:', counts.max()

        plt.clf()
        plt.hist(counts, range=(0,60), bins=61, histtype='step',
                 color=cc, log=True)
        plt.ylim(0.3, 1e6)
        ps.savefig()
    
    totals = None
    for nbands in [2,3,4]:
        bb = [1,2,3,4][:nbands]
        for band in bb:
            fn = 'cov-n%i-b%i.fits' % (nbands, band)
            I = fitsio.read(fn)
            print I.shape
            if totals is None:
                H,W = I.shape
                totals = [np.zeros((H,W), int) for b in range(4)]
            totals[band-1] += I

    M = reduce(np.logical_or, [t > 0 for t in totals])
    
    for t,cc in zip(totals, 'bgrm'):
        plt.clf()
        plt.hist(t[M].ravel(), range=(0,60), bins=61, histtype='step',
                 color=cc, log=True)
        plt.ylim(0.3, 1e6)
        ps.savefig()

    for t,cc in zip(totals, 'bgrm'):
        bt = binimg(t, 10)
        plt.clf()
        plt.imshow(bt, interpolation='nearest', origin='lower',
                   cmap='hot', vmin=0, vmax=100)
        plt.xticks([]); plt.yticks([])
        plt.colorbar()
        ps.savefig()



if __name__ == '__main__':

    if True:
        ps = PlotSequence('ngc4321')

        x,y = 1694, 1919
        dpix = 150

        slc = (slice(y-dpix, y+dpix), slice(x-dpix, x+dpix))

        W1 = fitsio.FITS('186/1862p151/unwise-1862p151-w1-img-u.fits')[0][slc]
        W2 = fitsio.FITS('186/1862p151/unwise-1862p151-w2-img-u.fits')[0][slc]
        W3 = fitsio.FITS('1862p151-w3.fits')[0][slc]
        W4 = fitsio.FITS('1862p151-w4.fits')[0][slc]

        fitsio.write('ngc4321-W1.fits', W1)
        fitsio.write('ngc4321-W2.fits', W2)
        fitsio.write('ngc4321-W3.fits', W3)
        fitsio.write('ngc4321-W4.fits', W4)
        
        W3 /= 10.
        W4 /= 50.

        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(W1, interpolation='nearest', origin='lower')
        plt.subplot(2,2,2)
        plt.imshow(W2, interpolation='nearest', origin='lower')
        plt.subplot(2,2,3)
        plt.imshow(W3, interpolation='nearest', origin='lower')
        plt.subplot(2,2,4)
        plt.imshow(W4, interpolation='nearest', origin='lower')
        ps.savefig()

        W3 /= 5.
        W4 /= 5.

        # Lanczos sub-sample
        sh,sw = W1.shape
        scale = 4
        xx,yy = np.meshgrid(np.linspace(-0.5, sw-0.5, scale*sw),
                            np.linspace(-0.5, sh-0.5, scale*sh))
        xx = xx.ravel()
        yy = yy.ravel()
        ix = np.round(xx).astype(np.int32)
        iy = np.round(yy).astype(np.int32)
        dx = (xx - ix).astype(np.float32)
        dy = (yy - iy).astype(np.float32)
        RR = [np.zeros(sh*sw*scale**2, np.float32) for i in range(4)]
        LL = [W1,W2,W3,W4]
        lanczos3_interpolate(ix, iy, dx, dy, RR, LL)
        Lsub = [R.reshape((sh*scale,sw*scale)) for R in RR]

        W1,W2,W3,W4 = Lsub
        
        
        QQ = [10,20,40]
        #QQ = [20,40,80]
        #SS = [50, 100, 200]
        #SS = [200, 400, 800]
        SS = [400, 800, 1600]

        QQ, SS = [10], [1600]

        for im,tt in [ ( (W1,W2), 'W1/W2'),
                       ( (W3,W4), 'W3/W4'),
                       (((W1+W2)/2., 4.*(W3+W4)/2.), 'W1+W2/W3+W4'),
            #(0.5*(W1+W2)/2., 2.*W3, 2.*W4)]:
            ]:
            plt.clf()
            k = 1
            for Q in QQ:
                for S in SS:
                    kwa = dict(Q=Q, clip=False)
                    L = _lupton_comp([i/S for i in im], **kwa)

                    plt.subplot(len(QQ), len(SS), k)
                    k += 1                    
                    plt.imshow(np.clip(L, 0, 1),
                               interpolation='nearest', origin='lower')
                    plt.xticks([]); plt.yticks([])
                    #plt.title('Q=%g, S=%g' % (Q,S))
                    plt.title(tt)
            ps.savefig()
        
        # composite('1862p151', dir2='186/1862p151',
        #           bands=[1,2], cname='4321-12',
        #           fxlo=(x-dpix)/2048., fxhi=(x+dpix)/2048.,
        #           fylo=(y-dpix)/2048., fyhi=(y+dpix)/2048.,
        #           official=False)
        # composite('1862p151', dir2='186/1862p151',
        #           bands=[3,4], cname='4321-34',
        #           fxlo=(x-dpix)/2048., fxhi=(x+dpix)/2048.,
        #           fylo=(y-dpix)/2048., fyhi=(y+dpix)/2048.,
        #           official=False)

        sys.exit(0)
    
    if False:
        ps = PlotSequence('pix')
        pixel_area()
        sys.exit(0)
    
    if False:
        ps = PlotSequence('cov')
        ps.pattern = 'cov-%s-bw.%s'
        ps.suffixes = ['png','pdf']
        coverage_plots(cmap=antigray)
        ps = PlotSequence('cov')
        ps.suffixes = ['png','pdf']
        coverage_plots()
        sys.exit(0)
        
    if True:
        # ps = PlotSequence('co')
        # ps.suffixes = ['png','pdf']
        # T = fits_table('sequels-atlas.fits')
        # paper_plots(T.coadd_id[0], 1, ps, dir2='unwise-coadds/138/1384p454')

        ps = PlotSequence('co')
        ps.pattern = 'co-%s-bw.%s'
        ps.suffixes = ['png','pdf']
        T = fits_table('sequels-atlas.fits')
        paper_plots(T.coadd_id[0], 1, ps, dir2='unwise-coadds/138/1384p454',
                    cmap_nims='gray', bw=True)
        
        #part1=False, part2=False, part4=False)
        # paper_plots(T.coadd_id[0], 3, dir2='f')
        # paper_plots(T.coadd_id[0], 4, dir2='f')
        sys.exit(0)

    if True:
        ps = PlotSequence('medfilt')
        ps.suffixes = ['png','pdf']
        medfilt_bg_plots(dirs=['nomedfilt/138/1384p454/','medfilt/138/1384p454/'])
        sys.exit(0)

    if False:
        ps = PlotSequence('npole')
        ps.suffixes = ['png','pdf']
        T = fits_table('npole-atlas.fits')
        plt.figure(figsize=(3,3))
        spa = dict(left=0.005, right=0.995, bottom=0.005, top=0.995)
        plt.subplots_adjust(**spa)
    
        composite(T.coadd_id[6], dir2='npole',
                  cname='npole2',
                  inset=[156/400., (156+37)/400., 1.-(54+37)/400., 1.-(54/400.)],
                  fit_sky=True)
        composite(T.coadd_id[6], dir2='npole', bands=[1,2,3],
                  cname='npole3',
                  inset=[156/400., (156+37)/400., 1.-(54+37)/400., 1.-(54/400.)],
                  fit_sky=True)
        sys.exit(0)
    
    
    if False:
        ps = PlotSequence('medfilt-bad')
        ps.suffixes = ['png','pdf']
        for band,vlo,vhi in [#(3, -1000, 5000),
                             (4,-10000,10000)]:
            medfilt_bg_plots(dirs=['m31-nomedfilt/009/0098p408',
                                   'm31-medfilt/009/0098p408',
                                   'm31-nomedfilt-bgmatch/009/0098p408'],
                                   coadd_id='0098p408', bands=[band],
                                   plo=0,
                cut=[0,0.7, 0.3, 1.0],
                binning=[6,3])
        sys.exit(0)
    

    if True:
        #ps = PlotSequence('m31')
        ps = PlotSequence('cus')
        ps.suffixes = ['png','jpg']
    
        plt.figure(figsize=(6,6))
        spa = dict(left=0.005, right=0.995, bottom=0.005, top=0.995)
        plt.subplots_adjust(**spa)
    
        # dir2 = 'unwise-coadds/266/2660m303/'
        # tile = '2660m303'
        # cpre = 'gc'
    
        # dir2 = 'unwise-coadds/266/2666m258/'
        # tile = '2666m258'
        # cpre = 'gc2'
    
        # dir2 = 'unwise-coadds/265/2657m288/'
        # tile = '2657m288'
        # cpre = 'gc3'
    
        # dir2 = 'unwise-coadds/cus/custom-0838m053/'
        # tile = 'custom-0838m053'
        # cpre = 'orion'
    
        # dir2 = 'bgmatch/cus/custom-0838m053/'
        # tile = 'custom-0838m053'
        # cpre = 'orionbg'
    
        # QQ = [2, 10, 50]; SS=[1000,2000,4000]
        # QQ = [10, 25, 50]; SS=[2000,3000,4000]
        # QQ = [10, 25, 50]; SS=[2000,3000,4000]
        # QQ = [20, 25, 30]; SS=[2000,3000,4000]
    
        # M31:
        QQ=[25]; SS=[3000]
    
        #QQ=[10, 25, 100]; SS=[2000, 3000, 4000]
        #QQ=[25, 50, 100]; SS=[2000, 3000, 4000]
        #QQ=[25, 50, 100]; SS=[30, 300, 3000, 30000]
        QQ=[2, 10, 50]; SS=[100, 1000, 10000]
        
        kwaup = dict(unmasked=True)
        kwa34 = dict(w3scale=100, w4scale=1000)
    
        for dir2,tile,cpre in [
            #('m31/cus/custom-0106p412/', 'custom-0106p412', 'm31'),
            #('unwise-coadds/cus/custom-0836p220/', 'custom-0836p220', 'm1'),
            ('unwise-coadds/cus/custom-2709m244/', 'custom-2709m244', 'm8'),
            ('unwise-coadds/cus/custom-2747m138/', 'custom-2747m138', 'm16'),
            ]:
            kwaup = dict(fit_sky=True, m=0, bin=4)
    
            kwargs = dict(dir2=dir2, official=False, unmasked=False,
                          fxlo=0, fxhi=1, fylo=0, fyhi=1, 
                          QQ=QQ, SS=SS,
                          bin=2,
                          fit_sky=True, sky_args=dict(plo=1),
                          skyplots=False)
            k = kwargs.copy()
            k.update(kwaup)
    
            composite(tile, bands=[1,2], cname='%s-12' % cpre, **k)
    
    
        sys.exit(0)

        
    #ps = PlotSequence('co')
    ps = PlotSequence('m31')
    ps.suffixes = ['png','jpg']#,'pdf']#,'jpg']
    
    T = fits_table('npole-atlas.fits')
    plt.figure(figsize=(3,3))
    spa = dict(left=0.005, right=0.995, bottom=0.005, top=0.995)
    plt.subplots_adjust(**spa)
    composite(T.coadd_id[6], dir2='npole', medpct=30, offset=1.,
              cname='npole2',
              fxlo = 156/400. - 0.01,
              fxhi = (156+37)/400. + 0.01,
              fylo = 1.-(54+37)/400. - 0.01,
              fyhi = 1.-(54/400.))
    
    #ps.skipto(3)
    #ps.skipto(6)
    
    #coverage_plots()
    
    plt.figure(figsize=(6,6))
    spa = dict(left=0.005, right=0.995, bottom=0.005, top=0.995)
    #spa = dict(left=0.005, right=0.995, bottom=0.005, top=0.9, hspace=0.2)
    plt.subplots_adjust(**spa)
    
    # dir2 = 'unwise-coadds/266/2660m303/'
    # tile = '2660m303'
    # cpre = 'gc'
    
    # dir2 = 'unwise-coadds/266/2666m258/'
    # tile = '2666m258'
    # cpre = 'gc2'
    
    # dir2 = 'unwise-coadds/265/2657m288/'
    # tile = '2657m288'
    # cpre = 'gc3'
    
    dir2 = 'unwise-coadds/cus/custom-0838m053/'
    tile = 'custom-0838m053'
    cpre = 'orion'
    
    # dir2 = 'bgmatch/cus/custom-0838m053/'
    # tile = 'custom-0838m053'
    # cpre = 'orionbg'
    
    #QQ = [2, 10, 50]; SS=[1000,2000,4000]
    #QQ = [10, 25, 50]; SS=[2000,3000,4000]
    #QQ = [10, 25, 50]; SS=[2000,3000,4000]
    #QQ = [20, 25, 30]; SS=[2000,3000,4000]
    QQ=[25]; SS=[3000]
    kwaup = dict(unmasked=True)
    kwa34 = dict(w3scale=100, w4scale=1000)
    
    
    dir2 = 'm31-bgmatch/cus/custom-0106p412/'



dir2 = 'm31-bgmatch/cus/custom-0106p412/'
tile = 'custom-0106p412'
cpre = 'm31-bgm'

#QQ=[100]; SS=[800]
#QQ=[75]; SS=[800]
QQ=[50]; SS=[1000]
#QQ=[50,75,100]; SS=[600,800,1000]
#QQ=[50,100,200]; SS=[100,200,400]
kwaup = dict(fit_sky=True, m=0)
kwa34 = dict(w3scale=3, w4scale=10)

# dir2 = 
# cpre = 'm31'

for dir2,cpre in [
        ('m31-bgm/cus/custom-0106p412/', 'm31-bgm'),
        ('m31/cus/custom-0106p412/', 'm31'),]:

    tile = 'custom-0106p412'
    cpre = 'm31-bgm'
    
    #QQ=[100]; SS=[800]
    #QQ=[75]; SS=[800]
    QQ=[50]; SS=[1000]
    #QQ=[50,75,100]; SS=[600,800,1000]
    #QQ=[50,100,200]; SS=[100,200,400]
    kwaup = dict(fit_sky=True, m=0)
    kwa34 = dict(w3scale=3, w4scale=10)
    
    # dir2 = 
    # cpre = 'm31'
    
    for dir2,cpre in [
            ('m31-bgm/cus/custom-0106p412/', 'm31-bgm'),
            ('m31/cus/custom-0106p412/', 'm31'),]:
    
        tile = 'custom-0106p412'
        kwaup = dict(fit_sky=True, m=0, bin=5)
    
        kwargs = dict(dir2=dir2, official=False, unmasked=False,
                      fxlo=0, fxhi=1, fylo=0, fyhi=1, 
                      QQ=QQ, SS=SS,
                      bin=2,
                      fit_sky=True, sky_args=dict(plo=1),
                      skyplots=False)
    
        k = kwargs.copy()
        k.update(kwaup)
    
        composite(tile, bands=[1,2], cname='%s-12' % cpre, **k)
    
    
    #dir2 = 'm31-dsky/cus/custom-0106p412/'
    dir2 = 'm31-bgm-nomf/cus/custom-0106p412/'
    tile = 'custom-0106p412'
    cpre = 'm31-bgm-nomf'
    
    #QQ=[100]; SS=[200]
    #QQ=[10, 20, 25]; SS=[700,1000,2000]
    #QQ=[5, 10, 20]; SS=[1000,2000,3000]
    QQ=[10]; SS=[1000]
    #QQ=[20, 25, 50]; SS=[500,700,1000]
    #QQ=[10, 100, 1000]; SS=[20, 200, 2000]
    #QQ=[100]; SS=[200]
    kwaup = dict(fit_sky=True, m=0., dir2=dir2, QQ=QQ, SS=SS,
                 bin=5)
    kwa34 = dict(w3scale=25, w4scale=80)
    
    
    ps.skipto(2)
    
    k = kwargs.copy()
    k.update(kwaup)

    k.update(kwa34)
    composite(tile, bands=[3,4], cname='%s-34' % cpre, **k)
    sys.exit(0)
    
    composite(tile, bands=[1,2,3], cname='%s-123' % cpre,
              mix=[[1.0, 0., 0.,],
                   [0., 0.5, 0.,],
                   [0., 0., 0.7,]],
                   w3scale = 100,
    #w3scale=40,
    #r0=5,
    #w3scale=30,
    #              r0=25,
              **kwargs)
    
    composite(tile, bands=[1,2,4], cname='%s-124' % cpre,
              mix=[[1.0, 0., 0.,],
                   [0., 0.4, 0.,],
                   [0., 0., 0.6,]],
                   w4scale = 1000,
    #w4scale=500,
    #r0=5,
              **kwargs)
    
    composite(tile, bands=[2,3,4], cname='%s-234' % cpre,
              mix=[[1.0, 0., 0.,],
                   [0., 1.0, 0.,],
                   [0., 0.4, 0.6,]],
                   w3scale=100,
                   w4scale=1000,
              **kwargs)
    
    sys.exit(0)
    
    T = fits_table('npole-atlas.fits')
    #download_tiles(T)
    #ps.skip(3)
    
    plt.figure(figsize=(4,4))#, edgecolor='w')
    spa = dict(left=0.005, right=0.995, bottom=0.005, top=0.995)
    plt.subplots_adjust(**spa)
    
    composite(T.coadd_id[6], dir2='npole', medpct=30, offset=1.,
              cname='npole2',
              inset=[156/400., (156+37)/400., 1.-(54+37)/400., 1.-(54/400.)])
    
    composite(T.coadd_id[6], dir2='npole', medpct=30, offset=1., bands=[1,2,3],
              cname='npole3',
              inset=[156/400., (156+37)/400., 1.-(54+37)/400., 1.-(54/400.)])
    #inset=[i/float(400.) for i in [98, 98+44, 198, 198+44]])
    sys.exit(0)
    
    plt.figure(figsize=(6,4))
    spa = dict(left=0.005, right=0.995, bottom=0.005, top=0.995)
    plt.subplots_adjust(**spa)
    
    composite(T.coadd_id[6], dir2='npole', medpct=30, offset=1., bands=[1,2,3],
              cname='npole4',
              fxlo=0.43, fxhi=0.535, fylo=0.51, fyhi=0.58)
    
    sys.exit(0)
    #composite(T.coadd_id[3], dir2='npole')
    #sys.exit(0)
    
    #northpole_plots()
    #T = fits_table('sequels-atlas.fits')
    
    #T = fits_table('sequels-atlas.fits')
    #paper_plots(T.coadd_id[0], 1)
    
    #composite(T.coadd_id[0])
    
    
    
    
    #T.cut(np.array([0]))
    bands = [1,2,3,4]
    
    plt.figure(figsize=(12,4))
    #plt.subplots_adjust(bottom=0.01, top=0.85, left=0., right=1., wspace=0.05)
    #plt.subplots_adjust(bottom=0.1, top=0.85, left=0., right=0.9, wspace=0.05)
    plt.subplots_adjust(bottom=0.1, top=0.85, left=0.05, right=0.9, wspace=0.15)
    
    
    for coadd_id in T.coadd_id[:5]:
        dir1 = os.path.join(wisel3, coadd_id[:2], coadd_id[:4], coadd_id + '_ab41')
    
        for band in bands:
    
            dir2 = 'e'
    
            try:
                wiseim = read(dir1, '%s_ab41-w%i-int-3.fits' % (coadd_id, band))
                imw    = read(dir2, 'unwise-%s-w%i-img-w.fits' % (coadd_id, band))
                im     = read(dir2, 'unwise-%s-w%i-img.fits' % (coadd_id, band))
    
                unc    = read(dir1, '%s_ab41-w%i-unc-3.fits.gz' % (coadd_id, band))
                ivw    = read(dir2, 'unwise-%s-w%i-invvar-w.fits' % (coadd_id, band))
                iv     = read(dir2, 'unwise-%s-w%i-invvar.fits' % (coadd_id, band))
    
                # cmd = ('wget -r -N -nH -np -nv --cut-dirs=5 -P %s "http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p3am_cdd/%s/%s/%s/%s"' %
                #        (wisel3, coadd_id[:2], coadd_id[:4], coadd_id + '_ab41', os.path.basename(ufn1)))
                # print 'Cmd:', cmd
                # os.system(cmd)
    
            except:
                continue
    
            I = wiseim
            J = imw
            K = im
            
            L = ivw
            M = iv
    
            binI = reduce(np.add, [I[i/5::5, i%5::5] for i in range(25)]) / 25.
            binJ = reduce(np.add, [J[i/4::4, i%4::4] for i in range(16)]) / 16.
            binK = reduce(np.add, [K[i/4::4, i%4::4] for i in range(16)]) / 16.
            # binI = I[::5,::5]
            # binJ = J[::4,::4]
            # binK = K[::4,::4]
            
            ima = dict(interpolation='nearest', origin='lower', cmap='gray')
    
            plo,phi = [np.percentile(binI, p) for p in [25,99]]
            imai = ima.copy()
            imai.update(vmin=plo, vmax=phi)
            plo,phi = [np.percentile(binJ, p) for p in [25,99]]
            imaj = ima.copy()
            imaj.update(vmin=plo, vmax=phi)
    
            plt.clf()
            plt.subplot(1,3,1)
            plt.imshow(binI, **imai)
            plt.colorbar()
            plt.xticks([]); plt.yticks([])
            plt.title('WISE')
            plt.subplot(1,3,2)
            plt.imshow(binJ, **imaj)
            plt.xticks([]); plt.yticks([])
            plt.title('unWISE weighted')
            plt.subplot(1,3,3)
            plt.imshow(binK, **imaj)
            plt.xticks([]); plt.yticks([])
            plt.title('unWISE')
            plt.colorbar()
            plt.suptitle('%s W%i' % (coadd_id, band))
            ps.savefig()
    
    
            # Emphasize the sky levels
            
            plo,phi = [np.percentile(binI, p) for p in [1,70]]
            imai = ima.copy()
            imai.update(vmin=plo, vmax=phi)
            plo,phi = [np.percentile(binJ, p) for p in [1,70]]
            imaj = ima.copy()
            imaj.update(vmin=plo, vmax=phi)
    
            plt.clf()
            plt.subplot(1,3,1)
            plt.imshow(binI, **imai)
            plt.colorbar()
            plt.xticks([]); plt.yticks([])
            plt.title('WISE')
            plt.subplot(1,3,2)
            plt.imshow(binJ, **imaj)
            plt.xticks([]); plt.yticks([])
            plt.title('unWISE weighted')
            plt.subplot(1,3,3)
            plt.imshow(binK, **imaj)
            plt.xticks([]); plt.yticks([])
            plt.title('unWISE')
            plt.colorbar()
            plt.suptitle('%s W%i' % (coadd_id, band))
            ps.savefig()
    
    
    
    
            sig1w = 1./np.sqrt(np.median(ivw))
            sig1 = 1./np.sqrt(np.median(iv))
            unc1 = np.median(unc)
            print 'sig1w:', sig1w
            print 'sig1:', sig1
            print 'unc:', unc1
    
            med = np.median(wiseim)
            sigw = 1./np.sqrt(ivw)
    
            plt.clf()
            lo,hi = -8,10
            ha = dict(bins=100, histtype='step', range=(lo,hi), log=True)
            plt.hist((im / sig1).ravel(), color='g', lw=2, **ha)
            n,b,p = plt.hist((imw / sig1w).ravel(), color='b', **ha)
            plt.hist((imw / sigw).ravel(), color='c', **ha)
            plt.hist(((wiseim - med) / unc1).ravel(), color='r', **ha)
            plt.hist(((wiseim - med) / unc).ravel(), color='m', **ha)
            yl,yh = plt.ylim()
            xx = np.linspace(lo, hi, 300)
            plt.plot(xx, max(n) * np.exp(-(xx**2)/(2.)), 'r--')
            plt.ylim(0.1, yh)
            plt.xlim(lo,hi)
            ps.savefig()
    
            # plt.clf()
            # loghist(im.ravel(), imw.ravel(), range=[[-10*sig1,10*sig1]]*2, bins=200)
            # plt.xlabel('im')
            # plt.ylabel('imw')
            # ps.savefig()
    
    
            L = 1./np.sqrt(L)
            M = 1./np.sqrt(M)
            # binL = reduce(np.add, [L[i/4::4, i%4::4] for i in range(16)])
            # binM = reduce(np.add, [M[i/4::4, i%4::4] for i in range(16)])
            # binunc = reduce(np.add, [unc[i/5::5, i%5::5] for i in range(25)])
            binL = L[::4,::4]
            binM = M[::4,::4]
            binunc = unc[::5,::5]
    
    
            plt.clf()
    
            plo,phi = [np.percentile(binunc, p) for p in [25,99]]
            imaj = ima.copy()
            imaj.update(vmin=plo, vmax=phi)
    
            plt.subplot(1,3,1)
            plt.imshow(binunc, **imaj)
            plt.colorbar()
            plt.xticks([]); plt.yticks([])
            plt.title('WISE unc')
    
            plo,phi = [np.percentile(binL, p) for p in [25,99]]
            imaj = ima.copy()
            imaj.update(vmin=plo, vmax=phi)
    
            plt.subplot(1,3,2)
            plt.imshow(binL, **imaj)
            plt.xticks([]); plt.yticks([])
            plt.title('unWISE unc (weighted)')
    
            plt.subplot(1,3,3)
            plt.imshow(binM, **imaj)
            plt.colorbar()
            plt.xticks([]); plt.yticks([])
            plt.title('unWISE unc')
    
            ps.savefig()
    
            ## J: wim
            ## K: im
            ## L: wiv -> sig
            ## M: iv  -> sig
            chia = J / L
            chib = K / M
    
            print 'chia:', chia.min(), chia.max()
            print 'chib:', chib.min(), chib.max()
    
            # plt.clf()
            # plt.subplot(1,3,2)
            # n,b,p = plt.hist(chia.ravel(), bins=100, log=True, range=(-20,20), histtype='step', color='r')
            # plt.ylim(0.1, max(n)*2)
            # plt.subplot(1,3,3)
            # n,b,p = plt.hist(chib.ravel(), bins=100, log=True, range=(-20,20), histtype='step', color='b')
            # plt.ylim(0.1, max(n)*2)
            # ps.savefig()
    
    
            fn6 = os.path.join(dir2, 'unwise-%s-w%i-ppstd-w.fits' % (coadd_id, band))
            print fn6
            if not os.path.exists(fn6):
                print '-> does not exist'
                continue
            ppstd = fitsio.read(fn6)
    
    
            plt.clf()
            plt.subplot(1,3,1)
            loghist(np.clip(np.log10(I.ravel()), -2,4), np.clip(np.log10(unc.ravel()), -2, 4), doclf=False, docolorbar=False)
            plt.title('WISE int vs unc')
            plt.subplot(1,3,2)
            loghist(np.clip(np.log10(J.ravel()), -1, 5), np.clip(np.log10(L.ravel()), -1, 5), doclf=False, docolorbar=False)
            plt.title('unWISE img vs 1/sqrt(iv)')
            plt.subplot(1,3,3)
            loghist(np.clip(np.log10(J.ravel()), -1, 5), np.clip(np.log10(ppstd.ravel()), -1, 5), doclf=False, docolorbar=False)
            plt.title('unWISE img vs ppstd')
            ps.savefig()
    
    
    
            fn1 = os.path.join(dir1, '%s_ab41-w%i-cov-3.fits.gz' % (coadd_id, band))
            print fn1
            if not os.path.exists(fn1):
                print '-> does not exist'
                cmd = ('wget -r -N -nH -np -nv --cut-dirs=5 -P %s "http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p3am_cdd/%s/%s/%s/%s"' %
                       (wisel3, coadd_id[:2], coadd_id[:4], coadd_id + '_ab41', os.path.basename(fn1)))
                print 'Cmd:', cmd
                os.system(cmd)
    
            fn2 = os.path.join(dir2, 'unwise-%s-w%i-n-w.fits' % (coadd_id, band))
            print fn2
            if not os.path.exists(fn2):
                print '-> does not exist'
                continue
    
            fn3 = os.path.join(dir2, 'unwise-%s-w%i-n.fits' % (coadd_id, band))
            print fn3
            if not os.path.exists(fn3):
                print '-> does not exist'
                continue
    
            I = fitsio.read(fn1)
            J = fitsio.read(fn2)
            K = fitsio.read(fn3)
            # binJ = reduce(np.add, [J[i/4::4, i%4::4] for i in range(16)])
            # binK = reduce(np.add, [K[i/4::4, i%4::4] for i in range(16)])
            binI = I[::5,::5]
            binJ = J[::4,::4]
            binK = K[::4,::4]
    
            plo,phi = min(binI.min(), binJ.min(), binK.min()), max(binI.max(), binJ.max(),binK.max())
            imaj = ima.copy()
            imaj.update(vmin=plo, vmax=phi, cmap='jet')
    
            plt.clf()
    
            plt.subplot(1,3,1)
            plt.imshow(binI, **imaj)
            plt.xticks([]); plt.yticks([])
            plt.title('WISE cov')
    
            plt.subplot(1,3,2)
            plt.imshow(binJ, **imaj)
            plt.xticks([]); plt.yticks([])
            plt.title('unWISE n (weighted)')
    
            plt.subplot(1,3,3)
            plt.imshow(binK, **imaj)
            plt.colorbar()
            plt.xticks([]); plt.yticks([])
            plt.title('unWISE n')
    
            ps.savefig()
    
        #break
    
    
    sys.exit(0)
    
    
    composite(tile, bands=[1,2], cname='%s-12' % cpre, **k)


#dir2 = 'm31-dsky/cus/custom-0106p412/'
dir2 = 'm31-bgm-nomf/cus/custom-0106p412/'
tile = 'custom-0106p412'
cpre = 'm31-bgm-nomf'

#QQ=[100]; SS=[200]
#QQ=[10, 20, 25]; SS=[700,1000,2000]
#QQ=[5, 10, 20]; SS=[1000,2000,3000]
QQ=[10]; SS=[1000]
#QQ=[20, 25, 50]; SS=[500,700,1000]
#QQ=[10, 100, 1000]; SS=[20, 200, 2000]
#QQ=[100]; SS=[200]
kwaup = dict(fit_sky=True, m=0., dir2=dir2, QQ=QQ, SS=SS,
             bin=5)
kwa34 = dict(w3scale=25, w4scale=80)


ps.skipto(2)

k = kwargs.copy()
k.update(kwaup)
k.update(kwa34)
composite(tile, bands=[3,4], cname='%s-34' % cpre, **k)
sys.exit(0)

composite(tile, bands=[1,2,3], cname='%s-123' % cpre,
          mix=[[1.0, 0., 0.,],
               [0., 0.5, 0.,],
               [0., 0., 0.7,]],
               w3scale = 100,
#w3scale=40,
#r0=5,
#w3scale=30,
#              r0=25,
          **kwargs)

composite(tile, bands=[1,2,4], cname='%s-124' % cpre,
          mix=[[1.0, 0., 0.,],
               [0., 0.4, 0.,],
               [0., 0., 0.6,]],
               w4scale = 1000,
#w4scale=500,
#r0=5,
          **kwargs)

composite(tile, bands=[2,3,4], cname='%s-234' % cpre,
          mix=[[1.0, 0., 0.,],
               [0., 1.0, 0.,],
               [0., 0.4, 0.6,]],
               w3scale=100,
               w4scale=1000,
          **kwargs)

sys.exit(0)

T = fits_table('npole-atlas.fits')
plt.figure(figsize=(6,4))
spa = dict(left=0.005, right=0.995, bottom=0.005, top=0.995)
plt.subplots_adjust(**spa)

composite(T.coadd_id[6], dir2='npole', medpct=30, offset=1., bands=[1,2,3],
          cname='npole4',
          fxlo=0.43, fxhi=0.535, fylo=0.51, fyhi=0.58)

sys.exit(0)

#T.cut(np.array([0]))
bands = [1,2,3,4]

plt.figure(figsize=(12,4))
#plt.subplots_adjust(bottom=0.01, top=0.85, left=0., right=1., wspace=0.05)
#plt.subplots_adjust(bottom=0.1, top=0.85, left=0., right=0.9, wspace=0.05)
plt.subplots_adjust(bottom=0.1, top=0.85, left=0.05, right=0.9, wspace=0.15)


for coadd_id in T.coadd_id[:5]:
    dir1 = os.path.join(wisel3, coadd_id[:2], coadd_id[:4], coadd_id + '_ab41')

    for band in bands:

        dir2 = 'e'

        try:
            wiseim = read(dir1, '%s_ab41-w%i-int-3.fits' % (coadd_id, band))
            imw    = read(dir2, 'unwise-%s-w%i-img-w.fits' % (coadd_id, band))
            im     = read(dir2, 'unwise-%s-w%i-img.fits' % (coadd_id, band))

            unc    = read(dir1, '%s_ab41-w%i-unc-3.fits.gz' % (coadd_id, band))
            ivw    = read(dir2, 'unwise-%s-w%i-invvar-w.fits' % (coadd_id, band))
            iv     = read(dir2, 'unwise-%s-w%i-invvar.fits' % (coadd_id, band))

            # cmd = ('wget -r -N -nH -np -nv --cut-dirs=5 -P %s "http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p3am_cdd/%s/%s/%s/%s"' %
            #        (wisel3, coadd_id[:2], coadd_id[:4], coadd_id + '_ab41', os.path.basename(ufn1)))
            # print 'Cmd:', cmd
            # os.system(cmd)

        except:
            continue

        I = wiseim
        J = imw
        K = im
    
    lst = os.listdir(wisel3)
    lst.sort()
    bands = [1,2,3,4]
    
    # HACK
    #lst = ['1917p454_ab41', '1273p575_ab41']
    #lst = ['1273p575_ab41']
    lst = ['1190p575_ab41']
    bands = [1,2]
    #bands = [2]
    
    
    for band in bands:
        for l3dir in lst:
            print 'dir', l3dir
            coadd = l3dir.replace('_ab41','')
            l3fn = os.path.join(wisel3, l3dir, '%s-w%i-int-3.fits' % (l3dir, band))
            if not os.path.exists(l3fn):
                print 'Missing', l3fn
                continue
            cofn  = os.path.join(coadds, 'unwise-%s-w%i-img.fits'   % (coadd, band))
            cowfn = os.path.join(coadds, 'unwise-%s-w%i-img-w.fits' % (coadd, band))
            if not os.path.exists(cofn) or not os.path.exists(cowfn):
                print 'Missing', cofn, 'or', cowfn
                continue
    
            I = fitsio.read(l3fn)
            J = fitsio.read(cofn)
            K = fitsio.read(cowfn)
    
            print 'coadd range:', J.min(), J.max()
            print 'w coadd range:', K.min(), K.max()
    
            hi,wi = I.shape
            hj,wj = J.shape
            flo,fhi = 0.45, 0.55
            slcI = slice(int(hi*flo), int(hi*fhi)+1), slice(int(wi*flo), int(wi*fhi)+1)
            slcJ = slice(int(hj*flo), int(hj*fhi)+1), slice(int(wj*flo), int(wj*fhi)+1)
    
            ima = dict(interpolation='nearest', origin='lower', cmap='gray')
    
            plo,phi = [np.percentile(I, p) for p in [25,99]]
            imai = ima.copy()
            imai.update(vmin=plo, vmax=phi)
    
            plt.clf()
            plt.imshow(I, **imai)
            plt.title('WISE team %s' % os.path.basename(l3fn))
            ps.savefig()
    
            plt.clf()
            plt.imshow(I[slcI], **imai)
            plt.title('WISE team %s' % os.path.basename(l3fn))
            ps.savefig()
    
            plo,phi = [np.percentile(J, p) for p in [25,99]]
            imaj = ima.copy()
            imaj.update(vmin=plo, vmax=phi)
    
            plt.clf()
            plt.imshow(J[slcJ], **imaj)
            plt.title('My unweighted %s' % os.path.basename(cofn))
            ps.savefig()
    
            plt.clf()
            plt.imshow(K[slcJ], **imaj)
            plt.title('My weighted %s' % os.path.basename(cowfn))
            ps.savefig()
    
                                
    sys.exit(0)
    
    
    
    
    
    
    
    for coadd in ['1384p454',
        #'2195p545',
                  ]:
    
        for band in []: #1,2,3,4]: #[1]:
            F = fits_table('wise-coadds/unwise-%s-w%i-frames.fits' % (coadd,band))
    
            frame0 = F[0]
    
            overlaps = np.zeros(len(F))
            for i in range(len(F)):
                ext = F.coextent[i]
                x0,x1,y0,y1 = ext
                poly = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
                if i == 0:
                    poly0 = poly
                clip = clip_polygon(poly, poly0)
                if len(clip) == 0:
                    continue
                print 'clip:', clip
                x0,y0 = np.min(clip, axis=0)
                x1,y1 = np.max(clip, axis=0)
                overlaps[i] = (y1-y0)*(x1-x0)
            I = np.argsort(-overlaps)
            for i in I[:5]:
                frame = '%s%03i' % (F.scan_id[i], F.frame_num[i])
                #imgfn = '%s-w%i-int-1b.fits' % (frame, band)
                imgfn = F.intfn[i]
                print 'Reading image', imgfn
                img = fitsio.read(imgfn)
    
                okimg = img.flat[np.flatnonzero(np.isfinite(img))]
                plo,phi = [np.percentile(okimg, p) for p in [25,99]]
                print 'Percentiles', plo, phi
                ima = dict(interpolation='nearest', origin='lower',
                           vmin=plo, vmax=phi)
                plt.clf()
                plt.imshow(img, **ima)
                plt.title('Image %s W%i' % (frame,band))
                ps.savefig()
    
            for i in I[:5]:
                frame = '%s%03i' % (F.scan_id[i], F.frame_num[i])
                #maskfn = '%s-w%i-msk-1b.fits.gz' % (frame, band)
                #mask = fitsio.read(maskfn)
                print 'Reading', comaskfn
                comaskfn = 'wise-coadds/masks-coadd-%s-w%i/coadd-mask-%s-%s-w%i-1b.fits' % (coadd, band, coadd, frame, band)
                comask = fitsio.read(comaskfn)
    
                #plt.clf()
                #plt.imshow(mask > 0, interpolation='nearest', origin='lower',
                #           vmin=0, vmax=1)
                #plt.axis(ax)
                #plt.title('WISE mask')
                #ps.savefig()
    
                plt.clf()
                plt.imshow(comask > 0, interpolation='nearest', origin='lower',
                           vmin=0, vmax=1)
                plt.title('Coadd mask')
                ps.savefig()
    
    
        for frame in []: #'05579a167']:
            for band in [1]:
                imgfn = '%s-w%i-int-1b.fits' % (frame, band)
                img = fitsio.read(imgfn)
                maskfn = '%s-w%i-msk-1b.fits.gz' % (frame, band)
                mask = fitsio.read(maskfn)
                comaskfn = 'coadd-mask-%s-%s-w%i-1b.fits' % (coadd, frame, band)
                comask = fitsio.read(comaskfn)
    
                plo,phi = [np.percentile(img, p) for p in [25,98]]
                ima = dict(interpolation='nearest', origin='lower',
                           vmin=plo, vmax=phi)
                ax = [200,700,200,700]
                plt.clf()
                plt.imshow(img, **ima)
                plt.axis(ax)
                plt.title('Image %s W%i' % (frame,band))
                ps.savefig()
    
                plt.clf()
                plt.imshow(mask > 0, interpolation='nearest', origin='lower',
                           vmin=0, vmax=1)
                plt.axis(ax)
                plt.title('WISE mask')
                ps.savefig()
    
                plt.clf()
                plt.imshow(comask > 0, interpolation='nearest', origin='lower',
                           vmin=0, vmax=1)
                plt.axis(ax)
                plt.title('Coadd mask')
                ps.savefig()
                
    
        II = []
        JJ = []
        KK = []
        ppI = []
        ppJ = []
        for band in [1,2]:#,3,4]:
            fni = 'L3a/%s_ab41/%s_ab41-w%i-int-3.fits' % (coadd, coadd, band)
            I = fitsio.read(fni)
            fnj = 'wise-coadds/coadd-%s-w%i-img.fits' % (coadd, band)
            J = fitsio.read(fnj)
            fnk = 'wise-coadds/coadd-%s-w%i-img-w.fits' % (coadd, band)
            K = fitsio.read(fnk)
    
            wcsJ = Tan(fnj)
    
            II.append(I)
            JJ.append(J)
            KK.append(K)
            
            plt.clf()
            plo,phi = [np.percentile(I, p) for p in [25,99]]
            pmid = np.percentile(I, 50)
            p95 = np.percentile(I, 95)
            ppI.append((plo,pmid, p95, phi))
    
            print 'Percentiles', plo,phi
            imai = dict(interpolation='nearest', origin='lower',
                       vmin=plo, vmax=phi)
            plt.imshow(I, **imai)
            plt.title(fni)
            ps.savefig()
    
            plt.clf()
            plo,phi = [np.percentile(J, p) for p in [25,99]]
            pmid = np.percentile(J, 50)
            p95 = np.percentile(J, 95)
            ppJ.append((plo,pmid,p95,phi))
            print 'Percentiles', plo,phi
            imaj = dict(interpolation='nearest', origin='lower',
                       vmin=plo, vmax=phi)
            plt.imshow(J, **imaj)
            plt.title(fnj)
            ps.savefig()
            
            plt.clf()
            plt.imshow(K, **imaj)
            plt.title(fnk)
            ps.savefig()
            
            hi,wi = I.shape
            hj,wj = J.shape
            flo,fhi = 0.45, 0.55
            slcI = slice(int(hi*flo), int(hi*fhi)+1), slice(int(wi*flo), int(wi*fhi)+1)
            slcJ = slice(int(hj*flo), int(hj*fhi)+1), slice(int(wj*flo), int(wj*fhi)+1)
    
            x,y = int(wj*(flo+fhi)/2.), int(hj*(flo+fhi)/2.)
            print 'J: x,y =', x,y
            print 'RA,Dec', wcsJ.pixelxy2radec(x,y)
    
            plt.clf()
            plt.imshow(I[slcI], **imai)
            plt.title(fni)
            ps.savefig()
    
            plt.clf()
            plt.imshow(J[slcJ], **imaj)
            plt.title(fnj)
            ps.savefig()
    
            print 'J size', J[slcJ].shape
    
            plt.clf()
            plt.imshow(K[slcJ], **imaj)
            plt.title(fnk)
            ps.savefig()
    
        flo,fhi = 0.45, 0.55
        hi,wi = I.shape
        hj,wj = J.shape
        slcI = slice(int(hi*flo), int(hi*fhi)+1), slice(int(wi*flo), int(wi*fhi)+1)
        slcJ = slice(int(hj*flo), int(hj*fhi)+1), slice(int(wj*flo), int(wj*fhi)+1)
    
        s = II[0][slcI]
        HI,WI = s.shape
        rgbI = np.zeros((HI, WI, 3))
        p0,px,p1,px = ppI[0]
        rgbI[:,:,0] = (II[0][slcI] - p0) / (p1-p0)
        p0,px,p1,px = ppI[1]
        rgbI[:,:,2] = (II[1][slcI] - p0) / (p1-p0)
        rgbI[:,:,1] = (rgbI[:,:,0] + rgbI[:,:,2])/2.
    
        plt.clf()
        plt.imshow(np.clip(rgbI, 0., 1.), interpolation='nearest', origin='lower')
        ps.savefig()
    
        plt.clf()
        plt.imshow(np.sqrt(np.clip(rgbI, 0., 1.)), interpolation='nearest', origin='lower')
        ps.savefig()
    
        s = JJ[0][slcJ]
        HJ,WJ = s.shape
        rgbJ = np.zeros((HJ, WJ, 3))
        p0,px,p1,px = ppJ[0]
        rgbJ[:,:,0] = (JJ[0][slcJ] - p0) / (p1-p0)
        p0,px,p1,px = ppJ[1]
        rgbJ[:,:,2] = (JJ[1][slcJ] - p0) / (p1-p0)
        rgbJ[:,:,1] = (rgbJ[:,:,0] + rgbJ[:,:,2])/2.
    
        plt.clf()
        plt.imshow(np.clip(rgbJ, 0., 1.), interpolation='nearest', origin='lower')
        ps.savefig()
    
        plt.clf()
        plt.imshow(np.sqrt(np.clip(rgbJ, 0., 1.)), interpolation='nearest', origin='lower')
        ps.savefig()
    
        I = (np.sqrt(np.clip(rgbI, 0., 1.))*255.).astype(np.uint8)
        I2 = np.zeros((3,HI,WI))
        I2[0,:,:] = I[:,:,0]
        I2[1,:,:] = I[:,:,1]
        I2[2,:,:] = I[:,:,2]
    
        J = (np.sqrt(np.clip(rgbJ, 0., 1.))*255.).astype(np.uint8)
        J2 = np.zeros((3,HJ,WJ))
        J2[0,:,:] = J[:,:,0]
        J2[1,:,:] = J[:,:,1]
        J2[2,:,:] = J[:,:,2]
    
        fitsio.write('I.fits', I2, clobber=True)
        fitsio.write('J.fits', J2, clobber=True)
    
        for fn in ['I.fits', 'J.fits']:
            os.system('an-fitstopnm -N 0 -X 255 -i %s -p 0 > r.pgm' % fn)
            os.system('an-fitstopnm -N 0 -X 255 -i %s -p 1 > g.pgm' % fn)
            os.system('an-fitstopnm -N 0 -X 255 -i %s -p 2 > b.pgm' % fn)
            os.system('rgb3toppm r.pgm g.pnm b.pnm | pnmtopng > %s' % ps.getnext())
        
        cmd = 'an-fitstopnm -N 0 -X 255 -i I.fits | pnmtopng > %s' % ps.getnext()
        os.system(cmd)
        cmd = 'an-fitstopnm -N 0 -X 255 -i J.fits | pnmtopng > %s' % ps.getnext()
        os.system(cmd)
    
        plt.clf()
        plt.figure(figsize=(6,6))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
        plt.imshow(I, interpolation='nearest', origin='lower')
        ps.savefig()
        plt.imshow(J, interpolation='nearest', origin='lower')
        ps.savefig()
    
    
        # fn = ps.getnext()
        # plt.imsave(fn, (np.sqrt(np.clip(rgbI, 0., 1.))*255.).astype(np.uint8))
        # fn = ps.getnext()
        # plt.imsave(fn, (np.sqrt(np.clip(rgbJ, 0., 1.))*255.).astype(np.uint8))
