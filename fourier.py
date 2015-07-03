import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')

    #matplotlib.rc('font', family='computer modern roman')
    matplotlib.rc('font', family='serif')
    fontsize = 14
    matplotlib.rc('font', size=fontsize)
    matplotlib.rc('text', usetex=True)

import numpy as np
import pylab as plt
import os
import sys

import fitsio

import scipy
import scipy.special

from astrometry.util.fits import *
from astrometry.util.starutil_numpy import *
from astrometry.util.util import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.libkd.spherematch import *
from astrometry.sdss import *

ps = PlotSequence('fourier')
ps.suffixes = ['png','pdf']

def sdss_psf():
    T = fits_table('pobjs.fits')
    print 'Read', len(T), 'objs'
    keep = np.ones(len(T), bool)
    I,J,d = match_radec(T.ra, T.dec, T.ra, T.dec, 30. * 0.396 / 3600., notself=True)
    keep[I] = False
    T.cut(keep)
    print 'Cut to', len(T), 'isolated'
    T.cut(T.probpsf == 1)
    print 'Cut to', len(T), 'point sources'
    T.cut(T.psfflux_r * np.sqrt(T.psffluxivar_r) > 80.)
    print 'Cut to', len(T), 'high S/N'

    F = fits_table('fields.fits')

    sdss = DR10(basedir='data/unzip')

    print sdss.dasurl

    band = 'r'
    rerun = '301'

    dbs = []
    
    for t in T:
        f = F[(F.ramin < t.ra) * (F.ramax > t.ra) * (F.decmin < t.dec) * (F.decmax > t.dec)]
        print 'Found', len(f), 'fields'
        if len(f) == 0:
            continue

        ims = []
        
        for r,c,fnum in zip(f.run, f.camcol, f.field):
            #url = sdss.get_url('frame', r, c, fnum, rerun=rerun, band=band)
            fn = sdss.retrieve('frame', r, c, fnum, band=band, rerun=rerun)
            print 'got', fn

            wcs = Tan(fn)
            ok,x,y = wcs.radec2pixelxy(t.ra, t.dec)
            x -= 1
            y -= 1
            print 'x,y', x,y
            ix = int(np.round(x))
            iy = int(np.round(y))
            S = 25

            img = fitsio.read(fn)
            print 'img', img.shape, img.dtype
            im = img[iy-S:iy+S+1, ix-S:ix+S+1]
            if im.shape != (2*S+1, 2*S+1):
                continue
            H,W = im.shape

            ims.append(im)
            
            # plt.clf()
            # plt.imshow(im, interpolation='nearest', origin='lower')
            # ps.savefig()

            imax = np.argmax(im)
            my,mx = np.unravel_index(imax, im.shape)

            Y = im[my, :]
            X = im[:, mx]
            Fy = np.fft.fft(Y)
            Fx = np.fft.fft(X)
            Sy = np.fft.fftshift(Fy)
            Sx = np.fft.fftshift(Fx)
            Px = Sx.real**2 + Sx.imag**2
            Py = Sy.real**2 + Sy.imag**2
            
            freqs1 = np.fft.fftfreq(len(Y))
            sfreqs1 = np.fft.fftshift(freqs1)
            s0 = np.flatnonzero(sfreqs1 == 0)
            print 's0', s0
            s0 = s0[0]
            
            dbs.append((Px, Py, np.abs(sfreqs1)))

        #break

        n = len(ims)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / float(cols)))
        plt.clf()
        for i,im in enumerate(ims):
            plt.subplot(rows, cols, i+1)
            plt.imshow(im, interpolation='nearest', origin='lower')
        ps.savefig()
        
        plt.clf()
        for Px,Py,sfreq in dbs:
            plt.plot(sfreq, 10.*np.log10(Py / Py[s0]), 'b-', alpha=0.2)
            plt.plot(sfreq, 10.*np.log10(Px / Px[s0]), 'g-', alpha=0.2)
        plt.ylabel('dB')
        plt.ylim(-50, 5)
        ps.savefig()


def gaussian_test():
    c = 20
    W,H = 2*c+1, 2*c+1
    xx,yy = np.meshgrid(np.arange(W), np.arange(H))
    rsq = (xx-c)**2 + (yy-c)**2

    dbs = []
    for sigma in np.arange(0.1, 1.01, 0.05):
        im = np.exp(-0.5 * rsq / sigma**2)

        #F = np.fft.fft2(im)
        #mx = max(np.abs(F.real).max(), np.abs(F.imag).max())

        # plt.clf()
        # plt.subplot(1,2, 1)
        # plt.imshow(F.real, interpolation='nearest', origin='lower', vmin=-mx, vmax=mx)
        # plt.subplot(1,2, 2)
        # plt.imshow(F.imag, interpolation='nearest', origin='lower', vmin=-mx, vmax=mx)
        # plt.suptitle('Sigma = %.2f' % sigma)
        # ps.savefig()

        imax = np.argmax(im)
        my,mx = np.unravel_index(imax, im.shape)

        Y = im[my, :]
        X = im[:, mx]
        Fy = np.fft.fft(Y)
        Fx = np.fft.fft(X)
        Sy = np.fft.fftshift(Fy)
        Sx = np.fft.fftshift(Fx)
        Px = Sx.real**2 + Sx.imag**2
        Py = Sy.real**2 + Sy.imag**2
        freqs1 = np.fft.fftfreq(len(Y))
        sfreqs1 = np.fft.fftshift(freqs1)
        s0 = np.flatnonzero(sfreqs1 == 0)
        s0 = s0[0]
        dbs.append((Px, Py, np.abs(sfreqs1), sigma))

    plt.clf()
    for Px,Py,sfreq,sigma in dbs:
        plt.plot(sfreq, 10.*np.log10(Py / Py[s0]), 'b-', alpha=0.2)
        plt.plot(sfreq, 10.*np.log10(Px / Px[s0]), 'g-', alpha=0.2)
        plt.text(sfreq[-1], 10.*np.log10(Px[-1] / Px[s0]), '%.2f (FWHM %.2f)' % (sigma, sigma * 2.35), ha='right', va='center')
    plt.ylabel('dB')
    plt.ylim(-40, 3)
    #plt.axhline(-3, color='r', alpha=0.5)
    ps.savefig()


def airy_test():

    # How much to compress the Airy profile so that it is just
    # well-sampled (with unit pixels)
    scale = np.pi / 2.
    x = np.linspace(-5, 5, 500)
    y = (2. * scipy.special.j1(x * scale) / (x * scale))**2

    # Its half-width at half-max
    hwhm = (1./scale * 1.61633)
    print 'hwhm', hwhm
    
    # Gaussian with same HWHM
    sigma = 2. * hwhm / 2.35
    print 'sigma', sigma
    g = np.exp(-0.5 * x**2 / sigma**2)
    
    plt.clf()
    plt.plot(x, y, 'b-')
    plt.plot(x, g, 'r-', lw=2, alpha=0.5)
    plt.axhline(0.5, color='r')
    plt.axvline(1./scale * 1.61633, color='b', lw=3, alpha=0.5)
    plt.axvline(sigma * 2.35 / 2., color='r')
    ps.savefig()

    c = 200
    W,H = 2*c+1, 2*c+1
    xx,yy = np.meshgrid(np.arange(W), np.arange(H))
    rr = np.hypot(xx-c, yy-c)

    dbs = []
    for scale in np.arange(1.0, 2.01, 0.1):
        im = (2. * scipy.special.j1(rr * scale) / (rr * scale))**2
        im[rr == 0] = 1.
        #(im,nil,nil,nil) = scipy.special.airy(rr * scale)

        F = np.fft.fft2(im)
        F = np.fft.fftshift(F)
        #F = np.fft.rfft2(im)
        mx = max(np.abs(F.real).max(), np.abs(F.imag).max())

        # plt.clf()
        # plt.subplot(2,2,1)
        # plt.imshow(im, interpolation='nearest', origin='lower')
        # plt.subplot(2,2,2)
        # plt.plot(rr.ravel(), np.hypot(F.real, F.imag).ravel(),
        #          'b.', alpha=0.1)
        # plt.axvline((W*scale) / np.pi, color='r')
        # plt.subplot(2,2,3)
        # plt.imshow(F.real, interpolation='nearest', origin='lower',
        #            vmin=-mx, vmax=mx)
        # plt.subplot(2,2,4)
        # plt.imshow(F.imag, interpolation='nearest', origin='lower',
        #            vmin=-mx, vmax=mx)
        # ps.savefig()
        
        imax = np.argmax(im)
        my,mx = np.unravel_index(imax, im.shape)

        Y = im[my, :]
        X = im[:, mx]
        Fy = np.fft.fft(Y)
        Fx = np.fft.fft(X)
        Sy = np.fft.fftshift(Fy)
        Sx = np.fft.fftshift(Fx)
        Px = Sx.real**2 + Sx.imag**2
        Py = Sy.real**2 + Sy.imag**2
        freqs1 = np.fft.fftfreq(len(Y))
        sfreqs1 = np.fft.fftshift(freqs1)
        s0 = np.flatnonzero(sfreqs1 == 0)
        s0 = s0[0]
        dbs.append((X, Y, Px, Py, np.abs(sfreqs1), scale))


    plt.clf()
    v = np.linspace(0., 2., 500)
    plt.plot(v, 2./np.pi * (np.arccos(v) - v*np.sqrt(1-v**2)), 'b-')
    ps.savefig()
        
    plt.clf()
    for X,Y,Px,Py,sfreq,scale in dbs:
        plt.plot(X, 'b-', alpha=0.2)
        plt.plot(Y, 'g-', alpha=0.2)
    ps.savefig()
        
    plt.clf()
    for X,Y,Px,Py,sfreq,scale in dbs:
        plt.plot(sfreq, 10.*np.log10(Py / Py[s0]), 'b-', alpha=0.2)
        plt.plot(sfreq, 10.*np.log10(Px / Px[s0]), 'g-', alpha=0.2)
        plt.text(sfreq[-1], 10.*np.log10(Px[-1] / Px[s0]), '%.2f' % (scale), ha='right', va='center')
        #plt.axvline(scale / np.pi, color='r')
    plt.ylabel('dB')
    #ps.savefig()
    plt.ylim(-50, 5)
    ps.savefig()

    

def wise_sources(bands, pred=False):
    S = 25

    if pred:
        psfs = wise_psf(bands=bands, fftcentral=S*2+1, noplots=True)
        print 'psfs:', len(psfs)
        (Sx,Sy,Px,Py,sf) = psfs[0]
        print 'sx:', len(Sx)
        print 'sf', sf

        pitch = 0.1
        print 'Pitch', pitch
        print 'lanczos_test...'
        lan3 = lanczos_test(pitch=pitch, srange=(-S-0.5+pitch/2., S+0.5))#, noplots=True)
        print 'lanczos_test finished.'
        (Slan,Plan,sflan) = lan3[0]
        sflan /= pitch
        sflan = sflan[:S+1]
        Plan = Plan[:S+1]
        Slan = Slan[:S+1]
        print 'lan3:', len(Slan)
        print 'sflan:', sflan
        print 'Plan:', Plan


    ps.basefn = 'wisestars'
    ps.skipto(0)

    wfn = 'wbox.fits'
    if os.path.exists(wfn):
        W = fits_table(wfn)
        print 'Read', len(W), 'from', wfn
    else:
        from wisecat import wise_catalog_radecbox
        W = wise_catalog_radecbox(150., 200., 45., 47.)
        print 'Got', len(W), 'WISE sources'
        W.writeto(wfn)

    from unwise_coadd import get_l1b_file, wisedir, zeropointToScale
    
    # plt.clf()
    # plt.hist(W.w1snr, 50, histtype='step', color='b')
    # plt.hist(W.w1snr[W.w1sat == 0], 50, histtype='step', color='r')
    # ps.savefig()
    
    # Find isolated stars...
    I,J,d = match_radec(W.ra, W.dec, W.ra, W.dec, 20. * 2.75 / 3600., notself=True)
    keep = np.ones(len(W), bool)
    keep[J] = False
    print np.sum(keep), 'of', len(W), 'have no neighbors'

    for iband,band in enumerate(bands):

        W.snr = W.get('w%isnr'%band)
        W.sat = W.get('w%isat' % band)
        Wi = W[keep * (W.sat == 0)]
        print len(Wi), 'after cuts'
        #(W.snr > 20) *         
        # plt.clf()
        # plt.hist(Wi.snr, 50, histtype='step', color='m')
        Wi.cut(Wi.ext_flg == 0)
        print 'ext_flg:', len(Wi)
        Wi.cut(Wi.na == 0)
        print 'na:', len(Wi)
        Wi.cut(Wi.nb == 1)
        print 'nb:', len(Wi)
        # plt.hist(Wi.snr, 50, histtype='step', color='k')
        # ps.savefig()

        T = Wi[np.argsort(-Wi.snr)]
        T.cut(T.snr > 25.)
        print 'SN cut:', len(T)
        
        #T = Wi[np.argsort(-Wi.snr)[:25]]
        #print 'Coadd S/N:', T.snr
        
        # band = 1
        # T = fits_table('w%i.fits' % band)
        # 
        # plt.clf()
        # n,b,p = plt.hist(T.get('w%impro' % band), 50, histtype='step', color='b')
        # plt.xlabel('w%impro' % band)
        # 
        # print 'blend_ext_flags:', ['0x%x' % x for x in np.unique(T.blend_ext_flags)]
        # 
        # T.cut(T.blend_ext_flags & 0xf == 1)
        # print 'Cut to', len(T), 'on blend'
        # T.cut(T.get('w%isat' % band) == 0)
        # print 'Cut to', len(T), 'on sat'
        # T.cut(T.get('w%iflg' % band) == 0)
        # print 'Cut to', len(T), 'on flg'
        # 
        # plt.hist(T.get('w%impro' % band), histtype='step', color='r', bins=b)
        # ps.savefig()
        
        dbs = []

        l1dbs = []
        
        #print 'Coadds', T.coadd_id
        for t in T:
            print 'coadd', t.coadd_id
            #print 'RA,Dec', t.ra, t.dec
            coadd = t.coadd_id.replace('_ab41', '').strip()
            for base in ['data/unwise', 'data/unwise-nersc']:
                dirnm = os.path.join(base, coadd[:3], coadd)
                fn = os.path.join(dirnm, 'unwise-%s-w%i-img-m.fits' % (coadd, band))
                if os.path.exists(fn):
                    break
            img = fitsio.read(fn)
            #print 'img', img.shape, img.dtype
        
            wcs = Tan(fn)
            ok,x,y = wcs.radec2pixelxy(t.ra, t.dec)
            x -= 1
            y -= 1
            #print 'x,y', x,y
            ix = int(np.round(x))
            iy = int(np.round(y))
        
            im = img[iy-S:iy+S+1, ix-S:ix+S+1]
            if im.shape != (2*S+1, 2*S+1):
                continue
            imH,imW = im.shape
            imax = np.argmax(im)
            my,mx = np.unravel_index(imax, im.shape)
            #print 'max pixel:', my,mx, im[my,mx]
            #print 'center pixel:', imH/2,imW/2, im[imH/2,imW/2]
            if my != imH/2 or mx != imW/2:
                continue
        
            # F = np.fft.fft2(im)
            # S = np.fft.fftshift(F)
            # P = S.real**2 + S.imag**2
            # P /= P.max()
            # freqs = np.fft.fftfreq(W)
            # sfreqs = np.fft.fftshift(freqs)
            # ff = np.hypot(sfreqs.reshape(1,W), sfreqs.reshape(W,1))
            # print 'ff', ff.shape
            #plt.clf()
            #plt.plot(ff.ravel(), P.ravel(), 'b.', alpha=0.1)
            #ps.savefig()
        
            Y = im[imH/2, :]
            X = im[:, imW/2]
            Fy = np.fft.fft(Y)
            Fx = np.fft.fft(X)
            Sy = np.fft.fftshift(Fy)
            Sx = np.fft.fftshift(Fx)
            Px = Sx.real**2 + Sx.imag**2
            Py = Sy.real**2 + Sy.imag**2

            # Grab "blank" lines nearby to assess noise
            n = im[imH/4, :]
            Fn = np.fft.fft(n)
            Sn = np.fft.fftshift(Fn)
            Pn = Sn.real**2 + Sn.imag**2
            
            freqs1 = np.fft.fftfreq(len(Y))
            sfreqs1 = np.fft.fftshift(freqs1)
            s0 = np.flatnonzero(sfreqs1 == 0)
            s0 = s0[0]
            s1 = s0
            dbs.append((Sx[s1:], Sy[s1:], Px[s1:], Py[s1:], sfreqs1[s1:], Pn[s1:]))

            print 'sx', len(Sx[s1:])
            print 'sf', sfreqs1[s1:]

            continue

            fn = os.path.join(dirnm, 'unwise-%s-w%i-frames.fits' % (coadd, band))
            T = fits_table(fn)
            print 'Read', len(T), 'from', fn
            T.cut(T.included > 0)
            T.cut(T.use > 0)
            print 'Cut to', len(T), 'used'
            T.cut(T.coextent[:,0] < ix - S)
            T.cut(T.coextent[:,1] > ix + S)
            T.cut(T.coextent[:,2] < iy - S)
            T.cut(T.coextent[:,3] > iy + S)
            print 'Cut to', len(T), 'containing target pixels'

            #ims = []

            #for f in T[:10]:
            for f in T:
                fn = get_l1b_file(wisedir, f.scan_id, f.frame_num, band)
                print 'frame', fn
                img,hdr = fitsio.read(fn, header=True)
                img -= f.sky1

                gain = hdr.get('FEBGAIN')
                print 'Gain:', gain

                sig1 = np.sqrt(1./f.weight)
                zpscale = 1. / zeropointToScale(f.zeropoint)
                sig1 /= zpscale
                print 'Sigma1:', sig1

                print 'Sky:', f.sky1

                wcs = Sip(fn)

                mask = fitsio.read(fn.replace('-int-', '-msk-') + '.gz')

                #print 'RA,Dec', t.ra, t.dec
                ok,x,y = wcs.radec2pixelxy(t.ra, t.dec)
                #print 'x,y', x,y
                x -= 1
                y -= 1
                fx = int(np.round(x))
                fy = int(np.round(y))
                im = img[fy-S:fy+S+1, fx-S:fx+S+1]
                if im.shape != (2*S+1, 2*S+1):
                    continue
                imH,imW = im.shape

                mask = mask[fy-S:fy+S+1, fx-S:fx+S+1]
                ok = (mask == 0)
                if not patch_image(im, ok):
                    print 'Patching failed'
                    continue

                Y = im[imH/2, :]
                X = im[:, imW/2]
                Fy = np.fft.fft(Y)
                Fx = np.fft.fft(X)
                Sy = np.fft.fftshift(Fy)
                Sx = np.fft.fftshift(Fx)
                Px = Sx.real**2 + Sx.imag**2
                Py = Sy.real**2 + Sy.imag**2

                freqs1 = np.fft.fftfreq(len(Y))
                sfreqs1 = np.fft.fftshift(freqs1)
                s0 = np.flatnonzero(sfreqs1 == 0)
                s0 = s0[0]

                # Grab "blank" lines nearby to assess noise
                n = im[imH/4, :]
                Fn = np.fft.fft(n)
                Sn = np.fft.fftshift(Fn)
                Pn = Sn.real**2 + Sn.imag**2

                l1dbs.append((Px[s0+1:], Py[s0+1:], sfreqs1[s0+1:], Pn[s0+1:]))

                #ims.append(im)

            # plt.clf()
            # for i,im in enumerate(ims[:9]):
            #     plt.subplot(3,3,1+i)
            #     plt.imshow(im, interpolation='nearest', origin='lower')
            # ps.savefig()

        print len(dbs), 'coadds'
        print len(l1dbs), 'L1bs'


        plt.figure(figsize=(4,4))
        left = 0.17
        right = 0.03
        bot = 0.12
        top = 0.03
        plt.subplots_adjust(left=left, right=1.-right, bottom=bot, top=1.-top)


        if pred:
            (Sxpsf, Sypsf, Pxpsf, Pypsf, sfpsf) = psfs[iband]
            plt.clf()
            Spsf = (Sxpsf + Sypsf) / 2.
            Ppsf = Spsf.real**2 + Spsf.imag**2
            plt.plot(sfpsf, 10.*np.log10(Pxpsf / Pxpsf[0]), 'r-')
            plt.plot(sfpsf, 10.*np.log10(Pypsf / Pypsf[0]), 'm-')
            plt.plot(sfpsf, 10.*np.log10(Ppsf / Ppsf[0]), '-', color=(1.,0.6,0))
            plt.plot(sflan, 10.*np.log10(Plan / Plan[0]), 'k-')
            #print 'sflan', sflan
            #print 'Plan', Plan
            #print 'db', 10.*np.log10(Plan / Plan[0])
            for Sx,Sy,Px,Py,sfreq,Pn in dbs:
                Sm = (Sx + Sy) / 2.
                P = Sm.real**2 + Sm.imag**2
                plt.plot(sfreq, 10.*np.log10(P / P[0]), 'b-', alpha=0.1)
            Spred = Spsf * Slan
            Ppred = Spred.real**2 + Spred.imag**2
            plt.plot(sfreq, 10.*np.log10(Ppred / Ppred[0]), 'g-')
            plt.ylabel('dB')
            plt.ylim(-40, 3)
            plt.title('PSF')
            ps.savefig()

        plt.clf()
        for Sx,Sy,Px,Py,sfreq,Pn in dbs:
            plt.plot(sfreq, 10.*np.log10(Py / Py[0]), 'b-', alpha=0.2)
            plt.plot(sfreq, 10.*np.log10(Px / Px[0]), 'g-', alpha=0.2)
            plt.plot(sfreq, 10.*np.log10(Pn / np.sqrt(Px[0]*Py[0])), 'r-', alpha=0.2)
        plt.ylabel('dB')
        plt.ylim(-40, 3)
        plt.title('Isolated sources in coadd: W%i' % band)
        ps.savefig()
        
        xx = np.zeros((len(dbs), len(dbs[0][0])))
        yy = np.zeros((len(dbs), len(dbs[0][0])))
        nn = np.zeros((len(dbs), len(dbs[0][0])))
        mm = np.zeros((len(dbs), len(dbs[0][0])))
        
        for i,(Sx,Sy,Px,Py,sfreq,Pn) in enumerate(dbs):
            xx[i,:] = 10.*np.log10(Px / Px[0])
            yy[i,:] = 10.*np.log10(Py / Py[0])
            nn[i,:] = 10.*np.log10(Pn / np.sqrt(Px[0]*Py[0]))
            Sm = (Sx + Sy) / 2.
            P = Sm.real**2 + Sm.imag**2
            mm[i,:] = 10.*np.log10(P / P[0])

        if pred:
            plt.clf()
            med = np.median(mm, axis=0)
            plt.plot(sfreq, med, 'g-')
            plt.plot(sfpsf, 10.*np.log10(Ppsf / Ppsf[0]), '-', color=(1.,0.6,0))
            plt.plot(sflan, 10.*np.log10(Plan / Plan[0]), 'k-')
            plt.plot(sfreq, 10.*np.log10(Ppred / Ppred[0]), 'r-')
            plt.ylabel('dB')
            plt.ylim(-40, 3)
            plt.title('PSF')
            ps.savefig()

        plt.clf()
        plt.plot(sfreq, np.median(xx, axis=0), 'g-')
        plt.plot(sfreq, np.median(yy, axis=0), 'b-')
        plt.plot(sfreq, np.median(nn, axis=0), 'r-')
        plt.title('Isolated sources in coadd: W%i median' % band)
        #plt.xlabel('frequency (cycles/pixel)')
        plt.ylabel('dB')
        plt.ylim(-40, 3)
        ps.savefig()

        print 'Freqs:', sfreq

        if len(l1dbs) == 0:
            continue
        
        plt.clf()
        for Px,Py,sf,noise in l1dbs:
            plt.plot(sf, 10.*np.log10(Px / Px[0]), 'g-', alpha=0.2)
            plt.plot(sf, 10.*np.log10(Py / Py[0]), 'b-', alpha=0.2)
            plt.plot(sf, 10.*np.log10(noise / Py[0]), 'r-', alpha=0.2)
            plt.plot(sf, 10.*np.log10(noise / Px[0]), 'r-', alpha=0.2)
        plt.ylabel('dB')
        plt.ylim(-40, 3)
        plt.title('Isolated sources in L1bs: W%i' % band)
        ps.savefig()

        xx = np.zeros((len(l1dbs), len(l1dbs[0][0])))
        yy = np.zeros((len(l1dbs), len(l1dbs[0][0])))
        zz = np.zeros((len(l1dbs), len(l1dbs[0][0])))
        zz2 = np.zeros((len(l1dbs), len(l1dbs[0][0])))
        for i,(Px,Py,sfreq,noise) in enumerate(l1dbs):
            xx[i,:] = 10.*np.log10(Px / Px[0])
            yy[i,:] = 10.*np.log10(Py / Py[0])
            zz[i,:] = 10.*np.log10(noise / Py[0])
            zz2[i,:] = 10.*np.log10(noise / Px[0])

        plt.clf()
        plt.plot(sfreq, np.median(xx, axis=0), 'g-')
        plt.plot(sfreq, np.median(yy, axis=0), 'b-')
        plt.plot(sfreq, np.median(zz, axis=0), 'r-')
        plt.plot(sfreq, np.median(zz2, axis=0), 'r-')
        plt.title('Isolated sources in L1bs: W%i median' % band)
        #plt.xlabel('frequency (cycles/pixel)')
        plt.ylabel('dB')
        plt.ylim(-40, 3)
        ps.savefig()

def wise_psf(bands = [1,2,3,4], fftcentral=None, noplots=False, bw=False):
    colors = ['b', 'g', 'r', 'm']

    dbs = []

    slices = []
    centrals = []
    
    for band in bands:
        if band in [1,2,3]:
            # im = fitsio.read('wise-psf-w%i-507.5-507.5.fits' % band)
            im = fitsio.read('wise-psf-w%i-507.5-507.5-bright.fits' % band)
        else:
            # im = fitsio.read('wise-psf-w%i-253.5-253.5.fits' % band)
            im = fitsio.read('wise-psf-w%i-253.5-253.5-bright.fits' % band)
        im /= im.max()
        print 'Image', im.shape, im.dtype
        H,W = im.shape
        assert(H == W)
        print 'center / max:', im[H/2, W/2]
        assert(im[H/2, W/2] == 1.0)
        c = W/2

        # plt.clf()
        # plt.imshow(np.log10(im), interpolation='nearest', origin='lower',
        #            vmin=-8, vmax=0)
        # ps.savefig()

        nclip = (W - 251) / 2
        slc = slice(nclip, -nclip)
        central = im[slc,slc]
        centrals.append(central)
        ch,cw = central.shape
        cc = ch/2
        slices.append((central[cc,:], central[:,cc]))

        if fftcentral is not None:
            nclip = (W - fftcentral) / 2
            im = im[nclip:-nclip, nclip:-nclip]
            H,W = im.shape

        Y = im[H/2, :]
        X = im[:, W/2]
        Fy = np.fft.fft(Y)
        Fx = np.fft.fft(X)
        Sy = np.fft.fftshift(Fy)
        Sx = np.fft.fftshift(Fx)
        Px = Sx.real**2 + Sx.imag**2
        Py = Sy.real**2 + Sy.imag**2

        freqs1 = np.fft.fftfreq(len(Y))
        sfreqs1 = np.fft.fftshift(freqs1)
        s0 = np.flatnonzero(sfreqs1 == 0)
        s0 = s0[0]

        Px = Px[s0:]
        Py = Py[s0:]
        sf = sfreqs1[s0:]
        Sx = Sx[s0:]
        Sy = Sy[s0:]
        dbs.append((Sx, Sy, Px, Py, sf))

        # plt.clf()
        # plt.plot(sf, 10.*np.log10(Px / Px[0]), 'g-')
        # plt.plot(sf, 10.*np.log10(Py / Py[0]), 'b-')
        # plt.ylabel('dB')
        # plt.ylim(-40, 3)
        # plt.title('PSF models: W%i' % band)
        # ps.savefig()

    if noplots:
        return dbs

    cmap = 'jet'
    ps.basefn = 'wisepsf'
    if bw:
        ps.pattern = 'wisepsf-%s-bw.%s'
        #cmap = 'gray'
        cmap = antigray
        colors = ['k']*4

    plt.figure(figsize=(4,4))
    if True:
        vmin=-7
        vmax=0
        # bot = 0.05
        # top = 0.02
        # left = 0.1
        # right = 0.02
        bot = 0.12
        top = 0.03
        left = 0.12
        right = 0.03
        
        # colorbar cbar
        plt.figure(figsize=(0.75,4))
        plt.clf()
        ax = plt.gca()
        print 'ax pos', ax.get_position()
        ax.set_position([0.01, bot, 0.35, 1.-bot-top])
        #cax,kw = matplotlib.colorbar.make_axes(ax, fraction=0.9)
        #fraction 0.15; fraction of original axes to use for colorbar
        #pad 0.05 if vertical, 0.15 if horizontal; fraction of original axes between colorbar and new image axes
        #shrink 1.0; fraction by which to shrink the colorbar
        #aspect 20; ratio of long to short dimensions
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
        tv = np.arange(vmin, vmax+1)
        cb.set_ticks(tv, update_ticks=False)
        tt = ['$10^{%i}$' % t for t in tv]
        tt[-1] = '$1$'
        cb.set_ticklabels(tt, update_ticks=False)
        cb.update_ticks()
        ps.savefig()

        plt.figure(figsize=(4,4))
        plt.subplots_adjust(left=left, right=1.-right, bottom=bot, top=1.-top)
        for central in centrals:
            ch,cw = central.shape
            cc = ch/2
            plt.clf()
            plt.imshow(np.log10(np.maximum(10.**(vmin-1), central)),
                       interpolation='nearest', origin='lower',
                       vmin=vmin, vmax=vmax, cmap=cmap,
                       extent=[-cc,cc,-cc,cc])
            ps.savefig()



    if True:
        left = 0.17
        right = 0.03
        bot = 0.12
        top = 0.03
        plt.subplots_adjust(left=left, right=1.-right, bottom=bot, top=1.-top)

        for i,(sx,sy) in enumerate(slices):
            band = bands[i]
            plt.clf()
            S = len(sx)
            mid = S/2
            #vmin,vmax = -6,0
            #vmin,vmax = -3,0
            x = np.arange(len(sx)) - mid
            pa = plt.semilogy(x, sx, '-', color=colors[i])
            pb = plt.plot(x, sy, '-', color=colors[i], lw=3, alpha=0.5)
            xlo,xhi = -10, 10
            xx = np.linspace(xlo, xhi, 500)
            scale = np.pi / 2.
            yy = (2. * scipy.special.j1(xx * scale) / (xx * scale))**2
            yy[xx == 0] = 1.
            pc = plt.plot(xx, yy, '--', color='k')
    
            plt.figlegend((pb[0], pa[0], pc[0]),
                       ('W%i(y)' % band, 'W%i(x)' % band, 'Airy*'),
                       loc='upper right', fontsize=fontsize)
    
            plt.xlim(xlo, xhi)
            plt.ylim(1e-3/1.1, 1.1)
            plt.xlabel('Pixel')
            plt.ylabel('PSF profile')
            ps.savefig()


    if True:
        left = 0.17
        right = 0.03
        bot = 0.12
        top = 0.03
        plt.subplots_adjust(left=left, right=1.-right, bottom=bot, top=1.-top)
    
        # How much to compress the Airy profile so that it is just
        # well-sampled (with unit pixels)
        scale = np.pi / 2.
        # Its half-width at half-max
        hwhm = (1./scale * 1.61633)
        # Gaussian with same HWHM
        sigma = 2. * hwhm / 2.35
        print 'sigma', sigma
        x = np.linspace(-10, 10, 1000)
        dx = x[1]-x[0]
        g = np.exp(-0.5 * x**2 / sigma**2)

        airy = (2. * scipy.special.j1(x * scale) / (x * scale))**2

        Fg = np.fft.fft(g)
        Sg = np.fft.fftshift(Fg)
        Pg = Sg.real**2 + Sg.imag**2
        freqs1 = np.fft.fftfreq(len(g))
        sfg = np.fft.fftshift(freqs1)
        s0 = np.flatnonzero(sfg == 0)

        Fa = np.fft.fft(airy)
        Sa = np.fft.fftshift(Fa)
        Pa = Sa.real**2 + Sa.imag**2
        
        Pg = Pg[s0:]
        Pa = Pa[s0:]
        sfg = sfg[s0:]
        print 'dx', dx
        sfg /= dx
        
        for i,(nil,nil,Px,Py,sf) in enumerate(dbs):
            band = bands[i]

            plt.clf()
            pa = plt.plot(sf, 10.*np.log10(Px / Px[0]), '-', color=colors[i])
            pb = plt.plot(sf, 10.*np.log10(Py / Py[0]), '-', color=colors[i],
                          lw=3, alpha=0.5)
            pc = plt.plot(sfg, 10.*np.log10(Pg / Pg[0]), 'k--')
            pd = plt.plot(sfg, 10.*np.log10(Pa / Pa[0]), 'k:')

            lp = (pb[0], pa[0], pc[0], pd[0])
            lt = ('W%i (y)' % band, 'W%i (x)' % band,
                  'Gaussian*', 'Airy*')

            if band == 1:
                pitch = 0.001
                lan3 = lanczos_test(noplots=True, pitch=pitch)
                (Slan,Plan,sflan) = lan3[0]
                sflan /= pitch
                pe = plt.plot(sflan, 10.*np.log10(Plan / Plan[0]), 'k-.')
                lp = (pe[0],) + lp
                lt = ('Lanczos-3',) + lt

            plt.ylabel('Power in Fourier component (dB)')
            plt.ylim(-40, 3)
            plt.xlabel('Frequency (cycles/pixel)')
            plt.legend(lp, lt, loc='lower left', fontsize=fontsize)
            plt.xlim(0, 0.5)
            ps.savefig()

        


def lanczos_test(pitch=0.001, noplots=False, srange=(-10,10)):

    pows = []
    #for subx in np.linspace(-0.5, 0.5, 11):
    for subx in [0]:
        lo,hi = srange
        dx = np.arange(lo, hi, pitch).astype(np.float32) + 3 + subx
        ix = np.round(dx).astype(np.int32)
        ddx = (dx - ix).astype(np.float32)
        outimg = np.zeros_like(dx)
        inimg = np.zeros((1,7), np.float32)
        inimg[0,3] = 1.
    
        lanczos3_interpolate(ix, np.zeros_like(ix),
                             ddx, np.zeros_like(ddx),
                             [outimg], [inimg])
    
        # plt.clf()
        # plt.plot(outimg)
        # ps.savefig()
    
        F = np.fft.fft(outimg)
        S = np.fft.fftshift(F)
        P = S.real**2 + S.imag**2
        freqs1 = np.fft.fftfreq(len(outimg))
        sfreqs1 = np.fft.fftshift(freqs1)
        s0 = np.flatnonzero(sfreqs1 == 0)
        s0 = s0[0]
        s1 = s0# + 1
        pows.append((S[s1:], P[s1:], sfreqs1[s1:]))

    if noplots:
        return pows

    plt.figure(figsize=(4,4))
    left = 0.17
    right = 0.03
    bot = 0.12
    top = 0.03
    plt.subplots_adjust(left=left, right=1.-right, bottom=bot, top=1.-top)

    plt.clf()
    for nil,P,sfreq in pows:
        plt.plot(sfreq / pitch, 10.*np.log10(P / P[0]), 'b-')
    #plt.plot(sfreq / pitch, 10.*np.log10(P / P[0]), 'b-')
    plt.ylabel('Power in Fourier component (dB)')
    plt.ylim(-40, 3)
    #plt.title('Lanczos3')
    plt.xlim(0, 0.5)
    ps.savefig()

    # plt.clf()
    # for S,P,sfreq in pows:
    #     pp = int(np.round(1./pitch))
    #     plt.plot(sfreq[::pp], 10.*np.log10(P[::pp] / P[0]), 'b-')
    #     print 'freqs', sfreq[::pp]
    # plt.ylabel('Power in Fourier component (dB)')
    # plt.ylim(-40, 3)
    # plt.xlim(0, 0.5)
    # ps.savefig()

    return pows

#sdss_psf()
#airy_test()
#sys.exit(0)
#gaussian_test()
#lanczos_test()
#wise_sources([1,2,3,4])
#wise_psf()
wise_psf(bw=True)

sys.exit(0)


# ra,dec = np.meshgrid(np.arange(360), np.arange(-90,91))
# T = fits_table()
# T.ra = ra.ravel()
# T.dec = dec.ravel()
# T.l,T.b = radectolb(T.ra, T.dec)
# T.u,T.v = radectoecliptic(T.ra, T.dec)
# T.cut(np.abs(T.b) > 40)
# T.cut(np.abs(T.v) > 15)
# plt.clf()
# plt.plot(T.ra, T.dec, 'b.')
# plt.xlabel('RA')
# plt.ylabel('Dec')
# for r in np.arange(0, 360, 10):
#     plt.axvline(r, color='k')
# for d in np.arange(-90, 90, 5):
#     plt.axhline(d, color='k')
# ps.savefig()


xx = np.zeros((len(dbs), len(dbs[0][0])))
yy = np.zeros((len(dbs), len(dbs[0][0])))
for i,(Px,Py,sfreq) in enumerate(dbs):
    xx[i,:] = 10.*np.log10(Px / Px[s0])
    yy[i,:] = 10.*np.log10(Py / Py[s0])

plt.clf()
plt.plot(sfreq, np.median(xx, axis=0), 'g-')
plt.plot(sfreq, np.median(yy, axis=0), 'b-')
plt.ylabel('dB')
plt.xlabel('frequency (cycles/pixel)')
ps.savefig()









sys.exit(0)

    
