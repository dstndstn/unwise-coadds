#! /usr/bin/env python

import os
import sys

if __name__ == '__main__':
    d = os.environ.get('PBS_O_WORKDIR')
    if d is not None:
        os.chdir(d)
        sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import fitsio

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.blind.plotstuff import *
from astrometry.libkd.spherematch import *


def image_way():

    plt.figure(figsize=(8,4))

    spa = dict(left=0.005, right=.995, bottom=0.005, top=0.995)
    plt.subplots_adjust(**spa)

    #W,H = 4000,2000
    W,H = 1000,500
    plot = Plotstuff(size=(W,H), outformat='png')
    plot.wcs = anwcs_create_allsky_hammer_aitoff(180., 0., W, H)
    out = plot.outline
    out.stepsize = 2000
    out.fill = 1
    
    wcs = Tan()
    out.wcs = anwcs_new_tan(wcs)
    wcs = anwcs_get_sip(out.wcs)
    wcs = wcs.wcstan

    totals = [np.zeros((H,W), int) for b in range(4)]

    count = np.zeros((H,W), np.int16)

    #ps = PlotSequence('cov', format='%04i')
    ps = PlotSequence('cov2', format='%04i')

    metadatadir = '/project/projectdirs/cosmo/data/wise/merge/merge_p1bm_frm/'
    for phase,metafn in enumerate([
            'WISE-l1b-metadata-4band.fits',
            'WISE-l1b-metadata-3band.fits',
            'WISE-l1b-metadata-2band.fits',
            'WISE-l1b-metadata-neowiser.fits',
            'WISE-l1b-metadata-neowiser2.fits',
            ]):
        fn = os.path.join(metadatadir, metafn)
        bb = [1]
        for band in bb:

            ofn = 'cov-phase%i-b%i.fits' % (phase, band)
            if os.path.exists(ofn):
                print 'Exists:', ofn
                count = fitsio.read(ofn)
                print 'Read', count.shape, count.dtype, 'max', count.max()
                totals[band-1] += count

                plt.clf()
                plt.imshow(count, interpolation='nearest', origin='lower',
                           vmin=0, vmax=100, cmap='gray')
                plt.colorbar()
                ps.savefig()
                continue

            cols = [('w%i'%band)+c for c in
                    ['crval1','crval2','crpix1','crpix2',
                     'cd1_1','cd1_2','cd2_1','cd2_2', 'naxis1','naxis2',]]
            cols += ['scan_id']
            print 'Reading', fn
            T = fits_table(fn, columns=cols)
            print 'Read', len(T), 'from', fn
            arrs = [T.get(c).astype(float) for c in cols[:10]]

            plot.clear()
            plot.color = 'white'
            plot.alpha = 1./255.
            plot.op = CAIRO_OPERATOR_ADD

            scans = np.unique(T.scan_id)
            print len(scans), 'scans'

            marg = 0.04
            slc = (slice(int(H*marg), int(H*(1-marg))), slice(int(W*marg), int(W*(1-marg))))

            for si,scan in enumerate(scans):
                I = np.flatnonzero(T.scan_id == scan)
                print len(I), 'for scan', scan
                for i in I:
                    if arrs[-1][i] == -1:
                        continue
                    wcs.set(*[a[i] for a in arrs])
                    plot.plot('outline')

                #if si % 10 == 0:
                if si % 20 == 0:
                    im = plot.get_image_as_numpy()
                    print 'max:', im[:,:,0].max()
                    count += im[:,:,0]
                    del im
                    print 'total max:', count.max()
                    plot.clear()

                    plt.clf()
                    # plt.imshow(np.log10(np.maximum(0.1, count)), interpolation='nearest',
                    #           origin='lower', cmap='hot', vmin=-1, vmax=3)
                    plt.imshow(count[slc], interpolation='nearest', #origin='lower',
                               cmap='hot', vmin=0, vmax=50)
                    plt.xticks([]); plt.yticks([])
                    #plt.colorbar()
                    ps.savefig()


            # N = len(T)
            # for i in xrange(N):
            #     if arrs[-1][i] == -1:
            #         continue
            #     wcs.set(*[a[i] for a in arrs])
            #     plot.plot('outline')
            # 
            #     if i and i % 10000 == 0 or i == N-1:
            #         print 'exposure', i, 'of', N
            #         im = plot.get_image_as_numpy()
            #         print 'max:', im[:,:,0].max()
            #         count += im[:,:,0]
            #         del im
            #         print 'total max:', count.max()
            #         plot.clear()
            #         
            # fitsio.write(ofn, count, clobber=True)
            # print 'Wrote', ofn
            # 
            # totals[band-1] += count
            # 
            # plt.clf()
            # plt.imshow(count, interpolation='nearest', origin='lower',
            #            vmin=0, vmax=100, cmap='gray')
            # plt.colorbar()
            # ps.savefig()
            
            del T
            del arrs

    ##
    return


    M = reduce(np.logical_or, [t > 0 for t in totals])

    for tot in totals:
        plt.clf()
        plt.imshow(tot, interpolation='nearest', origin='lower',
                   vmin=0, vmax=100, cmap='gray')
        plt.colorbar()
        ps.savefig()

    plt.clf()
    mx = 60
    for tot,cc in zip(totals, 'bgrm'):
        plt.hist(np.minimum(tot[M], mx), range=(0,mx),
                 bins=mx+1, histtype='step', color=cc)
    ps.savefig()

        
def healpix_way():
    Nside = 200
    NHP = 12 * Nside**2
    #r0,r1,d0,d1 = [np.zeros(NHP) for i in range(4)]
    ra,dec = [np.zeros(NHP) for i in range(2)]
    counts = [np.zeros(NHP) for i in range(4)]
    
    print 'Healpix ranges for', NHP
    for hp in range(NHP):
        #r0[hp],r1[hp],d0[hp],d1[hp] = healpix_radec_bounds(hp, Nside)
        ra[hp],dec[hp] = healpix_to_radecdeg(hp, Nside, 0.5, 0.5)
        
    wcs = Tan()
    for nbands in [4,3,2]:
        bb = [1,2,3,4][:nbands]

        fn = 'wise-frames/WISE-l1b-metadata-%iband.fits' % nbands
        cols = ['ra','dec']
        print 'Reading', fn
        T = fits_table(fn, columns=cols)
        print 'Read', len(T), 'from', fn

        I,J,d = match_radec(T.ra, T.dec, ra, dec, 1.)
        print 'Matched', len(I)
        
        for band in bb:
            #fn = 'wise-frames/WISE-l1b-metadata-%iband.fits' % nbands
            cols = [('w%i'%band)+c for c in
                    ['crval1','crval2','crpix1','crpix2',
                     'cd1_1','cd1_2','cd2_1','cd2_2', 'naxis1','naxis2']]
            print 'Reading', fn
            T = fits_table(fn, columns=cols, rows=I)
            print 'Read', len(T), 'from', fn
            arrs = [T.get(c).astype(float) for c in cols]

            N = len(T)
            for i in xrange(N):
                if arrs[-1][i] == -1:
                    continue
                wcs.set(*[a[i] for a in arrs])
                #rlo,rhi,dlo,dhi = wcs.radec_bounds()
                #I = np.flatnonzero(

                JJ = np.unique(J[I == i])
                print 'WCS', i, ':', len(JJ), 'matched'
                for j in JJ:
                    if wcs.is_inside(ra[j], dec[j]):
                        counts[band-1][j] += 1

    for i,c in enumerate(counts):
        fn = 'coverage-hp-w%i.fits' % (i+1)
        fitsio.write(fn, c, clobber=True)
        print 'Wrote', fn

                
                
    
if __name__ == '__main__':

    image_way()
    sys.exit(0)


    T = None

    Nside = 200
    NHP = 12 * Nside**2

    # Copy the healpix maps into a Hammer-Aitoff map
    W,H = 2000,1000
    wcs = anwcs_create_allsky_hammer_aitoff2(180., 0., W, H)        
    xx,yy = np.meshgrid(np.arange(W), np.arange(H))
    ok,ra,dec = wcs.pixelxy2radec(xx+1, yy+1)
    print('Ok:', np.unique(ok))
    ra  =  ra[ok]
    dec = dec[ok]
        
    hpra,hpdec = [np.zeros(NHP) for i in range(2)]
    for hp in range(NHP):
        hpra[hp],hpdec[hp] = healpix_to_radecdeg(hp, Nside, 0.5, 0.5)
    I,J,d = match_radec(ra, dec, hpra, hpdec, 1., nearest=True)

    hstr = wcs.getHeaderString()
    #print('Header:', 
    hdr = fitsio.FITSHDR()
    while len(hstr):
        card = hstr[:80]
        print('Card:', card)
        hdr.add_record(card, convert=True)
        hstr = hstr[80:]
    
    for band in [1,2,3,4]:
        counts = fitsio.read('coverage-hp-w%i.fits' % band)
        assert(NHP == len(counts))

        print('Percentile:', np.percentile(counts, [0, 1, 50, 99, 100]))
        
        
        img = np.zeros((H,W), np.int16)
        print('Max counts:', counts.max())
        img[np.round(yy[ok][I]).astype(int),
            np.round(xx[ok][I]).astype(int)] = counts[J]
        plt.clf()
        plt.imshow(img, interpolation='nearest', origin='lower',
                   vmin=0, vmax=30)
        plt.colorbar()
        plt.savefig('cov-%i.png' % band)

        fitsio.write('cov-%i.fits' % band, img, header=hdr, clobber=True)
        
        


        if T is None:
            T = fits_table()
            ra,dec = [np.zeros(NHP) for i in range(2)]
            for hp in range(NHP):
                ra[hp],dec[hp] = healpix_to_radecdeg(hp, Nside, 0.5, 0.5)
            T.ra = ra
            T.dec = dec
            #T.writeto('cov-hp-w%i.fits' % band)
        assert(np.all(counts.astype(int) == counts))
        T.set('counts_w%i' % band, counts.astype(np.int32))
    T.writeto('wise-coverage.fits')
            
    # This is the one used for making accurate counts, and the paper plots.
    ## healpix_way()
    
    #image_way()

    from astrometry.util.starutil_numpy import *
    
    W,H = 1000,500
    plot = Plotstuff(size=(W,H), outformat='png')
    plot.wcs = anwcs_create_allsky_hammer_aitoff(180., 0., W, H)
    plot.color = 'verydarkblue'
    plot.plot('fill')
    
    epoch = 2010.
    l = np.linspace(0, 360, 200)
    plot.color = 'gray'
    for b in range(-90, 91, 30):
        ra,dec = ecliptictoradec(l, b+np.zeros_like(b), epoch=epoch)
        plot.move_to_radec(ra[0], dec[0])
        for r,d in zip(ra,dec):
            plot.line_to_radec(r,d)
        plot.stroke()
    b = np.linspace(-90, 90, 200)
    for l in range(0, 360, 30):
        if l == 0:
            plot.color = 'red'
        elif l == 90:
            plot.color = 'green'
        else:
            plot.color = 'gray'
        ra,dec = ecliptictoradec(l+np.zeros_like(b), b, epoch=epoch)
        plot.move_to_radec(ra[0], dec[0])
        for r,d in zip(ra,dec):
            plot.line_to_radec(r,d)
        plot.stroke()
            
    plot.write('1.png')
