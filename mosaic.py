import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys

from astrometry.util.resample import *
from astrometry.util.util import *
from astrometry.util.fits import *
from astrometry.util.plotutils import *

from unwise_coadd import *

from check_coadd import read_wise_coadd

l3dir = 'wise-L3'

def main():
    # ra,dec = 266., -29.0
    # W,H = 2500,2500
    # binning = 4
    # afn = 'gc-atlas.fits'
    # bands = range(1,5)
    #base = 'gc-mosaic'
    
    # sequels split into 8 x 2 tiles of ~ 8 deg square
    W,H = 2600,2600
    binning = 4
    afn = 'sequels-atlas.fits'
    bands = [3,4]
    
    
    A = fits_table(afn)
    print len(A), 'atlas tiles'
    
    decstep = (60.-45.) / 2.
    rastep  = (210 - 120.) / 8.
    
    mosnum = 0
    for dec in 45. + decstep/2. + np.arange(2):
        for ra in 120. + rastep/2. + np.arange(8):
            mosnum += 1
            base = 'sequels-mosaic%i' % mosnum
            ps = PlotSequence(base)
            for band in bands:
                make_mosaic(ra, dec, W, H, binning, A, band, base, ps)


def make_mosaic(ra, dec, W, H, binning, A, band, base, ps):
    npixscale = 2.75
    pixscale = npixscale * binning
    outfn = '%s-w%i.fits' % (base, band)
    coadd = np.zeros((H,W), np.float32)
    coaddn = np.zeros((H,W), np.uint8)

    targetwcs = Tan(ra, dec, (W+1)/2., (H+1)/2.,
                    -pixscale/3600., 0., 0., pixscale/3600., W, H)

    print 'wcs extent', get_wcs_radec_bounds(targetwcs)


    tag = 'ac51'
    
    for a in A:
        print 'tile', a.coadd_id
        co = a.coadd_id

        img,hdr,localfn = read_wise_coadd(co, band, get_imgfn=True,
                                          basedir=l3dir)

        # imgdir = os.path.join(co[:2], co[:4], co + '_' + tag)
        # imgfn = '%s_%s-w%i-int-3.fits' % (co, tag, band)
        # pth = os.path.join(imgdir, imgfn)
        # 
        # localdir = os.path.join(l3dir, imgdir)
        # localfn = os.path.join(localdir, imgfn)
        # 
        # print '  ', localfn
        # if not os.path.exists(localfn):
        #     #url = 'http://irsa.ipac.caltech.edu/ibe/data/wise/allsky/4band_p3am_cdd/' + pth
        #     url = 'http://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd/' + pth
        #     print '  url', url
        #     #cmd = 'wget -r -N -nH -np -nv --cut-dirs=4 "%s"' % url
        #     if not os.path.exists(localdir):
        #         print 'creating', localdir
        #         os.makedirs(localdir)
        #         
        #     cmd = 'wget -nv -O %s %s' % (localfn, url)
        #     print '  cmd', cmd
        #     rtn = os.system(cmd)
        #     print '  rtn', rtn
        #     if rtn:
        #         break
        # 
        # img = fitsio.read(localfn)
        print 'img', img.shape
        hh,ww = img.shape
        ww2 = ww - ww % binning
        hh2 = hh - hh % binning
        img = img[:hh2,:ww2]
        print 'cut to', img.shape
        img = binimg(img, binning)
        print 'binned to', img.shape
            
        wcs = Tan(localfn)
        #wcs = anwcs_t(localfn)
        print 'wcs', wcs
        wcs = wcs.scale(1./binning)
        print 'binned', wcs

        plt.clf()
        plt.imshow(img, interpolation='nearest', origin='lower',
                   vmin=np.percentile(img, 1), vmax=np.percentile(img, 98))
        plt.title('%s W%i' % (co, band))
        ps.savefig()

        continue

        try:
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(targetwcs, wcs, [], 3)
        except OverlapError:
            continue
        coadd[Yo,Xo] += img[Yi,Xi]
        coaddn[Yo,Xo] += 1
    
    coadd /= coaddn
    #fitsio.write(outfn, coadd)

    # plt.figure(figsize=(12,12))
    # plt.clf()
    # plt.imshow(coadd, interpolation='nearest', origin='lower')
    # ps.savefig()
    

if __name__ == '__main__':
    main()
    
