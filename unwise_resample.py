#! /usr/bin/env python

import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys
import tempfile
import datetime
import gc
from functools import reduce
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.measurements import label, center_of_mass

import fitsio

from astrometry.util.file import trymakedirs
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.miscutils import estimate_mode, polygons_intersect, clip_polygon, patch_image
from astrometry.util.util import Tan, Sip, flat_median_f
from astrometry.util.resample import resample_with_wcs, OverlapError
from astrometry.util.run_command import run_command
from astrometry.util.starutil_numpy import degrees_between
from astrometry.util.ttime import Time, MemMeas
from astrometry.libkd.spherematch import match_radec

import logging
logger = None
def info(*args):
    msg = ' '.join(map(str, args))
    logger.info(msg)
def debug(*args):
    import logging
    if logger.isEnabledFor(logging.DEBUG):
        msg = ' '.join(map(str, args))
        logger.debug(msg)


from unwise_coadd import get_dir_for_coadd, get_coadd_tile_wcs

def main():
    import argparse
    parser = argparse.ArgumentParser('%prog [options]')

    parser.add_argument('--outdir', '-o', dest='outdir', default='unwise-coadds',
                      help='Output directory: default %(default)s')
    parser.add_argument('--wisedir', help='unWISE coadds input directory')
    
    parser.add_argument('--size', dest='size', default=2048, type=int,
                      help='Set output image size in pixels; default %(default)s')
    parser.add_argument('--width', dest='width', default=0, type=int,
                      help='Set output image width in pixels; default --size')
    parser.add_argument('--height', dest='height', default=0, type=int,
                      help='Set output image height in pixels; default --size')

    parser.add_argument('--pixscale', dest='pixscale', type=float, default=2.75,
                      help='Set coadd pixel scale, default %(default)s arcsec/pixel')
    parser.add_argument('--force', dest='force', action='store_true',
                      default=False, 
                      help='Run even if output file already exists?')
    parser.add_argument('--ra', dest='ra', type=float, default=None,
                      help='Build coadd at given RA center')
    parser.add_argument('--dec', dest='dec', type=float, default=None,
                      help='Build coadd at given Dec center')
    parser.add_argument('--band', type=int, default=None, action='append',
                      help='with --ra,--dec: band(s) to do (1,2,3,4)')
    parser.add_argument('--zoom', type=int, nargs=4,
                        help='Set target image extent (default "0 2048 0 2048")')
    parser.add_argument('--name', default=None,
                      help='Output file name: unwise-NAME-w?-*.fits')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')

    parser.add_argument('--masked-only', action='store_true', default=False)
    
    opt = parser.parse_args()
    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    global logger
    logger = logging.getLogger('unwise_resample')

    radec = opt.ra is not None and opt.dec is not None
    if not radec:
        print('Must specify --ra,--dec or --tile')
        parser.print_help()
        return -1
    if not opt.wisedir:
        print('Must specify --wisedir')
        parser.print_help()
        return -1
    Time.add_measurement(MemMeas)
    W = H = opt.size
    if opt.width:
        W = opt.width
    if opt.height:
        H = opt.height

    unmasked = True
    if opt.masked_only:
        unmasked = False

    dataset = ('custom-%04i%s%03i' %
               (int(opt.ra*10.), 'p' if opt.dec >= 0. else 'm', int(np.abs(opt.dec)*10.)))
    print('Setting custom dataset', dataset)
    # fake tiles table
    tiles = fits_table()
    tiles.coadd_id = np.array([dataset])
    tiles.ra = np.array([opt.ra])
    tiles.dec = np.array([opt.dec])
    tile = tiles[0]

    # cosd = np.cos(np.deg2rad(tile.dec))
    # r0 = tile.ra - (opt.pixscale * W/2.)/3600. / cosd
    # r1 = tile.ra + (opt.pixscale * W/2.)/3600. / cosd
    # d0 = tile.dec - (opt.pixscale * H/2.)/3600.
    # d1 = tile.dec + (opt.pixscale * H/2.)/3600.

    if opt.name:
        tile.coadd_id = opt.name

    if opt.band is None:
        bands = [1,2]
    else:
        bands = list(opt.band)

    outdir = get_dir_for_coadd(opt.outdir, tile.coadd_id)
    if not os.path.exists(outdir):
        print('Creating output directory', outdir)
        trymakedirs(outdir)

    from astrometry.util.run_command import run_command
    rtn,version,err = run_command('git describe')
    if rtn:
        raise RuntimeError('Failed to get version string (git describe):' + ver + err)
    version = version.strip()
    debug('"git describe" version info:', version)

    cowcs = get_coadd_tile_wcs(tile.ra, tile.dec, W, H, opt.pixscale)
    if opt.zoom is not None:
        (x0,x1,y0,y1) = opt.zoom
        W = x1-x0
        H = y1-y0
        zoomwcs = cowcs.get_subimage(x0, y0, W, H)
        print('Zooming WCS from', cowcs, 'to', zoomwcs)
        cowcs = zoomwcs
    ra_center,dec_center = cowcs.radec_center()

    wtiles = unwise_tiles_touching_wcs(cowcs)
    
    for band in bands:
        print('Doing coadd tile', tile.coadd_id, 'band', band)
        t0 = Time()

        tag = 'unwise-%s-w%i' % (tile.coadd_id, band)
        prefix = os.path.join(outdir, tag)
        ofn = prefix + '-img-m.fits'
        if os.path.exists(ofn):
            print('Output file exists:', ofn)
            if not opt.force:
                return 0

        img_m = np.zeros((H,W), np.float32)
        iv_m  = np.zeros((H,W), np.float32)
        std_m = np.zeros((H,W), np.float32)
        n_m   = np.zeros((H,W), np.int32)
        if unmasked:
            img_u = np.zeros((H,W), np.float32)
            iv_u  = np.zeros((H,W), np.float32)
            std_u = np.zeros((H,W), np.float32)
            n_u   = np.zeros((H,W), np.int32)

        for wtile in wtiles:
            from astrometry.util.resample import resample_with_wcs, ResampleError

            print('Reading unWISE tile', wtile.coadd_id)

            wtag = os.path.join(get_dir_for_coadd(opt.wisedir, wtile.coadd_id),
                                'unwise-%s-w%i' % (wtile.coadd_id, band))
            #wtag = os.path.join(opt.wisedir, wtile.coadd_id[:3], wtile.coadd_id,

            prod = 'img-m.fits'
            fn = wtag + '-' + prod
            I,whdr = fitsio.read(fn, header=True)
            wwcs = Tan(whdr)
            ims = [I]

            for prod in ['invvar-m.fits.gz', 'std-m.fits.gz']:
                fn = wtag + '-' + prod
                I = fitsio.read(fn)
                ims.append(I)
            
            if unmasked:
                prod = 'img-u.fits'
                fn = wtag + '-' + prod
                I = fitsio.read(fn)
                ims.append(I)

                for prod in ['invvar-u.fits.gz', 'std-u.fits.gz']:
                    fn = wtag + '-' + prod
                    I = fitsio.read(fn)
                    ims.append(I)
                
            try:
                Yo,Xo,Yi,Xi,rims = resample_with_wcs(cowcs, wwcs, ims, intType=np.int16)
                img_m[Yo,Xo] = rims[0]
                iv_m [Yo,Xo] = rims[1]
                std_m[Yo,Xo] = rims[2]
                if unmasked:
                    img_u[Yo,Xo] = rims[3]
                    iv_u [Yo,Xo] = rims[4]
                    std_u[Yo,Xo] = rims[5]
                del rims
                del ims

                prods = [#('img-m.fits', img_m),
                    #('invvar-m.fits.gz', iv_m),
                    #('std-m.fits.gz', std_m),
                    ('n-m.fits.gz', n_m),]
                if unmasked:
                    prods.extend([#('img-u.fits', img_u),
                        #('invvar-u.fits.gz', iv_u),
                        #('std-u.fits.gz', std_u),
                        ('n-u.fits.gz', n_u),])
                for prod,im in prods:
                    fn = wtag + '-' + prod
                    I = fitsio.read(fn)
                    im[Yo,Xo] = I[Yi,Xi]
                    del I
                del Yo,Xo,Yi,Xi
            except ResampleError:
                pass
                
        # Plug the WCS header cards into the output coadd files.
        hdr = fitsio.FITSHDR()
        cowcs.add_to_header(hdr)
        # Arbitarily plug in a number of header values from the *last* tile
        for r in whdr.records():
            key = r['name']
            if not key in ['MAGZP', 'UNW_SKY', 'UNW_VER', 'UNW_URL', 'UNW_DVER',
                           'UNW_DATE', 'UNW_FR0', 'UNW_FRN', 'UNW_MEDF',
                           'UNW_BGMA', 'REFEREN1', 'REFEREN2', 'EPOCH',
                           'MJDMIN', 'MJDMAX', 'BAND']:
                continue
            hdr.add_record(r)

        if unmasked:
            # "Unmasked" versions
            ofn = prefix + '-img-u.fits'
            fitsio.write(ofn, img_u, header=hdr, clobber=True)
            debug('Wrote', ofn)
            ofn = prefix + '-invvar-u.fits'
            fitsio.write(ofn, iv_u, header=hdr, clobber=True)
            debug('Wrote', ofn)
            ofn = prefix + '-std-u.fits'
            fitsio.write(ofn, std_u, header=hdr, clobber=True)
            debug('Wrote', ofn)
            ofn = prefix + '-n-u.fits'
            fitsio.write(ofn, n_u, header=hdr, clobber=True)
            debug('Wrote', ofn)
    
        # "Masked" versions
        ofn = prefix + '-img-m.fits'
        fitsio.write(ofn, img_m, header=hdr, clobber=True)
        debug('Wrote', ofn)
        ofn = prefix + '-invvar-m.fits'
        fitsio.write(ofn, iv_m, header=hdr, clobber=True)
        debug('Wrote', ofn)
        ofn = prefix + '-std-m.fits'
        fitsio.write(ofn, std_m, header=hdr, clobber=True)
        debug('Wrote', ofn)
        ofn = prefix + '-n-m.fits'
        fitsio.write(ofn, n_m, header=hdr, clobber=True)
        debug('Wrote', ofn)
    
        #ofn = prefix + '-frames.fits'
        #frames.writeto(ofn)
        #debug('Wrote', ofn)


###
# This is taken directly from tractor/wise.py, replacing only the filename.
###
def unwise_tiles_touching_wcs(wcs, atlasfn='data/wise-tiles.fits', polygons=True):
    '''
    Returns a FITS table (with RA,Dec,coadd_id) of unWISE tiles
    '''
    from astrometry.util.miscutils import polygons_intersect
    from astrometry.util.starutil_numpy import degrees_between
    from wise.unwise import unwise_tile_wcs

    T = fits_table(atlasfn)
    trad = wcs.radius()
    wrad = np.sqrt(2.) / 2. * 2048 * 2.75 / 3600.
    rad = trad + wrad
    r, d = wcs.radec_center()
    I, = np.nonzero(np.abs(T.dec - d) < rad)
    I = I[degrees_between(T.ra[I], T.dec[I], r, d) < rad]

    if not polygons:
        return T[I]
    # now check actual polygon intersection
    tw, th = wcs.imagew, wcs.imageh
    targetpoly = [(0.5, 0.5), (tw + 0.5, 0.5),
                  (tw + 0.5, th + 0.5), (0.5, th + 0.5)]
    cd = wcs.get_cd()
    tdet = cd[0] * cd[3] - cd[1] * cd[2]
    if tdet > 0:
        targetpoly = list(reversed(targetpoly))
    targetpoly = np.array(targetpoly)
    keep = []
    for i in I:
        wwcs = unwise_tile_wcs(T.ra[i], T.dec[i])
        cd = wwcs.get_cd()
        wdet = cd[0] * cd[3] - cd[1] * cd[2]
        H, W = wwcs.shape
        poly = []
        for x, y in [(0.5, 0.5), (W + 0.5, 0.5), (W + 0.5, H + 0.5), (0.5, H + 0.5)]:
            rr,dd = wwcs.pixelxy2radec(x, y)
            _,xx,yy = wcs.radec2pixelxy(rr, dd)
            poly.append((xx, yy))
        if wdet > 0:
            poly = list(reversed(poly))
        poly = np.array(poly)
        if polygons_intersect(targetpoly, poly):
            keep.append(i)
    I = np.array(keep)
    return T[I]


if __name__ == '__main__':
    sys.exit(main())

