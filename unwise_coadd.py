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
lvl = logging.INFO
#logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
#logger = logging.getLogger('unwise_coadd')
logger = None
def info(*args):
    msg = ' '.join(map(str, args))
    logger.info(msg)
def debug(*args):
    import logging
    if logger.isEnabledFor(logging.DEBUG):
        msg = ' '.join(map(str, args))
        logger.debug(msg)

#median_f = np.median
median_f = flat_median_f

# GLOBALS:
# Location of WISE Level 1b inputs
wisedir = 'wise-frames'

'''
at NERSC:
mkdir wise-frames-neo7
for x in /global/cfs/cdirs/cosmo/work/wise/etc/etc_neo7/W*; do ln -s $x wise-frames-neo7/; done
ln -s $COSMO/data/wise/merge/merge_p1bm_frm/wise_allsky_4band_p3as_cdd.fits wise-frames-neo7/
ln -s wise-frames-neo7 wise-frames
ln -s $COSMO/staging/wise/neowiser7/neowiser/p1bm_frm neowiser7-frames
ln -s $COSMO/staging/wise/neowiser6/neowiser/p1bm_frm neowiser6-frames
ln -s $COSMO/staging/wise/neowiser5/neowiser/p1bm_frm neowiser5-frames
ln -s $COSMO/staging/wise/neowiser4/neowiser/p1bm_frm/ neowiser4-frames
ln -s $COSMO/staging/wise/neowiser3/neowiser/p1bm_frm/ neowiser3-frames
ln -s $COSMO/staging/wise/neowiser2/neowiser/p1bm_frm/ neowiser2-frames
ln -s $COSMO/data/wise/neowiser/p1bm_frm/ neowiser-frames
ln -s $COSMO/data/wise/merge/merge_p1bm_frm/ .
'''

wisedirs = [wisedir,
            'merge_p1bm_frm',
            'neowiser-frames',
            'neowiser2-frames',
            'neowiser3-frames',
            'neowiser4-frames',
            'neowiser5-frames',
            'neowiser6-frames',
            'neowiser7-frames',
]
# when adding a year, also see below in "The metadata files to read:"...


mask_gz = True
unc_gz = True

def tile_to_radec(tileid):
    assert(len(tileid) == 8)
    ra = int(tileid[:4], 10) / 10.
    sign = -1 if tileid[4] == 'm' else 1
    dec = sign * int(tileid[5:], 10) / 10.
    return ra,dec

def get_l1b_file(basedir, scanid, frame, band):
    scangrp = scanid[-2:]
    return os.path.join(basedir, scangrp, scanid, '%03i' % frame, 
                        '%s%03i-w%i-int-1b.fits' % (scanid, frame, band))

# from tractor.basics.NanoMaggies
def zeropointToScale(zp):
    '''
    Converts a traditional magnitude zeropoint to a scale factor
    by which nanomaggies should be multiplied to produce image
    counts.
    '''
    return 10.**((zp - 22.5)/2.5)

class Duck():
    pass

def get_coadd_tile_wcs(ra, dec, W=2048, H=2048, pixscale=2.75):
    '''
    Returns a Tan WCS object at the given RA,Dec center, axis aligned, with the
    given pixel W,H and pixel scale in arcsec/pixel.
    '''
    cowcs = Tan(ra, dec, (W+1)/2., (H+1)/2.,
                -pixscale/3600., 0., 0., pixscale/3600., W, H)
    return cowcs

def walk_wcs_boundary(wcs, step=1024, margin=0):
    '''
    Walk the image boundary counter-clockwise.

    Returns rr,dd -- RA,Dec numpy arrays.
    '''
    W = wcs.get_width()
    H = wcs.get_height()
    xlo = 1
    xhi = W
    ylo = 1
    yhi = H
    if margin:
        xlo -= margin
        ylo -= margin
        xhi += margin
        yhi += margin
    
    xx,yy = [],[]
    xwalk = np.linspace(xlo, xhi, int(np.ceil((1+xhi-xlo)/float(step)))+1)
    ywalk = np.linspace(ylo, yhi, int(np.ceil((1+yhi-ylo)/float(step)))+1)
    # bottom edge
    x = xwalk[:-1]
    y = ylo
    xx.append(x)
    yy.append(np.zeros_like(x) + y)
    # right edge
    x = xhi
    y = ywalk[:-1]
    xx.append(np.zeros_like(y) + x)
    yy.append(y)
    # top edge
    x = list(reversed(xwalk))[:-1]
    y = yhi
    xx.append(x)
    yy.append(np.zeros_like(x) + y)
    # left edge
    x = xlo
    y = list(reversed(ywalk))[:-1]
    # (note, NOT closed)
    xx.append(np.zeros_like(y) + x)
    yy.append(y)
    #
    rr,dd = wcs.pixelxy2radec(np.hstack(xx), np.hstack(yy))
    return rr,dd

def get_wcs_radec_bounds(wcs):
    rr,dd = walk_wcs_boundary(wcs)
    r0,r1 = rr.min(), rr.max()
    d0,d1 = dd.min(), dd.max()
    return r0,r1,d0,d1

def in_radec_box(ra,dec, r0,r1,d0,d1, margin):
    assert(r0 <= r1)
    assert(d0 <= d1)
    assert(margin >= 0.)
    if r0 == 0. and r1 == 360.:
        # Just cut on Dec.
        return ((dec + margin >= d0) * (dec - margin <= d1))
        
    cosdec = np.cos(np.deg2rad(max(abs(d0),abs(d1))))
    debug('cosdec:', cosdec)
    # wrap-around... time to switch to unit-sphere instead?
    # Still issues near the Dec poles (if margin/cosdec -> 360)
    ## HACK: 89 degrees -> cosdec 0.017
    if cosdec < 0.02:
        return ((dec + margin >= d0) * (dec - margin <= d1))
    elif (r0 - margin/cosdec < 0) or (r1 + margin/cosdec > 360):
        # python mod: result has same sign as second arg
        rlowrap = (r0 - margin/cosdec) % 360.0
        rhiwrap = (r1 + margin/cosdec) % 360.0
        if (r0 - margin/cosdec < 0):
            raA = rlowrap
            raB = 360.
            raC = 0.
            raD = rhiwrap
        else:
            raA = rhiwrap
            raB = 360.0
            raC = 0.
            raD = rlowrap
        debug('RA wrap-around:', r0,r1, '+ margin', margin, '->', rlowrap, rhiwrap)
        debug('Looking at ranges (%.2f, %.2f) and (%.2f, %.2f)' % (raA,raB,raC,raD))
        assert(raA <= raB)
        assert(raC <= raD)
        return (np.logical_or((ra >= raA) * (ra <= raB),
                              (ra >= raC) * (ra <= raD)) *
                (dec + margin >= d0) *
                (dec - margin <= d1))
    else:
        return ((ra + margin/cosdec >= r0) *
                (ra - margin/cosdec <= r1) *
                (dec + margin >= d0) *
                (dec - margin <= d1))

def get_wise_frames(r0,r1,d0,d1, margin=2.):
    '''
    Returns WISE frames touching the given RA,Dec box plus margin.
    '''
    # Read WISE frame metadata

    #WISE = fits_table(os.path.join(wisedir, 'WISE-index-L1b.fits'))
    #print('Read', len(WISE), 'WISE L1b frames')
    WISE = []
    for band in [1,2,3,4]:
        fn = os.path.join(wisedir, 'WISE-index-L1b_w%i.fits' % band)
        print('Reading', fn)
        W = fits_table(fn)
        WISE.append(W)
    WISE = merge_tables(WISE)
    print('Total of', len(WISE), 'frames')
    WISE.row = np.arange(len(WISE))

    # Coarse cut on RA,Dec box.
    WISE.cut(in_radec_box(WISE.ra, WISE.dec, r0,r1,d0,d1, margin))
    debug('Cut to', len(WISE), 'WISE frames near RA,Dec box')

    # Join to WISE Single-Frame Metadata Tables
    WISE.qual_frame = np.zeros(len(WISE), np.int16) - 1
    WISE.moon_masked = np.zeros(len(WISE), bool)
    WISE.dtanneal = np.zeros(len(WISE), np.float32)

    # pixel distribution stats (used for moon masking)
    WISE.intmedian = np.zeros(len(WISE), np.float32)
    WISE.intstddev = np.zeros(len(WISE), np.float32)
    WISE.intmed16p = np.zeros(len(WISE), np.float32)
    WISE.matched = np.zeros(len(WISE), bool)

    # 4-band, 3-band, or 2-band phase
    WISE.phase = np.zeros(len(WISE), np.uint8)

    # The metadata files to read:
    for nbands,name in [(4,'4band'),
                        (3,'3band'),
                        (2,'2band'),
                        (2,'neowiser'),
                        (2, 'neowiser2'),
                        (2, 'neowiser3'),
                        (2, 'neowiser4'),
                        (2, 'neowiser5'),
                        (2, 'neowiser6'),
                        (2, 'neowiser7'),
                        ]:
        fn = os.path.join(wisedir, 'WISE-l1b-metadata-%s.fits' % name)
        if not os.path.exists(fn):
            print('WARNING: ignoring missing', fn)
            continue
        print('Reading', fn)
        bb = [1,2,3,4][:nbands]
        cols = (['ra', 'dec', 'scan_id', 'frame_num',
                 'qual_frame', 'moon_masked', ] +
                ['w%iintmed16ptile' % b for b in bb] +
                ['w%iintmedian' % b for b in bb] +
                ['w%iintstddev' % b for b in bb])
        if nbands > 2:
            cols.append('dtanneal')
        T = fits_table(fn, columns=cols)
        debug('Read', len(T), 'from', fn)
        # Cut with extra large margins
        T.cut(in_radec_box(T.ra, T.dec, r0,r1,d0,d1, 2.*margin))
        debug('Cut to', len(T), 'near RA,Dec box')
        if len(T) == 0:
            continue

        if not 'dtanneal' in T.get_columns():
            T.dtanneal = np.zeros(len(T), np.float64) + 1000000.
            
        I,J,d = match_radec(WISE.ra, WISE.dec, T.ra, T.dec, 60./3600.)
        debug('Matched', len(I))

        debug('WISE-index-L1b scan_id:', WISE.scan_id.dtype, 'frame_num:', WISE.frame_num.dtype)
        debug('WISE-metadata scan_id:', T.scan_id.dtype, 'frame_num:', T.frame_num.dtype)

        K = np.flatnonzero((WISE.scan_id  [I] == T.scan_id  [J]) *
                           (WISE.frame_num[I] == T.frame_num[J]))
        I = I[K]
        J = J[K]
        debug('Cut to', len(I), 'matching scan/frame')

        for band in bb:
            K = (WISE.band[I] == band)
            debug('Band', band, ':', sum(K))
            if sum(K) == 0:
                continue
            II = I[K]
            JJ = J[K]
            WISE.qual_frame [II] = T.qual_frame [JJ].astype(WISE.qual_frame.dtype)
            moon = T.moon_masked[JJ]
            WISE.moon_masked[II] = np.array([m[band-1] == '1' for m in moon]
                                            ).astype(WISE.moon_masked.dtype)
            WISE.dtanneal [II] = T.dtanneal[JJ].astype(WISE.dtanneal.dtype)
            WISE.intmedian[II] = T.get('w%iintmedian' % band)[JJ].astype(np.float32)
            WISE.intstddev[II] = T.get('w%iintstddev' % band)[JJ].astype(np.float32)
            WISE.intmed16p[II] = T.get('w%iintmed16ptile' % band)[JJ].astype(np.float32)
            WISE.matched[II] = True
            WISE.phase[II] = nbands

    debug(np.sum(WISE.matched), 'of', len(WISE), 'matched to metadata tables')
    assert(np.sum(WISE.matched) == len(WISE))
    WISE.delete_column('matched')
    # Reorder by scan, frame, band
    WISE.cut(np.lexsort((WISE.band, WISE.frame_num, WISE.scan_id)))
    return WISE

def get_dir_for_coadd(outdir, coadd_id):
    # base/RRR/RRRRsDDD/unwise-*
    return os.path.join(outdir, coadd_id[:3], coadd_id)

def get_epoch_breaks(mjds):
    mjds = np.sort(mjds)
    # define an epoch either as a gap of more than 3 months
    # between frames, or as > 6 months since start of epoch.
    start = mjds[0]
    ebreaks = []
    for lastmjd,mjd in zip(mjds, mjds[1:]):
        if (mjd - lastmjd >= 90.) or (mjd - start >= 180.):
            ebreaks.append((mjd + lastmjd) / 2.)
            start = mjd
    print('Defined epoch breaks', ebreaks)
    print('Found', len(ebreaks), 'epoch breaks')
    return ebreaks

def one_coadd(ti, band, W, H, frames,
              pixscale=2.75,
              zoom=None,
              outdir='unwise-coadds',
              medfilt=None,
              do_dsky=False,
              bgmatch=False, center=False,
              minmax=False, rchi_fraction=0.01, epoch=None,
              before=None, after=None,
              ps=None,
              wishlist=False,
              mp1=None, mp2=None,
              do_cube=False, do_cube1=False,
              plots2=False,
              frame0=0, nframes=0, nframes_random=0,
              force=False, maxmem=0,
              allow_download=False,
              force_outdir=False, just_image=False, version=None,
              write_masks=True):
    '''
    Create coadd for one tile & band.
    '''
    debug('Coadd tile', ti.coadd_id)
    debug('RA,Dec', ti.ra, ti.dec)
    debug('Band', band)

    from astrometry.util.multiproc import multiproc
    if mp1 is None:
        mp1 = multiproc()
    if mp2 is None:
        mp2 = multiproc()

    wisepixscale = 2.75

    if version is None:
        from astrometry.util.run_command import run_command
        rtn,version,err = run_command('git describe')
        if rtn:
            raise RuntimeError('Failed to get version string (git describe):' + ver + err)
        version = version.strip()
    debug('"git describe" version info:', version)

    if not force_outdir:
        outdir = get_dir_for_coadd(outdir, ti.coadd_id)
    trymakedirs(outdir)
    tag = 'unwise-%s-w%i' % (ti.coadd_id, band)
    prefix = os.path.join(outdir, tag)
    ofn = prefix + '-img-m.fits'
    if os.path.exists(ofn):
        print('Output file exists:', ofn)
        if not force:
            return 0

    cowcs = get_coadd_tile_wcs(ti.ra, ti.dec, W, H, pixscale)
    if zoom is not None:
        (x0,x1,y0,y1) = zoom
        W = x1-x0
        H = y1-y0
        zoomwcs = cowcs.get_subimage(x0, y0, W, H)
        print('Zooming WCS from', cowcs, 'to', zoomwcs)
        cowcs = zoomwcs

    # Intermediate world coordinates (IWC) polygon
    r,d = walk_wcs_boundary(cowcs, step=W, margin=10)
    ok,u,v = cowcs.radec2iwc(r,d)
    copoly = np.array(list(reversed(list(zip(u,v)))))
    #print('Coadd IWC polygon:', copoly)

    margin = (1.1 # safety margin
              * (np.sqrt(2.) / 2.) # diagonal
              * (max(W,H) * pixscale/3600.
                 + 1016 * wisepixscale/3600) # WISE FOV + coadd FOV side length
              ) # in deg
    t0 = Time()

    ra_center,dec_center = cowcs.radec_center()

    # cut
    frames = frames[frames.band == band]
    frames.cut(degrees_between(ra_center, dec_center, frames.ra, frames.dec) < margin)
    debug('Found', len(frames), 'WISE frames in range and in band W%i' % band)

    if before is not None:
        frames.cut(frames.mjd < before)
        debug('Cut to', len(frames), 'frames before MJD', before)
    if after is not None:
        frames.cut(frames.mjd > after)
        debug('Cut to', len(frames), 'frames after MJD', after)

    # Cut on IWC box
    ok,u,v = cowcs.radec2iwc(frames.ra, frames.dec)
    u0,v0 = copoly.min(axis=0)
    u1,v1 = copoly.max(axis=0)
    #print 'Coadd IWC range:', u0,u1, v0,v1
    margin = np.sqrt(2.) * (1016./2.) * (wisepixscale/3600.) * 1.01 # safety
    frames.cut((u + margin >= u0) * (u - margin <= u1) *
             (v + margin >= v0) * (v - margin <= v1))
    debug('cut to', len(frames), 'in RA,Dec box')

    # Use a subset of frames?
    if epoch is not None:
        ebreaks = get_epoch_breaks(frames.mjd)
        assert(epoch <= len(ebreaks))
        if epoch > 0:
            frames = frames[frames.mjd >= ebreaks[epoch - 1]]
        if epoch < len(ebreaks):
            frames = frames[frames.mjd <  ebreaks[epoch]]
        debug('Cut to', len(frames), 'within epoch')

    if bgmatch or center:
        # reorder by dist from center
        frames.cut(np.argsort(degrees_between(ra_center, dec_center, frames.ra, frames.dec)))
    
    if ps and False:
        plt.clf()
        plt.plot(copoly[:,0], copoly[:,1], 'r-')
        plt.plot(copoly[0,0], copoly[0,1], 'ro')
        plt.plot(u, v, 'b.')
        plt.axvline(u0 - margin, color='k')
        plt.axvline(u1 + margin, color='k')
        plt.axhline(v0 - margin, color='k')
        plt.axhline(v1 + margin, color='k')
        ok,u2,v2 = cowcs.radec2iwc(frames.ra, frames.dec)
        plt.plot(u2, v2, 'go')
        ps.savefig()
        
    # We keep all of the input frames in the list, marking ones we're not
    # going to use, for later diagnostics.
    frames.use = np.ones(len(frames), bool)
    frames.use *= (frames.qual_frame > 0)
    debug('Cut out qual_frame = 0;', sum(frames.use), 'remaining')

    if band in [3,4]:
        frames.use *= (frames.dtanneal > 2000.)
        debug('Cut out dtanneal <= 2000 seconds:', sum(frames.use), 'remaining')

    if band == 4:
        ok = np.array([np.logical_or(s < '03752a', s > '03761b')
                       for s in frames.scan_id])
        frames.use *= ok
        debug('Cut out bad scans in W4:', sum(frames.use), 'remaining')

    if band in [3,4]:
        # Cut on moon, based on (robust) measure of standard deviation
        if sum(frames.moon_masked[frames.use]):
            moon = frames.moon_masked[frames.use]
            nomoon = np.logical_not(moon)
            Imoon = np.flatnonzero(frames.use)[moon]
            assert(sum(moon) == len(Imoon))
            debug(sum(nomoon), 'of', sum(frames.use), 'frames are not moon_masked')
            nomoonstdevs = frames.intmed16p[frames.use][nomoon]
            med = np.median(nomoonstdevs)
            mad = 1.4826 * np.median(np.abs(nomoonstdevs - med))
            debug('Median', med, 'MAD', mad)
            moonstdevs = frames.intmed16p[frames.use][moon]
            okmoon = (moonstdevs - med)/mad < 5.
            debug(sum(np.logical_not(okmoon)), 'of', len(okmoon), 'moon-masked frames have large pixel variance')
            frames.use[Imoon] *= okmoon
            debug('Cut to', sum(frames.use), 'on moon')
            del Imoon
            del moon
            del nomoon
            del nomoonstdevs
            del med
            del mad
            del moonstdevs
            del okmoon

    if frame0 or nframes or nframes_random:
        i0 = frame0
        if nframes:
            frames = frames[frame0:frame0 + nframes]
        elif nframes_random:
            frames = frames[frame0 + np.random.permutation(len(frames)-frame0)[:nframes_random]]
        else:
            frames = frames[frame0:]
        debug('Cut to', len(frames), 'frames starting from index', frame0)
        
    debug('Frames to coadd:')
    for i,w in enumerate(frames):
        debug('  ', i, w.scan_id, '%4i' % w.frame_num, 'MJD', w.mjd)

    if len(frames) == 0:
        info('No frames overlap position x time')
        return -1

    if wishlist:
        for wise in frames:
            intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, band)
            if not os.path.exists(intfn):
                print('Need:', intfn)
                #cmd = 'rsync -LRvz carver:unwise/./%s .' % intfn
                #print cmd
                #os.system(cmd)
        return 0

    # Estimate memory usage and bail out if too high.
    if maxmem:
        mem = 1. + (len(frames) * 1e6/2. * 5. / 1e9)
        print('Estimated mem usage:', mem)
        if mem > maxmem:
            print('Estimated memory usage:', mem, 'GB > max', maxmem)
            return -1

    # *inclusive* coordinates of the bounding-box in the coadd of this
    # image (x0,x1,y0,y1)
    frames.coextent = np.zeros((len(frames), 4), np.int32)
    # *inclusive* coordinates of the bounding-box in the image
    # overlapping coadd
    frames.imextent = np.zeros((len(frames), 4), np.int32)

    frames.imagew = np.zeros(len(frames), np.int32)
    frames.imageh = np.zeros(len(frames), np.int32)
    frames.intfn  = np.zeros(len(frames), object)
    frames.wcs    = np.zeros(len(frames), object)

    # count total number of coadd-space pixels -- this determines memory use
    pixinrange = 0.

    nu = 0
    NU = sum(frames.use)
    failedfiles = []
    for wi,wise in enumerate(frames):
        if not wise.use:
            continue
        nu += 1
        debug(nu, 'of', NU, 'scan', wise.scan_id, 'frame', wise.frame_num, 'band', band)

        found = False
        for wdir in wisedirs + [None]:
            download = False
            if wdir is None:
                download = allow_download
                wdir = 'merge_p1bm_frm'

            intfn = get_l1b_file(wdir, wise.scan_id, wise.frame_num, band)
            debug('intfn', intfn)
            intfnx = intfn.replace(wdir+'/', '')

            if download:
                # Try to download the file from IRSA.
                cmd = (('(wget -r -N -nH -np -nv --cut-dirs=4 -A "*w%i*" ' +
                        '"http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p1bm_frm/%s/")') %
                        (band, os.path.dirname(intfnx)))
                print()
                print('Trying to download file:')
                print(cmd)
                print()
                os.system(cmd)
                print()

            if os.path.exists(intfn):
                try:
                    wcs = Sip(intfn)
                except RuntimeError:
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                debug('does not exist:', intfn)
                continue
            if (os.path.exists(intfn.replace('-int-', '-unc-') + '.gz') and
                os.path.exists(intfn.replace('-int-', '-msk-') + '.gz')):
                found = True
                break
            else:
                print('missing unc or msk file')
                continue
        if not found:
            print('WARNING: Not found: scan', wise.scan_id, 'frame', wise.frame_num, 'band', band)
            failedfiles.append(intfnx)
            continue

        h,w = int(wcs.get_height()), int(wcs.get_width())
        r,d = walk_wcs_boundary(wcs, step=2.*w, margin=10)
        ok,u,v = cowcs.radec2iwc(r, d)
        poly = np.array(list(reversed(list(zip(u,v)))))
        #print 'Image IWC polygon:', poly
        intersects = polygons_intersect(copoly, poly)

        if ps and False:
            plt.clf()
            plt.plot(copoly[:,0], copoly[:,1], 'r-')
            plt.plot(copoly[0,0], copoly[0,1], 'ro')
            plt.plot(poly[:,0], poly[:,1], 'b-')
            plt.plot(poly[0,0], poly[0,1], 'bo')
            cpoly = np.array(clip_polygon(copoly, poly))
            if len(cpoly) == 0:
                pass
            else:
                print('cpoly:', cpoly)
                plt.plot(cpoly[:,0], cpoly[:,1], 'm-')
                plt.plot(cpoly[0,0], cpoly[0,1], 'mo')
            ps.savefig()

        if not intersects:
            debug('Image does not intersect target')
            frames.use[wi] = False
            continue

        cpoly = np.array(clip_polygon(copoly, poly))
        if len(cpoly) == 0:
            debug('No overlap between coadd and image polygons')
            debug('copoly:', copoly)
            debug('poly:', poly)
            debug('cpoly:', cpoly)
            frames.use[wi] = False
            continue

        # Convert the intersected polygon in IWC space into image
        # pixel bounds.
        # Coadd extent:
        xy = np.array([cowcs.iwc2pixelxy(u,v) for u,v in cpoly])
        xy -= 1
        x0,y0 = np.floor(xy.min(axis=0)).astype(int)
        x1,y1 = np.ceil (xy.max(axis=0)).astype(int)
        frames.coextent[wi,:] = [np.clip(x0, 0, W-1),
                               np.clip(x1, 0, W-1),
                               np.clip(y0, 0, H-1),
                               np.clip(y1, 0, H-1)]

        # Input image extent:
        #   There was a bug in the as-run coadds; all imextents are
        #   [0,1015,0,1015] as a result.
        #rd = np.array([cowcs.iwc2radec(u,v) for u,v in poly])
        # Should be: ('cpoly' rather than 'poly' here)
        rd = np.array([cowcs.iwc2radec(u,v) for u,v in cpoly])
        ok,x,y = np.array(wcs.radec2pixelxy(rd[:,0], rd[:,1]))
        x -= 1
        y -= 1
        x0,y0 = [np.floor(v.min(axis=0)).astype(int) for v in [x,y]]
        x1,y1 = [np.ceil (v.max(axis=0)).astype(int) for v in [x,y]]
        frames.imextent[wi,:] = [np.clip(x0, 0, w-1),
                               np.clip(x1, 0, w-1),
                               np.clip(y0, 0, h-1),
                               np.clip(y1, 0, h-1)]

        frames.intfn[wi] = intfn
        frames.imagew[wi] = w
        frames.imageh[wi] = h
        frames.wcs[wi] = wcs
        debug('Image extent:', frames.imextent[wi,:])
        debug('Coadd extent:', frames.coextent[wi,:])

        # Count total coadd-space bounding-box size -- this x 5 bytes
        # is the memory toll of our round-1 coadds, which is basically
        # the peak memory use.
        e = frames.coextent[wi,:]
        pixinrange += (1+e[1]-e[0]) * (1+e[3]-e[2])
        debug('Total pixels in coadd space:', pixinrange)

    if len(failedfiles):
        print(len(failedfiles), 'failed:')
        for f in failedfiles:
            print('  ', f)
        print()

    # Now we can make a more informed estimate of memory use.
    if maxmem:
        mem = 1. + (pixinrange * 5. / 1e9)
        print('Estimated mem usage:', mem)
        if mem > maxmem:
            print('Estimated memory usage:', mem, 'GB > max', maxmem)
            return -1

    # convert from object array to string array; '' rather than '0'
    frames.intfn = np.array([{0:''}.get(s,s) for s in frames.intfn])
    debug('Cut to', sum(frames.use), 'frames intersecting target')

    t1 = Time()
    debug('Up to coadd_wise:', t1 - t0)

    # Now that we've got some information about the input frames, call
    # the real coadding code.  Maybe we should move this first loop into
    # the round 1 coadd...
    try:
        (coim,coiv,copp,con, coimb,coivb,coppb,conb,masks, cube, cosky,
         comin,comax,cominb,comaxb
         )= coadd_wise(ti.coadd_id, cowcs, frames[frames.use], ps, band, mp1, mp2, do_cube,
                       medfilt, plots2=plots2, do_dsky=do_dsky,
                       bgmatch=bgmatch, minmax=minmax,
                       rchi_fraction=rchi_fraction, do_cube1=do_cube1)
    except:
        print('coadd_wise failed:')
        import traceback
        traceback.print_exc()
        print('time up to failure:')
        t2 = Time()
        print(t2 - t1)
        return -1
    t2 = Time()
    debug('coadd_wise:', t2-t1)

    # For any "masked" pixels that have invvar = 0 (ie, NO pixels
    # contributed), fill in the image from the "unmasked" image.
    # Leave the invvar image untouched.
    coimb[coivb == 0] = coim[coivb == 0]

    # Plug the WCS header cards into the output coadd files.
    hdr = fitsio.FITSHDR()
    cowcs.add_to_header(hdr)

    hdr.add_record(dict(name='MAGZP', value=22.5,
                        comment='Magnitude zeropoint (in Vega mag)'))
    hdr.add_record(dict(name='UNW_SKY', value=cosky,
                        comment='Background value subtracted from coadd img'))
    hdr.add_record(dict(name='UNW_VER', value=version,
                        comment='unWISE code git revision'))
    hdr.add_record(dict(name='UNW_URL', value='https://github.com/dstndstn/unwise-coadds',
                        comment='git URL'))
    hdr.add_record(dict(name='UNW_DVER', value=1,
                        comment='unWISE data model version'))
    hdr.add_record(dict(name='UNW_DATE', value=datetime.datetime.now().isoformat(),
                        comment='unWISE run time'))
    hdr.add_record(dict(name='UNW_FR0', value=frame0, comment='unWISE frame start'))
    hdr.add_record(dict(name='UNW_FRN', value=nframes, comment='unWISE N frames'))
    hdr.add_record(dict(name='UNW_FRNR', value=nframes_random, comment='unWISE N random frames'))
    hdr.add_record(dict(name='UNW_MEDF', value=medfilt, comment='unWISE median filter sz'))
    hdr.add_record(dict(name='UNW_BGMA', value=bgmatch, comment='unWISE background matching?'))

    # "Unmasked" versions
    ofn = prefix + '-img-u.fits'
    fitsio.write(ofn, coim.astype(np.float32), header=hdr, clobber=True)
    debug('Wrote', ofn)

    if just_image:
        return 0

    ofn = prefix + '-invvar-u.fits'
    fitsio.write(ofn, coiv.astype(np.float32), header=hdr, clobber=True)
    debug('Wrote', ofn)
    ofn = prefix + '-std-u.fits'
    fitsio.write(ofn, copp.astype(np.float32), header=hdr, clobber=True)
    debug('Wrote', ofn)
    ofn = prefix + '-n-u.fits'
    fitsio.write(ofn, con.astype(np.int32), header=hdr, clobber=True)
    debug('Wrote', ofn)

    # "Masked" versions
    ofn = prefix + '-img-m.fits'
    fitsio.write(ofn, coimb.astype(np.float32), header=hdr, clobber=True)
    debug('Wrote', ofn)
    ofn = prefix + '-invvar-m.fits'
    fitsio.write(ofn, coivb.astype(np.float32), header=hdr, clobber=True)
    debug('Wrote', ofn)
    ofn = prefix + '-std-m.fits'
    fitsio.write(ofn, coppb.astype(np.float32), header=hdr, clobber=True)
    debug('Wrote', ofn)
    ofn = prefix + '-n-m.fits'
    fitsio.write(ofn, conb.astype(np.int32), header=hdr, clobber=True)
    debug('Wrote', ofn)

    if do_cube:
        ofn = prefix + '-cube.fits'
        fitsio.write(ofn, cube.astype(np.float32), header=hdr, clobber=True)

    if minmax:
        ofn = prefix + '-min-m.fits'
        fitsio.write(ofn, cominb.astype(np.float32), header=hdr, clobber=True)
        debug('Wrote', ofn)
        ofn = prefix + '-max-m.fits'
        fitsio.write(ofn, comaxb.astype(np.float32), header=hdr, clobber=True)
        debug('Wrote', ofn)
        ofn = prefix + '-min-u.fits'
        fitsio.write(ofn, comin.astype(np.float32), header=hdr, clobber=True)
        debug('Wrote', ofn)
        ofn = prefix + '-max-u.fits'
        fitsio.write(ofn, comax.astype(np.float32), header=hdr, clobber=True)
        debug('Wrote', ofn)

    frames.included = np.zeros(len(frames), bool)
    frames.sky1 = np.zeros(len(frames), np.float32)
    frames.sky2 = np.zeros(len(frames), np.float32)
    frames.zeropoint = np.zeros(len(frames), np.float32)
    frames.npixoverlap = np.zeros(len(frames), np.int32)
    frames.npixpatched = np.zeros(len(frames), np.int32)
    frames.npixrchi    = np.zeros(len(frames), np.int32)
    frames.weight      = np.zeros(len(frames), np.float32)

    Iused = np.flatnonzero(frames.use)
    assert(len(Iused) == len(masks))

    maskdir = os.path.join(outdir, tag + '-mask')
    if not os.path.exists(maskdir):
        os.mkdir(maskdir)
            
    for i,mm in enumerate(masks):
        if mm is None:
            continue

        ii = Iused[i]
        frames.sky1       [ii] = mm.sky
        frames.sky2       [ii] = mm.dsky
        frames.zeropoint  [ii] = mm.zp
        frames.npixoverlap[ii] = mm.ncopix
        frames.npixpatched[ii] = mm.npatched
        frames.npixrchi   [ii] = mm.nrchipix
        frames.weight     [ii] = mm.w

        if not mm.included:
            continue

        frames.included   [ii] = True

        # Write outlier masks
        if write_masks:
            ofn = frames.intfn[ii].replace('-int', '')
            ofn = os.path.join(maskdir, 'unwise-mask-' + ti.coadd_id + '-'
                               + os.path.basename(ofn) + '.gz')
            w,h = frames.imagew[ii],frames.imageh[ii]
            fullmask = np.zeros((h,w), mm.omask.dtype)
            x0,x1,y0,y1 = frames.imextent[ii,:]
            fullmask[y0:y1+1, x0:x1+1] = mm.omask
            fitsio.write(ofn, fullmask, clobber=True)
            debug('Wrote mask', (i+1), 'of', len(masks), ':', ofn)

    frames.delete_column('wcs')

    # downcast datatypes, and work around fitsio's issues with
    # "bool" columns
    for c,t in [('included', np.uint8),
                ('use', np.uint8),
                ('moon_masked', np.uint8),
                ('imagew', np.int16),
                ('imageh', np.int16),
                ('coextent', np.int16),
                ('imextent', np.int16),
                ]:
        frames.set(c, frames.get(c).astype(t))

    ofn = prefix + '-frames.fits'
    frames.writeto(ofn)
    debug('Wrote', ofn)

    if write_masks:
        md = tag + '-mask'
        cmd = ('cd %s && tar czf %s %s && rm -R %s' %
               (outdir, md + '.tgz', md, md))
        debug('tgz:', cmd)
        rtn,out,err = run_command(cmd)
        debug(out, err)
        if rtn:
            print('ERROR: return code', rtn, file=sys.stderr)
            print('Command:', cmd, file=sys.stderr)
            print(out, file=sys.stderr)
            print(err, file=sys.stderr)
            ok = False

    return rtn

def plot_region(r0,r1,d0,d1, ps, T, WISE, wcsfns, W, H, pixscale, margin=1.05,
                allsky=False, grid_ra_range=None, grid_dec_range=None,
                grid_spacing=[5, 5, 20, 10], label_tiles=True, draw_outline=True,
                tiles=[], ra=0., dec=0.):
    from astrometry.blind.plotstuff import Plotstuff
    maxcosdec = np.cos(np.deg2rad(min(abs(d0),abs(d1))))
    if allsky:
        W,H = 1000,500
        plot = Plotstuff(outformat='png', size=(W,H))
        plot.wcs = anwcs_create_allsky_hammer_aitoff(ra, dec, W, H)
    else:
        plot = Plotstuff(outformat='png', size=(800,800),
                         rdw=((r0+r1)/2., (d0+d1)/2., margin*max(d1-d0, (r1-r0)*maxcosdec)))

    plot.fontsize = 10
    plot.halign = 'C'
    plot.valign = 'C'

    for i in range(3):
        if i in [0,2]:
            plot.color = 'verydarkblue'
        else:
            plot.color = 'black'
        plot.plot('fill')
        plot.color = 'white'
        out = plot.outline

        if i == 0:
            if T is None:
                continue
            print('plot 0')
            for i,ti in enumerate(T):
                cowcs = get_coadd_tile_wcs(ti.ra, ti.dec, W, H, pixscale)
                plot.alpha = 0.5
                out.wcs = anwcs_new_tan(cowcs)
                out.fill = 1
                plot.plot('outline')
                out.fill = 0
                plot.plot('outline')

                if label_tiles:
                    plot.alpha = 1.
                    rc,dc = cowcs.radec_center()
                    plot.text_radec(rc, dc, '%i' % i)

        elif i == 1:
            if WISE is None:
                continue
            print('plot 1')
            # cut
            #WISE = WISE[WISE.band == band]
            plot.alpha = (3./256.)
            out.fill = 1
            print('Plotting', len(WISE), 'exposures')
            wcsparams = []
            fns = []
            for wi,wise in enumerate(WISE):
                if wi % 10 == 0:
                    print('.', end=' ')
                if wi % 1000 == 0:
                    print(wi, 'of', len(WISE))

                if wi and wi % 10000 == 0:
                    fn = ps.getnext()
                    plot.write(fn)
                    print('Wrote', fn)

                    wp = np.array(wcsparams)
                    WW = fits_table()
                    WW.crpix  = wp[:, 0:2]
                    WW.crval  = wp[:, 2:4]
                    WW.cd     = wp[:, 4:8]
                    WW.imagew = wp[:, 8]
                    WW.imageh = wp[:, 9]
                    WW.intfn = np.array(fns)
                    WW.writeto('sequels-wcs.fits')

                intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, wise.band)
                try:
                    wcs = Tan(intfn, 0, 1)
                except:
                    import traceback
                    traceback.print_exc()
                    continue
                out.wcs = anwcs_new_tan(wcs)
                plot.plot('outline')

                wcsparams.append((wcs.crpix[0], wcs.crpix[1], wcs.crval[0], wcs.crval[1],
                                  wcs.cd[0], wcs.cd[1], wcs.cd[2], wcs.cd[3],
                                  wcs.imagew, wcs.imageh))
                fns.append(intfn)

            wp = np.array(wcsparams)
            WW = fits_table()
            WW.crpix  = wp[:, 0:2]
            WW.crval  = wp[:, 2:4]
            WW.cd     = wp[:, 4:8]
            WW.imagew = wp[:, 8]
            WW.imageh = wp[:, 9]
            WW.intfn = np.array(fns)
            WW.writeto('sequels-wcs.fits')

            fn = ps.getnext()
            plot.write(fn)
            print('Wrote', fn)

        elif i == 2:
            print('plot 2')
            if wcsfns is None:
                print('wcsfns is none')
                continue
            print('wcsfns:', len(wcsfns), 'tiles', len(tiles))
            plot.alpha = 0.5
            for fn in wcsfns:
                out.set_wcs_file(fn, 0)
                out.fill = 1
                plot.plot('outline')
                out.fill = 0
                plot.plot('outline')

            for it,tile in enumerate(tiles):
                if it % 1000 == 0:
                    print('plotting tile', tile)
                ra,dec = tile_to_radec(tile)
                wcs = get_coadd_tile_wcs(ra, dec)
                out.wcs = anwcs_new_tan(wcs)
                out.fill = 1
                plot.plot('outline')
                out.fill = 0
                plot.plot('outline')

        plot.color = 'gray'
        plot.alpha = 1.
        grid = plot.grid
        grid.ralabeldir = 2

        if grid_ra_range is not None:
            grid.ralo, grid.rahi = grid_ra_range
        if grid_dec_range is not None:
            grid.declo, grid.dechi = grid_dec_range
        plot.plot_grid(*grid_spacing)

        if draw_outline:
            plot.color = 'red'
            plot.apply_settings()
            plot.line_constant_dec(d0, r0, r1)
            plot.stroke()
            plot.line_constant_ra(r1, d0, d1)
            plot.stroke()
            plot.line_constant_dec(d1, r1, r0)
            plot.stroke()
            plot.line_constant_ra(r0, d1, d0)
            plot.stroke()
        fn = ps.getnext()
        plot.write(fn)
        print('Wrote', fn)


def _bounce_one_round2(*A):
    try:
        return _coadd_one_round2(*A)
    except:
        import traceback
        print('_coadd_one_round2 failed:')
        traceback.print_exc()
        raise

def _coadd_one_round2(X):
    '''
    For multiprocessing, the function to be called for each round-2
    frame.
    '''
    (ri, N, scanid, rr, cow1, cowimg1, cowimgsq1, tinyw,
                       plotfn, ps1, do_dsky, rchi_fraction) = X
    if rr is None:
        return None
    debug('Coadd round 2, image', (ri+1), 'of', N)
    t00 = Time()
    mm = Duck()
    mm.npatched = rr.npatched
    mm.ncopix   = rr.ncopix
    mm.sky      = rr.sky
    mm.zp       = rr.zp
    mm.w        = rr.w
    mm.included = True

    cox0,cox1,coy0,coy1 = rr.coextent
    coslc = slice(coy0, coy1+1), slice(cox0, cox1+1)
    # Remove this image from the per-pixel std calculation...
    subw  = np.maximum(cow1[coslc] - rr.w, tinyw)
    subco = (cowimg1  [coslc] - (rr.w * rr.rimg   )) / subw
    subsq = (cowimgsq1[coslc] - (rr.w * rr.rimg**2)) / subw
    subv = np.maximum(0, subsq - subco**2)
    # previously, no prior:
    # subp = np.sqrt(np.maximum(0, subsq - subco**2))

    # "prior" estimate of per-pixel noise: sig1 + 3% flux in quadrature
    # rr.w = 1./sig1**2 for this image.
    priorv = 1./rr.w + (0.03 * np.maximum(subco, 0))**2
    # Weight that prior equal to the 'subv' estimate from nprior exposures
    nprior = 5
    priorw = nprior * rr.w
    subpp = np.sqrt((subv * subw + priorv * priorw) / (subw + priorw))
    
    # rr.rmask bit value 1 indicates that the pixel is within the coadd
    # region.
    mask = (rr.rmask & 1).astype(bool)

    # like in the WISE Atlas Images, estimate sky difference via
    # median difference in the overlapping area.
    if do_dsky:
        dsky = median_f((rr.rimg[mask] - subco[mask]).astype(np.float32))
        debug('Sky difference:', dsky)
    else:
        dsky = 0.

    rchi = ((rr.rimg - dsky - subco) * mask * (subw > 0) * (subpp > 0) /
            np.maximum(subpp, 1e-6))
    assert(np.all(np.isfinite(rchi)))

    badpix = (np.abs(rchi) >= 5.)
    #debug 'Number of rchi-bad pixels:', np.count_nonzero(badpix)

    mm.nrchipix = np.count_nonzero(badpix)

    # Bit 1: abs(rchi) >= 5
    badpixmask = badpix.astype(np.uint8)
    # grow by a small margin
    badpix = binary_dilation(badpixmask)
    # Bit 2: grown
    badpixmask += (2 * badpix).astype(np.uint8)

    # Add dilated rchi-masked pixels to the "rmask" (clear value 0x2)
    rr.rmask[badpix] &= (0xff - 0x2)

    # "omask" is the file we're going to write out saying which pixels
    # were rchi masked, in L1b pixel space.
    mm.omask = np.zeros((int(rr.wcs.get_height()), int(rr.wcs.get_width())),
                        badpixmask.dtype)
    try:
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(rr.wcs, rr.cosubwcs, [], None)
        mm.omask[Yo,Xo] = badpixmask[Yi,Xi]
    except OverlapError:
        import traceback
        debug('WARNING: Caught OverlapError resampling rchi mask')
        debug('rr WCS', rr.wcs)
        debug('shape', mm.omask.shape)
        debug('cosubwcs:', rr.cosubwcs)
        if logger.isEnabledFor(logging.DEBUG):
            traceback.print_exc(None, sys.stdout)

    if mm.nrchipix > mm.ncopix * rchi_fraction:
        print(('WARNING: dropping exposure %s: n rchi pixels %i / %i' %
               (scanid, mm.nrchipix, mm.ncopix)))
        mm.included = False

    if ps1:
        # save for later
        mm.rchi = rchi
        mm.badpix = badpix
        if mm.included:
            mm.rimg_orig = rr.rimg.copy()
            mm.rmask_orig = rr.rmask.copy()

    if mm.included:
        ok = patch_image(rr.rimg, np.logical_not(badpix),
                         required=(badpix * mask))
        if not ok:
            print('patch_image failed')
            return None

        rimg = (rr.rimg - dsky)
        mm.coslc = coslc
        mm.coimgsq = mask * rr.w * rimg**2
        mm.coimg   = mask * rr.w * rimg
        mm.cow     = mask * rr.w
        mm.con     = mask
        # mm.rmask2 is bit value 2 from rr.rmask: original L1b pixel good
        # times dilated rchi-based pixel good.
        mm.rmask2  = (rr.rmask & 2).astype(bool)

    mm.dsky = dsky / rr.zpscale
        
    if plotfn:
        # HACK
        rchihistrange = 6
        rchihistargs = dict(range=(-rchihistrange,rchihistrange), bins=100)
        rchihist = None
        rchihistedges = None

        R,C = 3,3
        plt.clf()
        I = rr.rimg - dsky
        # print 'rimg shape', rr.rimg.shape
        # print 'rmask shape', rr.rmask.shape
        # print 'rmask elements set:', np.sum(rr.rmask)
        # print 'len I[rmask]:', len(I[rr.rmask])
        mask = (rr.rmask & 1).astype(bool)
        if len(I[mask]):
            plt.subplot(R,C,1)
            plo,phi = [np.percentile(I[mask], p) for p in [25,99]]
            plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                       vmin=plo, vmax=phi)
            plt.xticks([]); plt.yticks([])
            plt.title('rimg')

        plt.subplot(R,C,2)
        I = subco
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.xticks([]); plt.yticks([])
        plt.title('subco')
        plt.subplot(R,C,3)
        I = subpp
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.xticks([]); plt.yticks([])
        plt.title('subpp')
        plt.subplot(R,C,4)
        plt.imshow(rchi, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=-5, vmax=5)
        plt.xticks([]); plt.yticks([])
        plt.title('rchi (%i)' % mm.nrchipix)

        plt.subplot(R,C,8)
        plt.imshow(np.abs(rchi) >= 5., interpolation='nearest', origin='lower',
                   cmap='gray', vmin=0, vmax=1)
        plt.xticks([]); plt.yticks([])
        plt.title('bad rchi')

        plt.subplot(R,C,5)
        I = rr.img
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.xticks([]); plt.yticks([])
        plt.title('img')

        plt.subplot(R,C,6)
        I = mm.omask
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=0, vmax=3)
        plt.xticks([]); plt.yticks([])
        plt.title('omask')

        plt.subplot(R,C,7)
        I = rr.rimg
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.xticks([]); plt.yticks([])
        plt.title('patched rimg')

        # plt.subplot(R,C,8)
        # I = (coimgb / np.maximum(cowb, tinyw))
        # plo,phi = [np.percentile(I, p) for p in [25,99]]
        # plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
        #            vmin=plo, vmax=phi)
        # plt.xticks([]); plt.yticks([])
        # plt.title('coimgb')

        I = (rchi != 0.)
        n,e = np.histogram(np.clip(rchi[I], -rchihistrange, rchihistrange),
                           **rchihistargs)
        if rchihist is None:
            rchihist, rchihistedges = n,e
        else:
            rchihist += n

        plt.subplot(R,C,9)
        e = rchihistedges
        e = (e[:-1]+e[1:])/2.
        #plt.semilogy(e, np.maximum(0.1, rchihist), 'b-')
        plt.semilogy(e, np.maximum(0.1, n), 'b-')
        plt.axvline(5., color='r')
        plt.xlim(-(rchihistrange+1), rchihistrange+1)
        plt.yticks([])
        plt.title('rchi')

        inc = ''
        if not mm.included:
            inc = '(not incl)'
        plt.suptitle('%s %s' % (scanid, inc))
        plt.savefig(plotfn)

    debug(Time() - t00)
    return mm

class coaddacc():
    '''Second-round coadd accumulator.'''
    def __init__(self, H,W, do_cube=False, nims=0, bgmatch=False,
                 minmax=False):
        self.coimg    = np.zeros((H,W))
        self.coimgsq  = np.zeros((H,W))
        self.cow      = np.zeros((H,W))
        self.con      = np.zeros((H,W), np.int16)
        self.coimgb   = np.zeros((H,W))
        self.coimgsqb = np.zeros((H,W))
        self.cowb     = np.zeros((H,W))
        self.conb     = np.zeros((H,W), np.int16)

        self.bgmatch = bgmatch

        self.minmax = minmax
        if minmax:
            self.comin  = np.empty((H,W))
            self.comax  = np.empty((H,W))
            self.cominb = np.empty((H,W))
            self.comaxb = np.empty((H,W))
            self.comin [:,:] =  1e30
            self.cominb[:,:] =  1e30
            self.comax [:,:] = -1e30
            self.comaxb[:,:] = -1e30
        else:
            self.comin  = None
            self.comax  = None
            self.cominb = None
            self.comaxb = None

        if do_cube:
            self.cube = np.zeros((nims, H, W), np.float32)
            self.cubei = 0
        else:
            self.cube = None

    def finish(self):
        if self.minmax:
            # Set pixels that weren't changed from their initial values to zero.
            self.comin [self.comin  ==  1e30] = 0.
            self.cominb[self.cominb ==  1e30] = 0.
            self.comax [self.comax  == -1e30] = 0.
            self.comaxb[self.comaxb == -1e30] = 0.
            
    def acc(self, mm, delmm=False):
        if mm is None or not mm.included:
            return

        if self.bgmatch:
            pass

        self.coimgsq [mm.coslc] += mm.coimgsq
        self.coimg   [mm.coslc] += mm.coimg
        self.cow     [mm.coslc] += mm.cow
        self.con     [mm.coslc] += mm.con
        self.coimgsqb[mm.coslc] += mm.rmask2 * mm.coimgsq
        self.coimgb  [mm.coslc] += mm.rmask2 * mm.coimg
        self.cowb    [mm.coslc] += mm.rmask2 * mm.cow
        self.conb    [mm.coslc] += mm.rmask2 * mm.con
        if self.cube is not None:
            self.cube[(self.cubei,) + mm.coslc] = (mm.coimg).astype(self.cube.dtype)
            self.cubei += 1
        if self.minmax:

            debug('mm.coslc:', mm.coslc)
            debug('mm.con:', np.unique(mm.con), mm.con.dtype)
            debug('mm.rmask2:', np.unique(mm.rmask2), mm.rmask2.dtype)

            self.comin[mm.coslc][mm.con] = np.minimum(self.comin[mm.coslc][mm.con],
                                                      mm.coimg[mm.con] / mm.w)
            self.comax[mm.coslc][mm.con] = np.maximum(self.comax[mm.coslc][mm.con],
                                                      mm.coimg[mm.con] / mm.w)
            self.cominb[mm.coslc][mm.rmask2] = np.minimum(self.cominb[mm.coslc][mm.rmask2],
                                                          mm.coimg[mm.rmask2] / mm.w)
            self.comaxb[mm.coslc][mm.rmask2] = np.maximum(self.comaxb[mm.coslc][mm.rmask2],
                                                          mm.coimg[mm.rmask2] / mm.w)

            debug('comin',  self.comin.min(),  self.comin.max())
            debug('comax',  self.comax.min(),  self.comax.max())
            debug('cominb', self.cominb.min(), self.cominb.max())
            debug('comaxb', self.comaxb.min(), self.comaxb.max())

        if delmm:
            del mm.coimgsq
            del mm.coimg
            del mm.cow
            del mm.con
            del mm.rmask2


def binimg(img, b):
    hh,ww = img.shape
    hh = int(hh / b) * b
    ww = int(ww / b) * b
    return (reduce(np.add, [img[i/b:hh:b, i%b:ww:b] for i in range(b*b)]) /
            float(b*b))

def coadd_wise(tile, cowcs, WISE, ps, band, mp1, mp2,
               do_cube, medfilt, plots2=False, table=True, do_dsky=False,
               bgmatch=False, minmax=False, rchi_fraction=0.01, do_cube1=False):
    L = 3
    W = int(cowcs.get_width())
    H = int(cowcs.get_height())
    # For W4, single-image ww is ~ 1e-10
    tinyw = 1e-16

    # Round-1 coadd:
    (rimgs, coimg1, cow1, coppstd1, cowimgsq1, cube1)= _coadd_wise_round1(
        cowcs, WISE, ps, band, table, L, tinyw, mp1, medfilt,
        bgmatch, do_cube1)
    cowimg1 = coimg1 * cow1
    assert(len(rimgs) == len(WISE))

    if mp1 != mp2:
        debug('Shutting down multiprocessing pool 1')
        mp1.close()

    if do_cube1:
        ofn = '%s-w%i-cube1.fits' % (tile, band)
        fitsio.write(ofn, cube1, clobber=True)
        debug('Wrote', ofn)

        ofn = '%s-w%i-coimg1.fits' % (tile, band)
        fitsio.write(ofn, coimg1, clobber=True)
        debug('Wrote', ofn)

        ofn = '%s-w%i-cow1.fits' % (tile, band)
        fitsio.write(ofn, cow1, clobber=True)
        debug('Wrote', ofn)

        ofn = '%s-w%i-coppstd1.fits' % (tile, band)
        fitsio.write(ofn, coppstd1, clobber=True)
        debug('Wrote', ofn)

    if ps:
        # Plot round-one images
        plt.figure(figsize=(8,8))

        # these large subplots were causing memory errors on carver...
        grid = False
        if not grid:
            plt.figure(figsize=(4,4))

        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99,
                            hspace=0.05, wspace=0.05)
        #plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9,
        #                    hspace=0.05, wspace=0.05)

        if True:
            ngood = len([x for x in rimgs if x is not None])
            cols = int(np.ceil(np.sqrt(float(ngood))))
            rows = int(np.ceil(ngood / float(cols)))
            print('ngood', ngood, 'rows,cols', rows,cols)

            if medfilt:
                sum_medfilt = np.zeros((H,W))
                sum_medfilt2 = np.zeros((H,W))
                n_medfilt = np.zeros((H,W), int)

                for rr in rimgs:
                    if rr is None:
                        continue
                    cox0,cox1,coy0,coy1 = rr.coextent
                    slc = slice(coy0,coy1+1), slice(cox0,cox1+1)

                    sum_medfilt [slc] += rr.rmedfilt
                    sum_medfilt2[slc] += rr.rmedfilt**2
                    n_medfilt   [slc][(rr.rmask & 1)>0] += 1

                mean_medfilt = sum_medfilt / n_medfilt
                std_medfilt = np.sqrt(sum_medfilt2 / n_medfilt - mean_medfilt**2)

                plt.clf()
                plt.imshow(mean_medfilt, interpolation='nearest', origin='lower')
                plt.colorbar()
                plt.title('Mean median filter')
                ps.savefig()

                plt.clf()
                plt.imshow(std_medfilt, interpolation='nearest', origin='lower')
                plt.colorbar()
                plt.title('Median filter standard deviation')
                ps.savefig()


        if False:
            stacks = []

            stack1 = []
            stack2 = []
            stack3 = []
            stack4 = []
            stack5 = []
            stack6 = []
            for j,rr in enumerate(rimgs):
                if rr is None:
                    continue

                sig1 = np.sqrt(1./rr.w)
                kwa = dict(interpolation='nearest', origin='lower',
                           vmin=-2.*sig1, vmax=3.*sig1, cmap='gray')
                rkwa = kwa.copy()
                rkwa.update(extent=rr.coextent)

                for shim,st,skwa in [(rr.rimg, stack1, rkwa),
                                     (rr.img,  stack2, kwa )]:
                    h,w = shim.shape
                    b = int(max(w,h) / 256)
                    if b>1:
                        shim = binimg(shim, b)
                    st.append((shim, skwa))
                if medfilt:
                    med = median_f(rr.medfilt.astype(np.float32).ravel())

                    for shim,st,skwa in [(rr.medfilt - med, stack3, kwa),
                                         (rr.medfilt - med + rr.img, stack4, kwa),
                                         (rr.rmedfilt, stack5, rkwa),
                                         (rr.rmedfilt + rr.rimg, stack6, rkwa)]:
                        h,w = shim.shape
                        b = int(max(w,h) / 256)
                        if b>1:
                            shim = binimg(shim, b)
                        st.append((shim, skwa))
                    
            stacks.append(stack2)
            if medfilt:
                stacks.append(stack3)
                stacks.append(stack4)
            stacks.append(stack1)
            if medfilt:
                stacks.append(stack5)
                stacks.append(stack6)

            if grid:
                for stack in stacks:
                    plt.clf()
                    for i,(im,kwa) in enumerate(stack):
                        plt.subplot(rows, cols, i+1)
                        plt.imshow(im, **kwa)
                        plt.xticks([]); plt.yticks([])
                    ps.savefig()
            else:
                # for stack in stacks:
                #     for i,(im,kwa) in enumerate(stack):
                #         plt.clf()
                #         plt.imshow(im, **kwa)
                #         plt.colorbar()
                #         plt.xticks([]); plt.yticks([])
                #         ps.savefig()
                #s1,s2,s3,s4,s5,s6 = stacks
                for i in range(len(stacks[0])):
                    plt.clf()
                    for j,stack in enumerate(stacks):
                        plt.subplot(2,3, j+1)
                        im,kwa = stack[i]
                        plt.imshow(im, **kwa)
                        if j >= 3:
                            plt.axis([0, W, 0, H])
                        plt.xticks([]); plt.yticks([])
                    ps.savefig()

            plt.clf()
            ploti = 0
            for j,rr in enumerate(rimgs):
                if rr is None:
                    continue
                fullimg = fitsio.read(WISE.intfn[j])
                fullimg -= rr.sky
                ploti += 1
                plt.subplot(rows, cols, ploti)
                print('zpscale', rr.zpscale)
                sig1 = np.sqrt(1./rr.w) / rr.zpscale
                plt.imshow(fullimg, interpolation='nearest', origin='lower',
                           vmin=-2.*sig1, vmax=3.*sig1)
                plt.xticks([]); plt.yticks([])
            ps.savefig()
    
            plt.clf()
            ploti = 0
            for j,rr in enumerate(rimgs):
                if rr is None:
                    continue
                fullimg = fitsio.read(WISE.intfn[j])
                binned = reduce(np.add, [fullimg[i/4::4, i%4::4] for i in range(16)])
                binned /= 16.
                binned -= rr.sky
                ploti += 1
                plt.subplot(rows, cols, ploti)
                sig1 = np.sqrt(1./rr.w) / rr.zpscale
                plt.imshow(binned, interpolation='nearest', origin='lower',
                           vmin=-2.*sig1, vmax=3.*sig1)
                plt.xticks([]); plt.yticks([])
            ps.savefig()

        # Plots of round-one per-image results.
        plt.figure(figsize=(4,4))
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        ngood = 0
        for i,rr in enumerate(rimgs):
            if ngood >= 5:
                break
            if rr is None:
                continue
            if rr.ncopix < 0.25 * W*H:
                continue
            ngood += 1
            print('Plotting rr', i)
            plt.clf()
            cim = np.zeros((H,W))
            # Make untouched pixels white.
            cim += 1e10
            cox0,cox1,coy0,coy1 = rr.coextent
            slc = slice(coy0,coy1+1), slice(cox0,cox1+1)
            mask = (rr.rmask & 1).astype(bool)
            cim[slc][mask] = rr.rimg[mask]
            sig1 = 1./np.sqrt(rr.w)
            plt.imshow(cim, interpolation='nearest', origin='lower', cmap='gray',
                       vmin=-1.*sig1, vmax=5.*sig1)
            ps.savefig()

            cmask = np.zeros((H,W), bool)
            cmask[slc] = mask
            plt.clf()
            # invert
            plt.imshow(cmask, interpolation='nearest', origin='lower',
                       cmap='gray', vmin=0, vmax=1)
            ps.savefig()

            mask2 = (rr.rmask & 2).astype(bool)
            cmask[slc] = mask2
            plt.clf()
            plt.imshow(cmask, interpolation='nearest', origin='lower',
                       cmap='gray', vmin=0, vmax=1)
            ps.savefig()

        sig1 = 1./np.sqrt(np.median(cow1))
        plt.clf()
        plt.imshow(coimg1, interpolation='nearest', origin='lower',
                   cmap='gray', vmin=-1.*sig1, vmax=5.*sig1)
        ps.savefig()

        plt.clf()
        plt.imshow(cow1, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=0, vmax=cow1.max())
        ps.savefig()

        coppstd  = np.sqrt(np.maximum(0, cowimgsq1  /
                                      (np.maximum(cow1,  tinyw)) - coimg1**2))
        mx = np.percentile(coppstd.ravel(), 99)
        plt.clf()
        plt.imshow(coppstd, interpolation='nearest', origin='lower',
                   cmap='gray', vmin=0, vmax=mx)
        ps.savefig()


    plt.figure(figsize=(8,6))

    # If we're not multiprocessing, do the loop manually to reduce
    # memory usage (so we don't need to keep all "rr" inputs and
    # "masks" outputs in memory at once).
    t0 = Time()
    debug('Before garbage collection:', Time()-t0)
    gc.collect()
    debug('After garbage collection:', Time()-t0)
    ps1 = (ps is not None)
    delmm = (ps is None)
    if not mp2.pool:
        coadd = coaddacc(H, W, do_cube=do_cube, nims=len(rimgs), minmax=minmax)
        masks = []
        ri = -1
        while len(rimgs):
            ri += 1
            rr = rimgs.pop(0)
            if ps and plots2:
                plotfn = ps.getnext()
            else:
                plotfn = None
            scanid = ('scan %s frame %i band %i' %
                      (WISE.scan_id[ri], WISE.frame_num[ri], band))
            mm = _coadd_one_round2(
                (ri, len(WISE), scanid, rr, cow1, cowimg1, cowimgsq1, tinyw,
                 plotfn, ps1, do_dsky, rchi_fraction))
            coadd.acc(mm, delmm=delmm)
            masks.append(mm)
    else:
        args = []
        N = len(WISE)
        for ri,rr in enumerate(rimgs):
            if ps and plots2:
                plotfn = ps.getnext()
            else:
                plotfn = None
            scanid = ('scan %s frame %i band %i' %
                      (WISE.scan_id[ri], WISE.frame_num[ri], band))
            args.append((ri, N, scanid, rr, cow1, cowimg1, cowimgsq1, tinyw,
                         plotfn, ps1, do_dsky, rchi_fraction))
        Nimgs = len(rimgs)
        del rimgs

        maskiter = mp2.imap_unordered(_bounce_one_round2, args)
        del args
        info('Accumulating second-round coadds...')
        coadd = coaddacc(H, W, do_cube=do_cube, nims=Nimgs, bgmatch=bgmatch,
                         minmax=minmax)
        t0 = Time()
        inext = 16
        i = 0
        masks = []
        while True:
            try:
                mm = next(maskiter)
            except StopIteration:
                break
            if mm is None:
                continue
            coadd.acc(mm, delmm=delmm)
            masks.append(mm)
            i += 1
            if i == inext:
                inext *= 2
                info('Accumulated', i, 'of', Nimgs, ':', Time()-t0)

    coadd.finish()

    t0 = Time()
    debug('Before garbage collection:', Time()-t0)
    gc.collect()
    debug('After garbage collection:', Time()-t0)

    if ps:
        ngood = 0
        for i,mm in enumerate(masks):
            if ngood >= 5:
                break
            if mm is None or not mm.included:
                continue
            if sum(mm.badpix) == 0:
                continue
            if mm.ncopix < 0.25 * W*H:
                continue
            ngood += 1

            print('Plotting mm', i)

            cim = np.zeros((H,W))
            cim += 1e6
            cim[mm.coslc][mm.rmask_orig] = mm.rimg_orig[mm.rmask_orig]
            w = np.max(mm.cow)
            sig1 = 1./np.sqrt(w)

            cbadpix = np.zeros((H,W))
            cbadpix[mm.coslc][mm.con] = mm.badpix[mm.con]
            blobs,nblobs = label(cbadpix, np.ones((3,3),int))
            blobcms = center_of_mass(cbadpix, labels=blobs,
                                     index=list(range(nblobs+1)))
            plt.clf()
            plt.imshow(cim, interpolation='nearest', origin='lower',
                       cmap='gray', vmin=-1.*sig1, vmax=5.*sig1)
            ax = plt.axis()
            for y,x in blobcms:
                plt.plot(x, y, 'o', mec='r', mew=2, mfc='none', ms=15)
            plt.axis(ax)
            ps.savefig()

            # cim[mm.coslc][mm.rmask_orig] = (mm.rimg_orig[mm.rmask_orig] -
            #                                 coimg1[mm.rmask_orig])
            # plt.clf()
            # plt.imshow(cim, interpolation='nearest', origin='lower',
            #            cmap='gray', vmin=-3.*sig1, vmax=3.*sig1)
            # ps.savefig()

            crchi = np.zeros((H,W))
            crchi[mm.coslc] = mm.rchi
            plt.clf()
            plt.imshow(crchi, interpolation='nearest', origin='lower',
                       cmap='gray', vmin=-5, vmax=5)
            ps.savefig()

            cbadpix[:,:] = 0.5
            cbadpix[mm.coslc][mm.con] = (1 - mm.badpix[mm.con])
            plt.clf()
            plt.imshow(cbadpix, interpolation='nearest', origin='lower',
                       cmap='gray', vmin=0, vmax=1)
            ps.savefig()

    coimg    = coadd.coimg
    coimgsq  = coadd.coimgsq
    cow      = coadd.cow
    con      = coadd.con
    coimgb   = coadd.coimgb
    coimgsqb = coadd.coimgsqb
    cowb     = coadd.cowb
    conb     = coadd.conb
    cube     = coadd.cube

    coimg /= np.maximum(cow, tinyw)
    coinvvar = cow

    coimgb /= np.maximum(cowb, tinyw)
    coinvvarb = cowb

    # per-pixel variance
    coppstd  = np.sqrt(np.maximum(0, coimgsq  / 
                                  np.maximum(cow,  tinyw) - coimg **2))
    coppstdb = np.sqrt(np.maximum(0, coimgsqb /
                                  np.maximum(cowb, tinyw) - coimgb**2))

    # normalize by number of frames to produce an estimate of the
    # stddev in the *coadd* rather than in the individual frames.
    # This is the sqrt of the unbiased estimator of the variance
    coppstd  /= np.sqrt(np.maximum(1., (con  - 1).astype(float)))
    coppstdb /= np.sqrt(np.maximum(1., (conb - 1).astype(float)))

    # re-estimate and subtract sky from the coadd.  approx median:
    #med = median_f(coimgb[::4,::4].astype(np.float32))
    #sig1 = 1./np.sqrt(median_f(coinvvarb[::4,::4].astype(np.float32)))
    try:
        sky = estimate_mode(coimgb)
        #sky = estimate_sky(coimgb, med-2.*sig1, med+1.*sig1, omit=None)
        debug('Estimated coadd sky:', sky)
        coimg  -= sky
        coimgb -= sky
    except np.linalg.LinAlgError:
        print('WARNING: Failed to estimate sky in coadd:')
        import traceback
        traceback.print_exc()
        sky = 0.


    if ps:
        plt.clf()
        I = coimg1
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 1')
        ps.savefig()

        plt.clf()
        I = coppstd1
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd per-pixel std 1')
        ps.savefig()

        plt.clf()
        I = cow1 / np.median([mm.w for mm in masks if mm is not None])
        plo,phi = I.min(), I.max()
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd weights 1 / median w')
        ps.savefig()

        # approx!
        con1 = np.round(I).astype(int)

        plt.clf()
        I = con
        plo,phi = I.min(), I.max()
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 2: N frames')
        ps.savefig()

        plt.clf()
        I = conb
        plo,phi = I.min(), I.max()
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 2: N frames (masked)')
        ps.savefig()

        plt.clf()
        I = con1 - con
        plo,phi = I.min(), I.max()
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd N round 1 - N round 2')
        ps.savefig()

        plt.clf()
        I = con1 - conb
        plo,phi = I.min(), I.max()
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd N round 1 - N round 2 (masked)')
        ps.savefig()


        plt.clf()
        I = coimg
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 2')
        ps.savefig()

        plt.clf()
        I = coimgb
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 2 (weighted)')
        ps.savefig()


        imlo,imhi = plo,phi

        if minmax:
            for I,tt in [(coadd.comin, 'min'), (coadd.comax, 'max'), ((coadd.comin + coadd.comax)/2., 'mean(min,max)'),
                         (coadd.cominb, 'min (weighted)'), (coadd.comaxb, 'max (weighted)'),
                         ((coadd.cominb + coadd.comaxb)/2., 'mean(min,max), weighted')]:
                plt.clf()
                plt.imshow(I - sky, interpolation='nearest', origin='lower', cmap='gray',
                           vmin=plo, vmax=phi)
                plt.colorbar()
                plt.title('Coadd %s' % tt)
                ps.savefig()

            plt.clf()
            plt.imshow(((coimg * con) - (coadd.comin-sky) - (coadd.comax-sky)) / np.maximum(1, con-2),
                       interpolation='nearest', origin='lower', cmap='gray',
                       vmin=plo, vmax=phi)
            plt.colorbar()
            plt.title('Coadd - min,max')
            ps.savefig()

            plt.clf()
            plt.imshow(((coimgb * conb) - (coadd.cominb-sky) - (coadd.comaxb-sky)) / np.maximum(1, conb-2),
                       interpolation='nearest', origin='lower', cmap='gray',
                       vmin=plo, vmax=phi)
            plt.colorbar()
            plt.title('Coadd - min,max (weighted)')
            ps.savefig()

        plt.clf()
        I = coppstd
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 2 per-pixel std')
        ps.savefig()

        plt.clf()
        I = coppstdb
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 2 per-pixel std (weighted)')
        ps.savefig()

        nmax = max(con.max(), conb.max())

        plt.clf()
        I = coppstd
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 2 per-pixel std')
        ps.savefig()


    return (coimg,  coinvvar,  coppstd,  con,
            coimgb, coinvvarb, coppstdb, conb,
            masks, cube, sky,
            coadd.comin, coadd.comax, coadd.cominb, coadd.comaxb)


def estimate_sky(img, lo, hi, omit=None, maxdev=0., return_fit=False):
    # Estimate sky level by: compute the histogram within [lo,hi], fit
    # a parabola to the log-counts, return the argmax of that parabola.
    binedges = np.linspace(lo, hi, 25)
    counts,e = np.histogram(img.ravel(), bins=binedges)
    bincenters = binedges[:-1] + (binedges[1]-binedges[0])/2.

    if omit is not None:
        # Omit the bin containing value 'omit'
        okI = np.logical_not((binedges[:-1] < omit) * (omit < binedges[1:]))
        bincenters = bincenters[okI]
        counts = counts[okI]

    b = np.log10(np.maximum(1, counts))

    if maxdev > 0:
        # log-deviation of a bin from the mean of its neighbors --
        de = (b[1:-1] - (b[:-2] + b[2:])/2)
        print('Max deviation:', np.max(de))
        okI = np.append(np.append([True], (de < maxdev)), [True])
        bincenters = bincenters[okI]
        b = b[okI]

    xscale = 0.5 * (hi - lo)
    x0 = (hi + lo) / 2.
    x = (bincenters - x0) / xscale

    A = np.zeros((len(x), 3))
    A[:,0] = 1.
    A[:,1] = x
    A[:,2] = x**2
    res = np.linalg.lstsq(A, b)
    X = res[0]
    mx = -X[1] / (2. * X[2])
    mx = (mx * xscale) + x0

    if return_fit:
        bfit = X[0] + X[1] * x + X[2] * x**2
        return (x * xscale + x0, b, bfit, mx)

    return mx


def estimate_sky_2(img, lo=None, hi=None, plo=1, phi=70, bins1=30,
                   flo=0.5, fhi=0.8, bins2=30,
                   return_fit=False):
    # Estimate sky level by: compute the histogram within [lo,hi], fit
    # a parabola to the log-counts, return the argmax of that parabola.
    # Coarse bin to find the peak (mode)
    if lo is None:
        lo = np.percentile(img, plo)
    if hi is None:
        hi = np.percentile(img, phi)

    binedges1 = np.linspace(lo, hi, bins1+1)
    counts1,e = np.histogram(img.ravel(), bins=binedges1)
    bincenters1 = binedges1[:-1] + (binedges1[1]-binedges1[0])/2.
    maxbin = np.argmax(counts1)
    maxcount = counts1[maxbin]
    mode = bincenters1[maxbin]

    # Search for bin containing < {flo,fhi} * maxcount
    ilo = maxbin
    while ilo > 0:
        ilo -= 1
        if counts1[ilo] < flo*maxcount:
            break
    ihi = maxbin
    while ihi < bins1-1:
        ihi += 1
        if counts1[ihi] < fhi*maxcount:
            break
    
    lo = bincenters1[ilo]
    hi = bincenters1[ihi]
    
    binedges = np.linspace(lo, hi, bins2)
    counts,e = np.histogram(img.ravel(), bins=binedges)
    bincenters = binedges[:-1] + (binedges[1]-binedges[0])/2.
    
    b = np.log10(np.maximum(1, counts))

    xscale = 0.5 * (hi - lo)
    x0 = (hi + lo) / 2.
    x = (bincenters - x0) / xscale

    A = np.zeros((len(x), 3))
    A[:,0] = 1.
    A[:,1] = x
    A[:,2] = x**2
    res = np.linalg.lstsq(A, b)
    X = res[0]
    mx = -X[1] / (2. * X[2])
    mx = (mx * xscale) + x0

    warn = False
    if not (mx > lo and mx < hi):
        print('WARNING: sky estimate not bracketed by peak: lo %f, sky %f, hi %f' % (lo, mx, hi))
        warn = True
        
    if return_fit:
        bfit = X[0] + X[1] * x + X[2] * x**2
        return (x * xscale + x0, b, bfit, mx, warn, binedges1,counts1)
                

    return mx


def _coadd_one_round1(X):
    '''
    For multiprocessing, the function called to do round 1 on a single
    input frame.
    '''
    (i, N, wise, table, L, ps, band, cowcs, medfilt) = X
    t00 = Time()
    debug('Coadd round 1, image', (i+1), 'of', N)
    intfn = wise.intfn
    uncfn = intfn.replace('-int-', '-unc-')
    if unc_gz:
        uncfn = uncfn + '.gz'
    maskfn = intfn.replace('-int-', '-msk-')
    if mask_gz:
        maskfn = maskfn + '.gz'
    debug('intfn', intfn)
    debug('uncfn', uncfn)
    debug('maskfn', maskfn)

    wcs = wise.wcs
    x0,x1,y0,y1 = wise.imextent
    wcs = wcs.get_subimage(int(x0), int(y0), int(1+x1-x0), int(1+y1-y0))
    slc = (slice(y0,y1+1), slice(x0,x1+1))

    cox0,cox1,coy0,coy1 = wise.coextent
    coW = int(1 + cox1 - cox0)
    coH = int(1 + coy1 - coy0)

    # We read the full images for sky-estimation purposes -- really necessary?
    fullimg,ihdr = fitsio.read(intfn, header=True)
    fullmask = fitsio.read(maskfn)
    fullunc  = fitsio.read(uncfn )
    img  = fullimg [slc]
    mask = fullmask[slc]
    unc  = fullunc [slc]

    zp = ihdr['MAGZP']
    zpscale = 1. / zeropointToScale(zp)
    debug('Zeropoint:', zp, '-> scale', zpscale)

    if band == 4:
        # In W4, the WISE single-exposure images are binned down
        # 2x2, so we are effectively splitting each pixel into 4
        # sub-pixels.  Spread out the flux.
        zpscale *= 0.25

    badbits = [0,1,2,3,4,5,6,7, 9, 
               10,11,12,13,14,15,16,17,18,
               21,26,27,28]
    if wise.phase == 3:
        # 3-band cryo phase:
        ## 19 pixel is "hard-saturated"
        ## 23 for W3 only: static-split droop residual present
        badbits.append(19)
        if band == 3:
            badbits.append(23)

    maskbits = sum([1<<bit for bit in badbits])
    goodmask = ((mask & maskbits) == 0)
    goodmask[unc == 0] = False
    goodmask[np.logical_not(np.isfinite(img))] = False
    goodmask[np.logical_not(np.isfinite(unc))] = False

    sig1 = median_f(unc[goodmask])
    debug('sig1:', sig1)
    del mask
    del unc

    # our return value (quack):
    rr = Duck()
    # Patch masked pixels so we can interpolate
    rr.npatched = np.count_nonzero(np.logical_not(goodmask))
    debug('Pixels to patch:', rr.npatched)
    # Many of the post-cryo frames have ~160,000 masked!
    if rr.npatched > 200000:
        print('WARNING: too many pixels to patch:', rr.npatched)
        return None
    ok = patch_image(img, goodmask.copy())
    if not ok:
        print('WARNING: Patching failed:')
        print('Image size:', img.shape)
        print('Number to patch:', rr.npatched)
        return None
    assert(np.all(np.isfinite(img)))

    # Estimate sky level
    fullok = ((fullmask & maskbits) == 0)
    fullok[fullunc == 0] = False
    fullok[np.logical_not(np.isfinite(fullimg))] = False
    fullok[np.logical_not(np.isfinite(fullunc))] = False

    if medfilt:
        from astrometry.util.util import median_smooth
        from scipy.ndimage.filters import uniform_filter
        tmf0 = Time()
        mf = np.zeros_like(fullimg)
        ok = median_smooth(fullimg, np.logical_not(fullok), int(medfilt), mf)
        # Now mask out significant pixels and repeat the median filter!
        # This method is courtesy of John Moustakas
        # Smooth by a boxcar filter before cutting pixels above threshold --
        boxcar = 5
        # Sigma of boxcar-smoothed image
        bsig1 = sig1 / boxcar
        diff = fullimg - mf
        diff[np.logical_not(fullok)] = 0.
        masked = np.abs(uniform_filter(diff, size=boxcar, mode='constant')) > (3.*bsig1)
        del diff
        masked = binary_dilation(masked, iterations=3)
        masked[np.logical_not(fullok)] = True
        if ps:
            mf1 = mf.copy()
        mf[:,:] = 0.
        h,w = masked.shape
        frac_masked = np.sum(masked) / (h*w)
        debug('%.1f %% of pixels masked by boxcar' % (100.*frac_masked))
        ok = median_smooth(fullimg, masked, int(medfilt), mf)
        fullimg -= mf
        img = fullimg[slc]
        debug('Median filtering with box size', medfilt, 'took', Time()-tmf0)
        if ps:
            plt.clf()
            plt.subplot(2,2,1)
            ima1 = dict(interpolation='nearest', origin='lower',
                       cmap='gray')
            ima = dict(interpolation='nearest', origin='lower', vmin=-3*sig1, vmax=+3*sig1,
                       cmap='gray')
            plt.imshow(fullimg * fullok, **ima)
            plt.title('image')
            plt.subplot(2,2,2)
            plt.imshow(mf1 - np.median(mf[np.isfinite(mf)]), **ima1)
            plt.colorbar()
            plt.title('med.filt 1')
            plt.subplot(2,2,3)
            plt.imshow(masked, interpolation='nearest', origin='lower', vmin=0, vmax=1,
                       cmap='gray')
            plt.title('mask')
            plt.subplot(2,2,4)
            plt.imshow(mf - np.median(mf[np.isfinite(mf)]), **ima1)
            plt.title('med.filt')
            plt.colorbar()
            ps.savefig()

            # save for later...
            rr.medfilt = mf * zpscale
        del mf
        
    # add some noise to smooth out "dynacal" artifacts
    # NOTE -- this is just for the background estimation!
    fim = fullimg[fullok]
    fim += np.random.normal(scale=sig1, size=fim.shape) 
    if ps:
        vals,counts,fitcounts,sky,warn,be1,bc1 = estimate_mode(fim, return_fit=True)
        rr.hist = np.histogram(fullimg[fullok], range=(vals[0],vals[-1]), bins=100)
        rr.skyest = sky
        rr.skyfit = (vals, counts, fitcounts)
        
        if warn:
            # Background estimation plot
            plt.clf()

            # first-round histogram
            ee1 = be1.repeat(2)[1:-1]
            nn1 = bc1.repeat(2)
            plt.plot(ee1, nn1, 'b-', alpha=0.5)

            # full-image histogram
            n,e = rr.hist
            ee = e.repeat(2)[1:-1]
            nn = n.repeat(2)
            plt.plot(ee, nn, 'm-', alpha=0.5)

            # extended range
            n,e = np.histogram(fim, range=(np.percentile(fim, 1),
                                           np.percentile(fim, 90)), bins=100)
            ee = e.repeat(2)[1:-1]
            nn = n.repeat(2)
            plt.plot(ee, nn, 'g-', alpha=0.5)

            plt.twinx()
            plt.plot(vals, counts, 'm-', alpha=0.5)
            plt.plot(vals, fitcounts, 'r-', alpha=0.5)
            plt.axvline(sky, color='r')
            plt.title('%s %i' % (wise.scan_id, wise.frame_num))
            ps.savefig()

            plt.xlim(ee1[0], ee1[-1])
            ps.savefig()
            

    else:
        sky = estimate_mode(fim)

    debug('Estimated sky:', sky)
    debug('Image median:', np.median(fullimg[fullok]))
    debug('Image median w/ noise:', np.median(fim))

    del fim
    del fullunc
    del fullok
    del fullimg
    del fullmask

    # Convert to nanomaggies
    img -= sky
    img  *= zpscale
    sig1 *= zpscale

    # coadd subimage
    cosubwcs = cowcs.get_subimage(int(cox0), int(coy0), coW, coH)
    try:
        Yo,Xo,Yi,Xi,rims = resample_with_wcs(cosubwcs, wcs, [img], L,
                                             table=table)
    except OverlapError:
        debug('No overlap; skipping')
        return None
    rim = rims[0]
    assert(np.all(np.isfinite(rim)))
    debug('Pixels in range:', len(Yo))

    if ps:
        # save for later...
        rr.img = img
        
        if medfilt:
            debug('Median filter: rr.medfilt range', rr.medfilt.min(), rr.medfilt.max())
            debug('Sky:', sky*zpscale)
            med = median_f(rr.medfilt.astype(np.float32).ravel())
            rr.rmedfilt = np.zeros((coH,coW), img.dtype)
            rr.rmedfilt[Yo,Xo] = (rr.medfilt[Yi, Xi].astype(img.dtype) - med)
            debug('rr.rmedfilt range', rr.rmedfilt.min(), rr.rmedfilt.max())

    # Scalar!
    rr.w = (1./sig1**2)
    rr.rimg = np.zeros((coH, coW), img.dtype)
    rr.rimg[Yo, Xo] = rim
    rr.rmask = np.zeros((coH, coW), np.uint8)
    '''
    rr.rmask bit 0 (value 1): This pixel is within the coadd footprint.
    rr.rmask bit 1 (value 2): This pixel is good.
    '''
    rr.rmask[Yo, Xo] = 1 + 2*goodmask[Yi, Xi]
    rr.wcs = wcs
    rr.sky = sky
    rr.zpscale = zpscale
    rr.zp = zp
    rr.ncopix = len(Yo)
    rr.coextent = wise.coextent
    rr.cosubwcs = cosubwcs

    if ps and medfilt and False:
        plt.clf()
        rows,cols = 2,2
        kwa = dict(interpolation='nearest', origin='lower',
                   vmin=-2.*sig1, vmax=3.*sig1, cmap='gray')

        mm = median_f(rr.medfilt.astype(np.float32))
        debug('Median medfilt:', end=' ') 
        #mm = sky * zpscale
        debug('Sky*zpscale:', sky*zpscale)
        
        origimg = rr.img + rr.medfilt - mm

        plt.subplot(rows, cols, 1)
        plt.imshow(binimg(origimg, 4), **kwa)
        plt.title('Image')
        plt.subplot(rows, cols, 2)
        plt.imshow(binimg(rr.medfilt - mm, 4), **kwa)
        plt.title('Median')
        plt.subplot(rows, cols, 3)
        plt.imshow(binimg(rr.img, 4), **kwa)
        plt.title('Image - Median')
        tag = ''
        if wise.moon_masked:
            tag += ' moon'
        plt.suptitle('%s %i%s' % (wise.scan_id, wise.frame_num, tag))
        ps.savefig()

    debug(Time() - t00)
    return rr


def _coadd_wise_round1(cowcs, WISE, ps, band, table, L, tinyw, mp, medfilt,
                       bgmatch, cube1):
                       
    '''
    Do round-1 coadd.
    '''
    W = int(cowcs.get_width())
    H = int(cowcs.get_height())
    coimg   = np.zeros((H,W))
    coimgsq = np.zeros((H,W))
    cow     = np.zeros((H,W))

    args = []
    for wi,wise in enumerate(WISE):
        args.append((wi, len(WISE), wise, table, L, ps, band, cowcs, medfilt))
    rimgs = mp.map(_coadd_one_round1, args)
    del args

    debug('Accumulating first-round coadds...')
    cube = None
    if cube1:
        cube = np.zeros((len([rr for rr in rimgs if rr is not None]), H, W),
                        np.float32)
        z = 0
    t0 = Time()
    for wi,rr in enumerate(rimgs):
        if rr is None:
            continue
        cox0,cox1,coy0,coy1 = rr.coextent
        slc = slice(coy0,coy1+1), slice(cox0,cox1+1)

        if bgmatch:
            # Overlapping pixels:
            I = np.flatnonzero((cow[slc] > 0) * (rr.rmask&1 > 0))
            rr.bgmatch = 0.
            if len(I) > 0:
                bg = median_f(((coimg[slc].flat[I] / cow[slc].flat[I]) - 
                               rr.rimg.flat[I]).astype(np.float32))
                debug('Matched bg:', bg)
                rr.rimg[(rr.rmask & 1) > 0] += bg
                rr.bgmatch = bg
                
        # note, rr.w is a scalar.
        # (rr.rmask & 1) means "use these coadd pixels"
        #  rr.rimg is 0 where that bit is zero, so no need to multiply by
        #  that mask when accumulating here.
        coimgsq[slc] += rr.w * (rr.rimg**2)
        coimg  [slc] += rr.w *  rr.rimg
        cow    [slc] += rr.w * (rr.rmask & 1)

        if cube1:
            cube[(z,)+slc] = rr.rimg.astype(np.float32)
            z += 1
            
        # if ps:
        #     # Show the coadd as it's accumulated
        #     plt.clf()
        #     s1 = np.median([r2.w for r2 in rimgs if r2 is not None])
        #     s1 /= 5.
        #     plt.imshow(coimg / np.maximum(cow, tinyw), interpolation='nearest',
        #                origin='lower', vmin=-2.*s1, vmax=5.*s1)
        #     plt.title('%s %i' % (WISE.scan_id[wi], WISE.frame_num[wi]))
        #     ps.savefig()

    debug(Time()-t0)

    coimg /= np.maximum(cow, tinyw)
    # Per-pixel std
    coppstd = np.sqrt(np.maximum(0, coimgsq / np.maximum(cow, tinyw)
                                 - coimg**2))

    if ps:
        # plt.clf()
        # for rr in rimgs:
        #     if rr is None:
        #         continue
        #     n,e = rr.hist
        #     e = (e[:-1] + e[1:])/2.
        #     plt.plot(e - rr.skyest, n, 'b-', alpha=0.1)
        #     plt.axvline(e[0] - rr.skyest, color='r', alpha=0.1)
        #     plt.axvline(e[-1] - rr.skyest, color='r', alpha=0.1)
        # plt.xlabel('image - sky')
        # ps.savefig()
        # plt.yscale('log')
        # ps.savefig()

        plt.clf()
        for rr in rimgs:
            if rr is None:
                continue
            n,e = rr.hist
            ee = e.repeat(2)[1:-1]
            nn = n.repeat(2)
            plt.plot(ee - rr.skyest, nn, 'b-', alpha=0.1)
        plt.xlabel('image - sky')
        ps.savefig()
        plt.yscale('log')
        ps.savefig()

        plt.clf()
        for rr in rimgs:
            if rr is None:
                continue
            vals, counts, fitcounts = rr.skyfit
            plt.plot(vals - rr.skyest, counts, 'b-', alpha=0.1)
            plt.plot(vals - rr.skyest, fitcounts, 'r-', alpha=0.1)
        plt.xlabel('image - sky')
        plt.title('sky hist vs fit')
        ps.savefig()

        plt.clf()
        o = 0
        for rr in rimgs:
            if rr is None:
                continue
            vals, counts, fitcounts = rr.skyfit
            off = o * 0.01
            o += 1
            plt.plot(vals - rr.skyest, counts + off, 'b.-', alpha=0.1)
            plt.plot(vals - rr.skyest, fitcounts + off, 'r.-', alpha=0.1)
        plt.xlabel('image - sky')
        plt.title('sky hist vs fit')
        ps.savefig()

        plt.clf()
        for rr in rimgs:
            if rr is None:
                continue
            vals, counts, fitcounts = rr.skyfit
            plt.plot(vals - rr.skyest, counts - fitcounts, 'b-', alpha=0.1)
        plt.ylabel('log counts - log fit')
        plt.xlabel('image - sky')
        plt.title('sky hist fit residuals')
        ps.savefig()

        plt.clf()
        for rr in rimgs:
            if rr is None:
                continue
            vals, counts, fitcounts = rr.skyfit
            plt.plot(vals - rr.skyest, counts - fitcounts, 'b.', alpha=0.1)
        plt.ylabel('log counts - log fit')
        plt.xlabel('image - sky')
        plt.title('sky hist fit residuals')
        ps.savefig()

        ha = dict(range=(-8,8), bins=100, log=True, histtype='step')
        plt.clf()
        nn = []
        for rr in rimgs:
            if rr is None:
                continue
            mask = (rr.rmask & 1).astype(bool)
            rim = rr.rimg[mask]
            if len(rim) == 0:
                continue
            #n,b,p = plt.hist(rim, alpha=0.1, **ha)
            #nn.append((n,b))
            n,e = np.histogram(rim, range=ha['range'], bins=ha['bins'])
            lo = 3e-3
            nnn = np.maximum(3e-3, n/float(sum(n)))
            #print 'e', e
            #print 'nnn', nnn
            nn.append((nnn,e))
            plt.semilogy((e[:-1]+e[1:])/2., nnn, 'b-', alpha=0.1)
        plt.xlabel('rimg (-sky)')
        #yl,yh = plt.ylim()
        yl,yh = [np.percentile(np.hstack([n for n,e in nn]), p) for p in [3,97]]
        print('percentiles', yl,yh)
        plt.ylim(yl, yh)
        ps.savefig()

        plt.clf()
        for n,b in nn:
            plt.semilogy((b[:-1] + b[1:])/2., n, 'b.', alpha=0.2)
        plt.xlabel('rimg (-sky)')
        plt.ylim(yl, yh)
        ps.savefig()

        plt.clf()
        n,b,p = plt.hist(coimg.ravel(), **ha)
        plt.xlabel('coimg')
        plt.ylim(max(1, min(n)), max(n)*1.1)
        ps.savefig()

    return rimgs, coimg, cow, coppstd, coimgsq, cube

def get_wise_frames_for_dataset(dataset, r0,r1,d0,d1,
                                randomize=False, cache=True, dirnm=None, cachefn=None):
    WISE = None
    if cache:
        if cachefn is None:
            fn = '%s-frames.fits' % dataset
        else:
            fn = cachefn
        if dirnm is not None:
            fn = os.path.join(dirnm, fn)
        if os.path.exists(fn):
            print('Reading', fn)
            try:
                WISE = fits_table(fn)
            except:
                pass

    if WISE is None:
        # FIXME -- do we need 'margin' here any more?
        WISE = get_wise_frames(r0,r1,d0,d1)
        # bool -> uint8 to avoid confusing fitsio
        WISE.moon_masked = WISE.moon_masked.astype(np.uint8)
        if randomize:
            print('Randomizing frame order...')
            WISE.cut(np.random.permutation(len(WISE)))

        if cache:
            WISE.writeto(fn)
    # convert to boolean
    WISE.moon_masked = (WISE.moon_masked != 0)
    return WISE

def main():
    import argparse
    from astrometry.util.multiproc import multiproc

    parser = argparse.ArgumentParser('%prog [options]')

    parser.add_argument('--threads', dest='threads', type=int, help='Multiproc',
                      default=None)
    parser.add_argument('--threads1', dest='threads1', type=int, default=None,
                      help='Multithreading during round 1')

    parser.add_argument('-w', dest='wishlist', action='store_true',
                      default=False, help='Print needed frames and exit?')
    parser.add_argument('--plots', dest='plots', action='store_true',
                      default=False)
    parser.add_argument('--plots2', dest='plots2', action='store_true',
                      default=False)
    parser.add_argument('--pdf', dest='pdf', action='store_true', default=False)

    parser.add_argument('--plot-prefix', dest='plotprefix', default=None)

    parser.add_argument('--outdir', '-o', dest='outdir', default='unwise-coadds',
                      help='Output directory: default %(default)s')

    parser.add_argument('--size', dest='size', default=2048, type=int,
                      help='Set output image size in pixels; default %(default)s')
    parser.add_argument('--width', dest='width', default=0, type=int,
                      help='Set output image width in pixels; default --size')
    parser.add_argument('--height', dest='height', default=0, type=int,
                      help='Set output image height in pixels; default --size')

    parser.add_argument('--pixscale', dest='pixscale', type=float, default=2.75,
                      help='Set coadd pixel scale, default %(default)s arcsec/pixel')
    parser.add_argument('--cube', dest='cube', action='store_true',
                      default=False, help='Save & write out image cube')
    parser.add_argument('--cube1', dest='cube1', action='store_true',
                      default=False, help='Save & write out image cube for round 1')

    parser.add_argument('--frame0', dest='frame0', default=0, type=int,
                      help='Only use a subset of the frames: starting with frame0')
    parser.add_argument('--nframes', dest='nframes', default=0, type=int,
                      help='Only use a subset of the frames: number nframes')
    parser.add_argument('--nframes-random', dest='nframes_random', default=0, type=int,
                      help='Only use a RANDOM subset of the frames: number nframes')

    parser.add_argument('--medfilt', dest='medfilt', type=int, default=None,
                      help=('Median filter with a box twice this size (+1),'+
                            ' to remove varying background.  Default: none for W1,W2; 50 for W3,W4.'))

    parser.add_argument('--force', dest='force', action='store_true',
                      default=False, 
                      help='Run even if output file already exists?')

    parser.add_argument('--maxmem', dest='maxmem', type=float, default=0,
                      help='Quit if predicted memory usage > n GB')

    parser.add_argument('--dsky', dest='dsky', action='store_true',
                      default=False,
                      help='Do background-matching by matching medians '
                      '(to first-round coadd)')

    parser.add_argument('--bgmatch', dest='bgmatch', action='store_true',
                      default=False,
                      help='Do background-matching by matching medians '
                      '(when accumulating first-round coadd)')

    parser.add_argument('--center', dest='center', action='store_true',
                      default=False,
                      help='Read frames in order of distance from center; for debugging.')

    parser.add_argument('--minmax', action='store_true',
                      help='Record the minimum and maximum values encountered during coadd?')

    parser.add_argument('--ra', dest='ra', type=float, default=None,
                      help='Build coadd at given RA center')
    parser.add_argument('--dec', dest='dec', type=float, default=None,
                      help='Build coadd at given Dec center')
    parser.add_argument('--band', type=int, default=None, action='append',
                      help='with --ra,--dec: band(s) to do (1,2,3,4)')

    parser.add_argument('--zoom', type=int, nargs=4,
                        help='Set target image extent (default "0 2048 0 2048")')
    parser.add_argument('--grid', type=int, nargs='?', default=0, const=2048,
                        help='Grid this (large custom) image into pieces of this size.')

    parser.add_argument('--name', default=None,
                      help='Output file name: unwise-NAME-w?-*.fits')

    parser.add_argument('--tile', dest='tile', type=str, default=None,
                      help='Run a single tile, eg, 0832p196')

    parser.add_argument('--preprocess', dest='preprocess', action='store_true',
                      default=False, help='Preprocess (write *-atlas, *-frames.fits) only')

    parser.add_argument('--rchi-fraction', dest='rchi_fraction', type=float,
                      default=0.01, help='Fraction of outlier pixels to reject frame')

    parser.add_argument('--epoch', type=int, help='Keep only input frames in the given epoch, zero-indexed')

    parser.add_argument('--before', type=float, help='Keep only input frames before the given MJD')
    parser.add_argument('--after',  type=float, help='Keep only input frames after the given MJD')

    parser.add_argument('--no-download', dest='download', default=True, action='store_false',
                      help='Do not download data from IRSA, assume it is already on disk')

    parser.add_argument('--cache-frames', help='For custom --ra,--dec coadds, cache the overlapping frames in this file.')

    parser.add_argument('--period', type=float, help='Build a series of coadds separated by this period, in days.')

    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')

    opt = parser.parse_args()

    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    global logger
    logger = logging.getLogger('unwise_coadd')

    if opt.threads:
        mp2 = multiproc(opt.threads)
    else:
        mp2 = multiproc()
    if opt.threads1 is None:
        mp1 = mp2
    else:
        mp1 = multiproc(opt.threads1)

    radec = opt.ra is not None and opt.dec is not None

    if not radec or opt.tile:
        print('Must specify --ra,--dec or --tile')
        parser.print_help()
        return -1

    print('unwise_coadd.py starting: args:', sys.argv)
    #print('opt:', opt)
    #print(dir(opt))

    Time.add_measurement(MemMeas)

    W = H = opt.size
    if opt.width:
        W = opt.width
    if opt.height:
        H = opt.height

    randomize = False
    pmargin = 1.05
    pallsky = False
    plotargs = {}

    if radec:
        dataset = ('custom-%04i%s%03i' %
                   (int(opt.ra*10.), 'p' if opt.dec >= 0. else 'm', int(np.abs(opt.dec)*10.)))
        print('Setting custom dataset', dataset)
        # fake tiles table
        tiles = fits_table()
        tiles.coadd_id = np.array([dataset])
        tiles.ra = np.array([opt.ra])
        tiles.dec = np.array([opt.dec])
    else:
        # parse opt.tile
        if len(opt.tile) != 8:
            print('--tile expects string like RRRR[pm]DDD')
            return -1
        # Read Atlas Image table to find tile id
        fn = os.path.join(wisedir, 'wise_allsky_4band_p3as_cdd.fits')
        print('Reading', fn)
        tiles = fits_table(fn, columns=['coadd_id', 'ra', 'dec'])
        print('Read', len(tiles), 'Atlas tiles')
        tiles.coadd_id = np.array([c.replace('_ab41','') for c in tiles.coadd_id])
        I = np.flatnonzero(tiles.coadd_id == np.array(opt.tile).astype(tiles.coadd_id.dtype))
        if len(I) != 1:
            print('Found', len(I), '(not 1) tiles matching desired', opt.tile)
            return -1
        tiles.cut(I)
        dataset = opt.tile

    # In the olden days, we ran multiple tiles
    assert(len(tiles) == 1)
    tile = tiles[0]

    cosd = np.cos(np.deg2rad(tile.dec))
    r0 = tile.ra - (opt.pixscale * W/2.)/3600. / cosd
    r1 = tile.ra + (opt.pixscale * W/2.)/3600. / cosd
    d0 = tile.dec - (opt.pixscale * H/2.)/3600.
    d1 = tile.dec + (opt.pixscale * H/2.)/3600.

    if opt.name:
        tile.coadd_id = opt.name

    # cache the file DATASET-frames.fits ?
    cache = True
    cachefn = None
    if radec:
        if opt.cache_frames:
            cachefn = opt.cache_frames
            cache = True
        else:
            cache = False

    WISE = get_wise_frames_for_dataset(dataset, r0,r1,d0,d1, cache=cache, cachefn=cachefn)

    if not os.path.exists(opt.outdir) and not opt.wishlist:
        print('Creating output directory', opt.outdir)
        trymakedirs(opt.outdir)

    if opt.preprocess:
        print('Preprocessing done')
        return 0

    if opt.plots:
        from astrometry.util.plotutils import PlotSequence
        if opt.plotprefix is None:
            opt.plotprefix = dataset
        ps = PlotSequence(opt.plotprefix, format='%03i')
        if opt.pdf:
            ps.suffixes = ['png','pdf']
    else:
        ps = None

    if opt.band is None:
        bands = [1,2]
    else:
        bands = list(opt.band)

    period = opt.period
    grid = opt.grid

    kwargs = vars(opt)
    print('kwargs:', kwargs)
    # rename
    for fr,to in [('dsky', 'do_dsky'),
                  ('cube', 'do_cube'),
                  ('cube1', 'do_cube1'),
                  ('download', 'allow_download'),
                  ]:
        kwargs.update({ to: kwargs.pop(fr) })
    for key in ['threads', 'threads1', 'plots', 'pdf', 'plotprefix',
                'size', 'width', 'height', 'ra', 'dec', 'band', 'name',
                'tile', 'preprocess', 'cache_frames', 'period', 'grid',
                'verbose']:
        kwargs.pop(key)

    if period:
        # Switch to the mode of building short-cadence coadds.
        if not opt.before:
            opt.before = max(WISE.mjd)
        if not opt.after:
            opt.after = min(WISE.mjd)
        epochs = np.arange(opt.after, opt.before, period)

        # if not opt.cache_frames:
        #     f,fn = tempfile.mkstemp()
        #     os.close(f)
        #     opt.cache_frames = fn
        #     WISE.writeto(fn)
            
        todo = []
        keep = []
        nframes_w1 = []
        nframes_w2 = []

        kwargs.update(write_masks=False, force_outdir=True)
        
        epnum = 0
        for i,epoch in enumerate(epochs):
            epdir = os.path.join(opt.outdir, 'ep%04i' % epnum)
            #trymakedirs(epdir)
            kw = kwargs.copy()
            kw.update(outdir=epdir)

            I, = np.nonzero((WISE.mjd >= epoch) *
                            (WISE.mjd < epoch+period))
            if len(I) == 0:
                print('No coverage for period', epoch, 'to', epoch+period)
                continue
            I = I[np.array([WISE.band[ii] in bands for ii in I])]
            if len(I) == 0:
                print('No coverage for period', epoch, 'to', epoch+period)
                continue
            keep.append(i)
            nframes_w1.append(np.sum(WISE.band[I] == 1))
            nframes_w2.append(np.sum(WISE.band[I] == 2))
            epnum += 1

            for band in bands:
                todo.append((epdir, (tile, band, W, H, WISE[I]), kw))

        keep = np.array(keep)
        epochs = epochs[keep]
        print('Total of', len(epochs), 'epochs to run')

        rtnvals = mp2.map(bounce_one_epoch, todo)

        rtnvals = np.array(rtnvals)
        
        # measure coverage of central pixel in each epoch??

        summary = fits_table()
        summary.epochnum = np.arange(len(rtnvals))
        summary.start = epochs
        summary.finish = epochs + period
        ## always the same??
        summary.nframes_w1 = np.array(nframes_w1)
        summary.nframes_w2 = np.array(nframes_w2)

        summary.cut(rtnvals == 0)

        summary.writeto(os.path.join(opt.outdir, 'summary.fits'))
                
        sys.exit(0)

    if grid:
        orig_name = tile.coadd_id
        nw = int(np.ceil(W / float(grid)))
        nh = int(np.ceil(H / float(grid)))
        cowcs = get_coadd_tile_wcs(tile.ra, tile.dec, W, H, opt.pixscale)
        for band in bands:
            for y in range(nh):
                for x in range(nw):
                    t0 = Time()
                    print('Doing coadd grid tile', tile.coadd_id, 'band', band, 'x,y', x,y)
                    kwcopy = kwargs.copy()
                    kwcopy['zoom'] = (x*grid, min((x+1)*grid, W),
                                      y*grid, min((y+1)*grid, H))
                    kwcopy.update(ps=ps, mp1=mp1, mp2=mp2)
                    tile.coadd_id = orig_name + '_grid_%i_%i' % (x, y)
                    if one_coadd(tile, band, W, H, WISE, **kwcopy):
                        return -1
                    print('Grid tile', tile.coadd_id, 'band', band, 'x,y', x,y, ':', Time()-t0)
            frames = []
            for suffix in ['-img-m.fits', '-invvar-m.fits', '-std-m.fits', '-n-m.fits',
                           '-img-u.fits', '-invvar-u.fits', '-std-u.fits', '-n-u.fits',
                           '-frames.fits']:
                dtype = np.float32
                if '-n-' in suffix:
                    dtype = np.int32
                elif 'frames' in suffix:
                    dtype = None

                if dtype is not None:
                    hdr = fitsio.FITSHDR()
                    hdr.add_record(dict(name='UNW_GRID', value=grid, comment='Grid size for sub-coadds'))
                    cowcs.add_to_header(hdr)
                    img = np.zeros((H,W), dtype)
                for y in range(nh):
                    for x in range(nw):
                        coadd_id = orig_name + '_grid_%i_%i' % (x, y)
                        tag = 'unwise-%s-w%i' % (coadd_id, band)
                        indir = opt.outdir
                        indir = get_dir_for_coadd(indir, coadd_id)
                        prefix = os.path.join(indir, tag)
                        fn = prefix + suffix
                        print('Reading', fn)
                        if dtype is not None:
                            gimg,ghdr = fitsio.read(fn, header=True)
                            #if x == 0 and y == 0:
                            #    hdr = ghdr
                            img[y*grid : min((y+1)*grid, H),
                                x*grid : min((x+1)*grid, W)] = gimg
                            del gimg
                            for r in ghdr.records():
                                key = r['name']
                                if key == 'UNW_SKY':
                                    hdr.add_record(dict(name='UNSK%i_%i' % (x,y),
                                                        value=r['value'],
                                                        comment='UNW_SKY (subtracted) from tile %i,%i' % (x,y)))
                                elif key == 'MAGZP' or key.startswith('UNW_'):
                                    hdr.add_record(r)
                        else:
                            gf = fits_table(fn)
                            gf.grid_x = np.zeros(len(gf), np.int16) + x
                            gf.grid_y = np.zeros(len(gf), np.int16) + y
                            frames.append(gf)
                tag = 'unwise-%s-w%i' % (orig_name, band)
                outdir = opt.outdir
                outdir = get_dir_for_coadd(outdir, orig_name)
                prefix = os.path.join(outdir, tag)
                fn = prefix + suffix
                print('Writing', fn)
                if dtype is not None:
                    fitsio.write(fn, img, clobber=True, header=hdr)
                    del img
                else:
                    frames = merge_tables(frames)
                    frames.writeto(fn)
                    del frames

        #tile.coadd_id = orig_name
        sys.exit(0)
        
    for band in bands:
        print('Doing coadd tile', tile.coadd_id, 'band', band)
        t0 = Time()

        medfilt = opt.medfilt
        if medfilt is None:
            if band in [3,4]:
                medfilt = 50
            else:
                medfilt = 0

        kwargs.update(ps=ps, mp1=mp1, mp2=mp2)

        if one_coadd(tile, band, W, H, WISE, **kwargs):
            return -1
        print('Tile', tile.coadd_id, 'band', band, 'took:', Time()-t0)
    return 0

def bounce_one_epoch(X):
    dirnm, args, kwargs = X
    rtn = one_coadd(*args, **kwargs)
    print('Finished', dirnm, 'with return value', rtn)
    return rtn

if __name__ == '__main__':
    sys.exit(main())


# python -u unwise_coadd.py --ra 10.68 --dec 41.27 --force 1000 --plots > m31.log
# python -u unwise_coadd.py --ra 80.63 --dec 33.43 1000 > tad-1.log 2>&1 &
# python -u unwise_coadd.py --ra 83.8 --dec -5.39 1000 > orion-1.log 2>&1 &
# M1 (Crab Nebula): python -u unwise_coadd.py --ra 83.6 --dec 22.0 -o data/unwise 1000
# M8 (Lagoon Nebula): python -u unwise_coadd.py --ra 270.9 --dec -24.4 -o data/unwise 1000 > 1b.log 2>&1 &
# M16 (Eagle nebula): python -u unwise_coadd.py --ra 274.7 --dec -13.8 -o data/unwise 1000 > 1c.log 2>&1 &
# M17 (Omega nebula)
# M20 (Trifid nebula)
# M27 (Dumbbell nebula)
# M32 (andromeda companion)
# M33 (Triangulum)
# M42 (Orion nebula)
# M43 (nebula in Orion)
# M45 (Pleiades)
# M51 (Whirlpool galaxy)
# M57 (ring nebula)
# M58, 61, 65 (spiral)
# M49, 59, M60 (elliptical)
# M63 (Sunflower galaxy)
# M64 (Black eye galaxy)

