#! /usr/bin/env python
from time import time as _time
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import numpy as np
from copy import deepcopy
#import pylab as plt
import os
import sys
import tempfile
import datetime
import gc
from functools import reduce
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.measurements import label
from zp_lookup import ZPLookUp
import random
from warp_utils import WarpMetaParameters, mask_extreme_pix, compute_warp, apply_warp, gen_warp_table, update_included_bitmask, parse_write_quadrant_masks, RecoveryStats, pad_rebin_weighted, ReferenceImage, QuadrantWarp, reference_image_from_dir
from unwise_utils import tile_to_radec, int_from_scan_frame, zeropointToScale, retrieve_git_version, get_dir_for_coadd, get_epoch_breaks, get_coadd_tile_wcs, get_l1b_file, download_frameset_1band, sanity_check_inputs, phase_from_scanid, header_reference_keywords, get_l1b_dirs, is_nearby, good_scan_mask, ascending
from hi_lo import HiLo

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

#median_f = np.median
median_f = flat_median_f

# GLOBALS:
# Location of WISE Level 1b inputs
unwise_symlink_dir = os.environ.get('UNWISE_SYMLINK_DIR')
if unwise_symlink_dir is None:
    unwise_symlink_dir = '/scratch1/scratchdirs/ameisner/code/unwise-coadds'

wisedir = os.path.join(unwise_symlink_dir, 'wise-frames')

mask_gz = True
unc_gz = True
int_gz = None # should get assigned in main
use_zp_meta = None # should get assigned in main
compare_moon_all = None # should get assigned in main

class FirstRoundCoadd():
    def __init__(self, coimg1, cow1, coppstd1, cowimgsq1, con1):
        self.coimg1 = coimg1
        self.cow1 = cow1
        self.coppstd1 = coppstd1
        self.cowimgsq1 = cowimgsq1
        self.cowimg1 = coimg1*cow1
        self.con1 = con1 # integer coverage, "unmasked"

class FirstRoundImage():
    def __init__(self, quadrant=-1):
        self.coextent = None
        self.cosubwcs = None
        self.ncopix = None
        self.npatched = None
        self.rimg = None
        self.rmask = None
        self.sky = None
        self.w = None
        self.wcs = None
        self.zp = None
        self.pa = None # for john fowler
        self.ascending = None # for john fowler
        self.zpscale = None
        self.quadrant = quadrant
        self.included_round1 = False
        # optional
        self.x_l1b = None
        self.y_l1b = None
        self.x_coadd = None
        self.y_coadd = None
        self.wcs_full = None
        self.cowcs_full = None
        self.warp = None # of type QuadrantWarp
        self.warped = False
        self.rimg_bak = None # for debugging only; once warping performed, will hold raw image
        self.scan_id = None
        self.frame_num = None

    def clear_xy_coords(self):
        print "deleting x, y coordinates for quadrant " + str(self.quadrant)
        del self.x_l1b, self.y_l1b, self.x_coadd, self.y_coadd
        self.x_l1b, self.y_l1b, self.x_coadd, self.y_coadd = None, None, None, None

class SecondRoundImage():
    def __init__(self, quadrant=-1):
        self.sky = None
        self.dsky = None
        self.zp = None
        self.pa = None # for john fowler
        self.ascending = None # for john fowler
        self.ncopix = None
        self.npatched = None
        self.nrchipix = None
        self.w = None
        self.included = None
        self.omask = None

        self.coslc = None
        self.coimgsq = None
        self.coimg = None
        self.cow = None
        self.con = None
        self.rmask2 = None

        # optional
        self.scan_id = None
        self.frame_num = None
        self.quadrant = quadrant

        # only for plotting ??
        self.rchi = None
        self.badpix = None
        self.rimg_orig = None
        self.rmask_orig = None

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

def get_atlas_tiles(r0,r1,d0,d1, W=2048, H=2048, pixscale=2.75, coadd_id=None):
    '''
    Select Atlas Image tiles touching a desired RA,Dec box.

    pixscale in arcsec/pixel
    '''
    # Read Atlas Image table
    fn = os.path.join(os.environ.get('UNWISE_META_DIR'), 'wise_allsky_4band_p3as_cdd.fits')
    print 'Reading', fn
    T = fits_table(fn, columns=['coadd_id', 'ra', 'dec'])
    T.row = np.arange(len(T))
    print 'Read', len(T), 'Atlas tiles'

    margin = (max(W,H) / 2.) * (pixscale / 3600.)

    T.cut(in_radec_box(T.ra, T.dec, r0,r1,d0,d1, margin))
    print 'Cut to', len(T), 'Atlas tiles near RA,Dec box'

    T.coadd_id = np.array([c.replace('_ab41','') for c in T.coadd_id])

    if coadd_id is not None:
        T = T[T.coadd_id == coadd_id] # hack
        return T # hack

    # Some of them don't *actually* touch our RA,Dec box...
    print 'Checking tile RA,Dec bounds...'
    keep = []
    for i in range(len(T)):
        wcs = get_coadd_tile_wcs(T.ra[i], T.dec[i], W, H, pixscale)
        R0,R1,D0,D1 = get_wcs_radec_bounds(wcs)
        # FIXME RA wrap
        if R1 < r0 or R0 > r1 or D1 < d0 or D0 > d1:
            print 'Coadd tile', T.coadd_id[i], 'is outside RA,Dec box'
            continue
        keep.append(i)
    T.cut(np.array(keep))
    print 'Cut to', len(T), 'tiles'
    # sort
    T.cut(np.argsort(T.coadd_id))
    return T

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

def get_wise_frames(racen, deccen, band, margin=1.7):
    '''
    Returns WISE frames touching the given RA,Dec box plus margin.
    '''
    # Read WISE frame metadata
    metadir = os.environ.get('UNWISE_META_DIR')
    if metadir is None:
        metadir = wisedir
    index_fname = os.path.join(metadir, 'WISE-index-L1b_w'+str(band)+'.fits')
    WISE = fits_table(index_fname)
    print 'Read', len(WISE), 'WISE L1b frames from ' + index_fname
    WISE.row = np.arange(len(WISE))

    # Coarse cut on RA,Dec box.
    t0 = _time()
    WISE.cut(is_nearby(WISE.ra, WISE.dec, racen, deccen, margin, fast=True))
    dt = _time()-t0
    print 'figuring out which frames are nearby took ' + str(dt) + ' seconds'
    print 'Cut to', len(WISE), 'WISE frames near RA,Dec box'

    # Join to WISE Single-Frame Metadata Tables
    WISE.planets = np.zeros(len(WISE), np.int16) - 1
    WISE.nearby_planets = np.zeros(len(WISE), np.int16) - 1
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

    for nbands,name in [(4,'4band'), (3,'3band'), (2,'2band'), (2,'neowiser'),
                        (2, 'neowiser2'), (2, 'neowiser3'), (2, 'neowiser4'),
                        (2, 'neowiser5'), (2, 'neowiser6'), (2, 'neowiser7'),
                        (2, 'neowiser8'), (2, 'neowiser9'), (2, 'neowiser10'),
                        (2, 'neowiser11')]:
        fn = os.path.join(metadir, 'WISE-l1b-metadata-%s.fits' % name)
        print 'Reading', fn
        bb = [1,2,3,4][:nbands]

        if not any([b in bands for b in bb]):
            # no bands of interest in this observation phase - skip
            continue

        fn = os.path.join(wisedir, 'WISE-l1b-metadata-%s.fits' % name)
        if not os.path.exists(fn):
            print('WARNING: ignoring missing', fn)
            continue
        print('Reading', fn)
        cols = (['ra', 'dec', 'scan_id', 'frame_num',
                 'qual_frame', 'planets', 'nearby_planets', 'moon_masked', ] +
                ['w%iintmed16ptile' % b for b in bb] +
                ['w%iintmedian' % b for b in bb] +
                ['w%iintstddev' % b for b in bb])
        if nbands > 2:
            cols.append('dtanneal')
        T = fits_table(fn, columns=cols)
        debug('Read', len(T), 'from', fn)
        # Cut with extra large margins
        T.cut(is_nearby(T.ra, T.dec, racen, deccen, 2.*margin, fast=True))
        print 'Cut to', len(T), 'near RA,Dec box'
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
            WISE.planets[II] = T.planets[JJ]
            WISE.nearby_planets[II] = T.nearby_planets[JJ]

    debug(np.sum(WISE.matched), 'of', len(WISE), 'matched to metadata tables')
    assert(np.sum(WISE.matched) == len(WISE))
    WISE.delete_column('matched')
    # Reorder by scan, frame, band
    WISE.cut(np.lexsort((WISE.band, WISE.frame_num, WISE.scan_id)))
    return WISE

def check_one_md5(wise):
    intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, wise.band, int_gz=int_gz)
    uncfn = intfn.replace('-int-', '-unc-')
    if unc_gz and (not int_gz):
        uncfn = uncfn + '.gz'
    maskfn = intfn.replace('-int-', '-msk-')
    if mask_gz and (not int_gz):
        maskfn = maskfn + '.gz'
    instr = ''
    ok = True
    for fn in [intfn,uncfn,maskfn]:
        if not os.path.exists(fn):
            print >>sys.stderr, '%s: DOES NOT EXIST' % fn
            ok = False
            continue
        mdfn = fn + '.md5'
        if not os.path.exists(mdfn):
            print >>sys.stderr, '%s: DOES NOT EXIST' % mdfn
            ok = False
            continue
        md5 = read_file(mdfn)
        instr += '%s  %s\n' % (md5, fn)
    if len(instr):
        cmd = "echo '%s' | md5sum -c" % instr
        rtn,out,err = run_command(cmd)
        print out, err
        if rtn:
            print >>sys.stderr, 'ERROR: return code', rtn
            print >>sys.stderr, out
            print >>sys.stderr, err
            ok = False
    return ok

def check_md5s(WISE):
    from astrometry.util.run_command import run_command
    from astrometry.util.file import read_file
    ibad = []
    for i,wise in enumerate(WISE):
        print 'Checking md5', i+1, 'of', len(WISE)
        if not check_one_md5(wise):
            ibad.append(i)
    return np.array(ibad)

def cut_to_epoch(WISE, epoch, before, after, ti=None):
    if epoch is not None:
        if ti is not None:
            lambd, beta = radectoecliptic(ti.ra, ti.dec)
            subdivide = (np.abs(beta) > 80)
        else:
            subdivide = False

        ebreaks = get_epoch_breaks(WISE[WISE.qual_frame > 0].mjd, 
                                   subdivide=subdivide)
        assert(epoch <= len(ebreaks))
        if epoch > 0:
            WISE = WISE[WISE.mjd >= ebreaks[epoch - 1]]
        if epoch < len(ebreaks):
            WISE = WISE[WISE.mjd <  ebreaks[epoch]]
        print 'Cut to', len(WISE), 'within epoch'

    if before is not None:
        WISE.cut(WISE.mjd < before)
        print 'Cut to', len(WISE), 'frames before MJD', before
    if after is not None:
        WISE.cut(WISE.mjd > after)
        print 'Cut to', len(WISE), 'frames after MJD', after

    # should probably add a check here on whether this is all sane
    # i.e. i could imagine values of before and after that are
    # totally inconsistent with ebreaks, leading to WISE
    # having zero rows after cuts

    return WISE

def split_one_l1b_round1(rimg, wise, delete_xy_coords=False, reference=None, do_apply_warp=False, save_raw=False,
                         only_good_chi2=True, debug=False, do_rebin=False):
    # split one round1 image into its constituent quadrants
    # rimg is a round1 image object **that must hold xy coordinates**
    # wise is corresponding row of WISE metadata table

    if rimg is None:
        return None

    quadrant_list = []
    for quad_num in range(1, 5):
        rimg_quad = split_one_quadrant(rimg, wise, quad_num, delete_xy_coords=delete_xy_coords, 
                                       reference=reference, do_apply_warp=do_apply_warp, save_raw=save_raw, 
                                       only_good_chi2=only_good_chi2, debug=debug, do_rebin=do_rebin)
        if rimg_quad is not None:
            quadrant_list.append(rimg_quad)

    # delete the x,y coordinates stored in rimg !!!!
    rimg.clear_xy_coords()

    if len(quadrant_list) == 0:
        return None
    else:
        return quadrant_list

def split_one_quadrant(rimg, wise, quad_num, redo_sky=False, reference=None, delete_xy_coords=False,
                       do_apply_warp=False, save_raw=False, only_good_chi2=True, debug=False, do_rebin=False):
    # helper function for split_one_image_quadrants, to deal with just one of the four
    # quadrants

    # need to implement: if reference is not None, then fit a warp
    # reference will be an object holding (at least) reference image, its uncertainty, and its
    # integer coverage

    # these are (supposed to be) zero indexed coordinates
    x_l1b_absolute = rimg.x_l1b + wise.imextent[0]
    y_l1b_absolute = rimg.y_l1b + wise.imextent[2]

    par = WarpMetaParameters()

    xmin = par.get_xmin_quadrant(quad_num)
    xmax = par.get_xmax_quadrant(quad_num)
    ymin = par.get_ymin_quadrant(quad_num)
    ymax = par.get_ymax_quadrant(quad_num)
    quad_mask = (x_l1b_absolute >= xmin) & (x_l1b_absolute <= xmax) & (y_l1b_absolute >= ymin) & (y_l1b_absolute <= ymax)

    # return None if there's no coverage of quadrant quad_num
    if np.sum(quad_mask) == 0:
        print 'Skipping quadrant ' + str(quad_num)
        return None

    # make sure to delete x_l1b, y_l1b, x_coadd, y_coadd fields once they're no longer needed
    # fields that need to be filled in
    #    coextent   -- needs to be updated  XXX
    #    cosubwcs   -- needs to be updated  XXX
    #    ncopix     -- needs to be updated  XXX
    #    npatched   -- needs to be updated  XXX
    #    rimg       -- needs to be trimmed  XXX
    #    rmask      -- needs to be trimmed  XXX
    #    sky        -- change if redo_sky
    #    w          -- NO CHANGE            XXX
    #    wcs        -- needs to be updated  XXX
    #    zp         -- NO CHANGE            XXX
    #    zpscale    -- NO CHANGE            XXX
    #    wcs_full   -- NO CHANGE            XXX
    #    cowcs_full -- NO CHANGE            XXX
    #    x_l1b      -- needs to be updated  XXX
    #    y_l1b      -- needs to be updated  XXX
    #    x_coadd    -- needs to be updated  XXX
    #    y_coadd    -- needs to be updated  XXX
    #    quadrant   -- needs to be set      XXX

    rimg_quad = deepcopy(rimg)
    rimg_quad.quadrant = quad_num

    if quad_num == 1:
        coextent_q = wise.coextent_q1
        imextent_q = wise.imextent_q1
    elif quad_num == 2:
        coextent_q = wise.coextent_q2
        imextent_q = wise.imextent_q2
    elif quad_num == 3:
        coextent_q = wise.coextent_q3
        imextent_q = wise.imextent_q3
    elif quad_num == 4:
        coextent_q = wise.coextent_q4
        imextent_q = wise.imextent_q4

    x0_l1b, x1_l1b, y0_l1b, y1_l1b = imextent_q
    rimg_quad.coextent = coextent_q
    rimg_quad.wcs = rimg.wcs_full.get_subimage(int(x0_l1b), int(y0_l1b), int(1+x1_l1b-x0_l1b), int(1+y1_l1b-y0_l1b))

    cox0,cox1,coy0,coy1 = coextent_q
    coW = int(1 + cox1 - cox0)
    coH = int(1 + coy1 - coy0)

    rimg_quad.cosubwcs = rimg.cowcs_full.get_subimage(int(cox0), int(coy0), coW, coH)

    quad_mask_image = np.zeros(rimg.rimg.shape)
    #print quad_mask_image.shape, quad_num, np.min(rimg.x_coadd), np.max(rimg.x_coadd)
    quad_mask_image[(rimg.y_coadd)[quad_mask], (rimg.x_coadd)[quad_mask]] = 1
    
    quad_rimg = (rimg.rimg)*quad_mask_image
    quad_rmask = ((rimg.rmask)*quad_mask_image).astype('int')

    # need to extract relevant cutouts that become rimg_quad.rimg and rimg_quad.rmask
    assert((coextent_q[0] >= rimg.coextent[0]) and (coextent_q[1] <= rimg.coextent[1]))
    assert((coextent_q[2] >= rimg.coextent[2]) and (coextent_q[3] <= rimg.coextent[3]))

    # watch out for fence-posting
    y_bot = coextent_q[2] - rimg.coextent[2]
    y_top = y_bot + coextent_q[3] - coextent_q[2] + 1 # fence-posting
    x_left = coextent_q[0] - rimg.coextent[0]
    x_right = x_left + coextent_q[1] - coextent_q[0] + 1 # fence-posting
    quad_rimg = quad_rimg[y_bot:y_top, x_left:x_right]
    quad_rmask = quad_rmask[y_bot:y_top, x_left:x_right]

    rimg_quad.rimg = quad_rimg
    rimg_quad.rmask = quad_rmask

    rimg_quad.ncopix = np.sum(rimg_quad.rmask != 0)

    # note that this isn't consistent with definition of 
    # npatched from _coadd_one_round1 in unwise_coadd.py
    rimg_quad.npatched = np.sum(rimg_quad.rmask == 1)

    rimg_quad.x_l1b = x_l1b_absolute[quad_mask] # for full exposure .x_l1b, .y_l1b were NOT absolute
    rimg_quad.y_l1b = y_l1b_absolute[quad_mask] # for full exposure .x_l1b, .y_l1b were NOT absolute
    rimg_quad.x_coadd = rimg.x_coadd[quad_mask] - x_left
    rimg_quad.y_coadd = rimg.y_coadd[quad_mask] - y_bot

    rimg_quad.scan_id = wise.scan_id
    rimg_quad.frame_num = wise.frame_num

    assert(len(rimg_quad.x_l1b) == np.sum(rimg_quad.rmask != 0))

    # if reference is not None, call the warping code here !!
    if reference is not None:
        warp = do_one_warp(rimg_quad, wise, reference, debug=debug, do_rebin=do_rebin)

        if warp is not None:
             rimg_quad.warp = warp
#            if do_apply_warp kw set, modify rimg_quad.rimg by subtracting the warp image
             if do_apply_warp:
                 rimg_quad = apply_warp(rimg_quad, wise.band, save_raw=save_raw, only_good_chi2=only_good_chi2)
    # clear some space in memory if x,y coords no longer needed
    if delete_xy_coords:
        rimg_quad.clear_xy_coords()

    return rimg_quad

def process_round1_quadrants(WISE, cowcs, zp_lookup_obj, r1_coadd=None, delete_xy_coords=False, reference=None,
                             do_apply_warp=False, save_raw=False, coadd=None, only_good_chi2=True, debug=False,
                             do_rebin=False):
    # WISE is a table with all the relevant L1b metadata
    # particularly imextent, coextent, imextent_q?, coextent_q?
    # should return a list of FirstRoundImage objects one per **quadrant**

    # WISE assumed to be already trimmed down to rows for which
    # per-quadrant FirstRoundImage objects are desired

    if coadd is None:
        quad_rimgs = []
    # for each exposure in the input WISE table
    N = len(WISE)
    table = True # think this is always the case everywhere else ...
    L = 3 # think this is always the case everywhere else ...
    ps = None # hack
    band = WISE[0].band # hack
    medfilt = False # hack
    do_check_md5 = False # hack

    warp_list = [] # list of QuadrantWarp objects for successfully warped quadrants
    r2_masks = [] # list of per-quadrant SecondRoundImage objects for successfully warped quadrants
    n_succeeded = 0
    n_attempted = 0
    n_skipped = 0
    print "Creating per-quadrant FirstRoundImage objects"
    for wi, wise in enumerate(WISE):
        # do the usual call to _coadd_one_round1 to get a typical FirstRoundImage
        rr = _coadd_one_round1((wi, N, wise, table, L, ps, band, cowcs, medfilt,
                                do_check_md5, zp_lookup_obj), store_xy_coords=True)
        # do *not* want to make an intermediate list of the rr objects, since these
        # are holding x_l1b, y_l1b, x_coadd, y_coadd coordinate lists, so this would
        # require a lot of RAM
        quadrants_this_exp = split_one_l1b_round1(rr, wise, reference=reference, delete_xy_coords=delete_xy_coords,
                                                  do_apply_warp=do_apply_warp, save_raw=save_raw, 
                                                  only_good_chi2=only_good_chi2, debug=debug, do_rebin=do_rebin)
        if coadd is None:
            if quadrants_this_exp is not None:
                quad_rimgs.extend(quadrants_this_exp)
        else:
            if quadrants_this_exp is not None:
                for qq in quadrants_this_exp:
                    if qq.warped:
                        if r1_coadd is not None:
                            scanid = ('scan %s frame %i band %i' % (wise.scan_id, wise.frame_num, band))
                            plotfn = None
                            ps1 = False # maybe this should be None instead ?
                            do_dsky = False
                            delmm = True
                            tinyw = 1e-16
                            rchi_fraction = 0.01 # might need to tune this
                            mm = _coadd_one_round2((wi, N, scanid, qq, r1_coadd.cow1, r1_coadd.cowimg1, r1_coadd.cowimgsq1, tinyw,
                                                    plotfn, ps1, do_dsky, rchi_fraction, r1_coadd.con1))
                            if mm is None:
                                n_skipped += 1
                                continue
                            coadd.acc(mm, delmm=delmm)
                            mm.scan_id = wise.scan_id
                            mm.frame_num = wise.frame_num
                            r2_masks.append(mm)
                        warp_list.append(qq.warp)
                        n_succeeded += 1
                        n_attempted += 1
                        print 'Recovered a quadrant.'
                    elif qq.warp is None:
                        n_skipped += 1
                        print 'No warp attempted.'
                    else:
                        n_attempted += 1
                        print 'Quadrant recovery failed.'
        del quadrants_this_exp
        del rr

    if coadd is None:
        # get n_attempted, n_succeeded, n_skipped right ... should really clean this up at some point
        for rimg_quad in quad_rimgs:
            if rimg_quad.warp is None:
                n_skipped += 1
            else:
                n_attempted += 1
                if rimg_quad.warped:
                    n_succeeded += 1

    gc.collect()
    rstats = RecoveryStats(n_attempted, n_succeeded, n_skipped)
    if coadd is None:
        if len(quad_rimgs) == 0:
            return None, rstats
        else:
            return quad_rimgs, rstats
    else:
        return coadd, warp_list, r2_masks, rstats

def get_extents_quadrant(wcs, cowcs, copoly, W, H, WISE, wi, ps, quad_num, coextent, imextent, margin=10):
    # want to calculate coextent-like and imextent-like values for an L1b 
    # quadrant rather than an entire L1b exposure
    
    # assert that quadrant number is one of 1-4 inclusive
    assert((quad_num == 1) or (quad_num == 2) or (quad_num == 3) or (quad_num == 4))

    par = WarpMetaParameters()
    xmin = par.get_xmin_quadrant(quad_num, one_indexed=True) - margin
    xmax = par.get_xmax_quadrant(quad_num, one_indexed=True) + margin
    ymin = par.get_ymin_quadrant(quad_num, one_indexed=True) - margin
    ymax = par.get_ymax_quadrant(quad_num, one_indexed=True) + margin

    xcorners = np.array([xmin, xmax, xmax, xmin])
    ycorners = np.array([ymin, ymin, ymax, ymax])
    r,d = wcs.pixelxy2radec(xcorners, ycorners)

    ok,u,v = cowcs.radec2iwc(r, d)
    poly = np.array(list(reversed(zip(u,v))))
    intersects = polygons_intersect(copoly, poly)

    coextent_q = np.zeros(4, dtype=int)
    imextent_q = np.zeros(4, dtype=int)

    if not intersects:
        print 'Quadrant ' + str(quad_num) + ' does not intersect target'
        return coextent_q, imextent_q

    cpoly = np.array(clip_polygon(copoly, poly))
    if len(cpoly) == 0:
        print 'No overlap between coadd and quadrant ' + str(quad_num) + ' polygons'
        return coextent_q, imextent_q

    # Convert the intersected polygon in IWC space into image
    # pixel bounds.
    # Coadd extent:
    xy = np.array([cowcs.iwc2pixelxy(u,v) for u,v in cpoly])
    xy -= 1
    x0,y0 = np.floor(xy.min(axis=0)).astype(int)
    x1,y1 = np.ceil (xy.max(axis=0)).astype(int)
    coextent_q = [np.clip(x0, coextent[0], coextent[1]),
                  np.clip(x1, coextent[0], coextent[1]),
                  np.clip(y0, coextent[2], coextent[3]),
                  np.clip(y1, coextent[2], coextent[3])]

    # Input image extent:
    rd = np.array([cowcs.iwc2radec(u,v) for u,v in cpoly])
    ok,x,y = np.array(wcs.radec2pixelxy(rd[:,0], rd[:,1]))
    x -= 1 # now things are 0 indexed ...
    y -= 1 # now things are 0 indexed ...
    x0,y0 = [np.floor(v.min(axis=0)).astype(int) for v in [x,y]]
    x1,y1 = [np.ceil (v.max(axis=0)).astype(int) for v in [x,y]]
    imextent_q = [np.clip(x0, max(imextent[0], xmin-1+margin), min(imextent[1], xmax-1-margin)),
                  np.clip(x1, max(imextent[0], xmin-1+margin), min(imextent[1], xmax-1-margin)),
                  np.clip(y0, max(imextent[2], ymin-1+margin), min(imextent[3], ymax-1-margin)),
                  np.clip(y1, max(imextent[2], ymin-1+margin), min(imextent[3], ymax-1-margin))]

    return coextent_q, imextent_q

def get_extents(wcs, cowcs, copoly, W, H, WISE, wi, ps):
    h,w = wcs.get_height(), wcs.get_width()
    r,d = walk_wcs_boundary(wcs, step=2.*w, margin=10)
    ok,u,v = cowcs.radec2iwc(r, d)
    poly = np.array(list(reversed(zip(u,v))))
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
            print 'cpoly:', cpoly
            plt.plot(cpoly[:,0], cpoly[:,1], 'm-')
            plt.plot(cpoly[0,0], cpoly[0,1], 'mo')
        ps.savefig()

    if not intersects:
        print 'Image does not intersect target'
        WISE.use[wi] = False
        return WISE, False

    cpoly = np.array(clip_polygon(copoly, poly))
    if len(cpoly) == 0:
        print 'No overlap between coadd and image polygons'
        print 'copoly:', copoly
        print 'poly:', poly
        print 'cpoly:', cpoly
        WISE.use[wi] = False
        return WISE, False

    # Convert the intersected polygon in IWC space into image
    # pixel bounds.
    # Coadd extent:
    xy = np.array([cowcs.iwc2pixelxy(u,v) for u,v in cpoly])
    xy -= 1
    x0,y0 = np.floor(xy.min(axis=0)).astype(int)
    x1,y1 = np.ceil (xy.max(axis=0)).astype(int)
    WISE.coextent[wi,:] = [np.clip(x0, 0, W-1),
                           np.clip(x1, 0, W-1),
                           np.clip(y0, 0, H-1),
                           np.clip(y1, 0, H-1)]

    # Input image extent:
    #   There was a bug in the an-ran coadds; all imextents are
    #   [0,1015,0,1015] as a result.
    #rd = np.array([cowcs.iwc2radec(u,v) for u,v in poly])
    # Should be: ('cpoly' rather than 'poly' here)
    rd = np.array([cowcs.iwc2radec(u,v) for u,v in cpoly])
    ok,x,y = np.array(wcs.radec2pixelxy(rd[:,0], rd[:,1]))
    x -= 1
    y -= 1
    x0,y0 = [np.floor(v.min(axis=0)).astype(int) for v in [x,y]]
    x1,y1 = [np.ceil (v.max(axis=0)).astype(int) for v in [x,y]]
    WISE.imextent[wi,:] = [np.clip(x0, 0, w-1),
                           np.clip(x1, 0, w-1),
                           np.clip(y0, 0, h-1),
                           np.clip(y1, 0, h-1)]

    WISE.imagew[wi] = w
    WISE.imageh[wi] = h
    WISE.wcs[wi] = wcs # not clear that this belongs in this subroutine
    print 'Image extent:', WISE.imextent[wi,:]
    print 'Coadd extent:', WISE.coextent[wi,:]

    ### now deal with the quadrants
    WISE.coextent_q1[wi,:], WISE.imextent_q1[wi,:] = get_extents_quadrant(wcs, cowcs, copoly, W, H, WISE, 
                                                                          wi, ps, 1, WISE.coextent[wi,:],
                                                                          WISE.imextent[wi,:], margin=10)
    WISE.coextent_q2[wi,:], WISE.imextent_q2[wi,:] = get_extents_quadrant(wcs, cowcs, copoly, W, H, WISE, 
                                                                          wi, ps, 2, WISE.coextent[wi,:],
                                                                          WISE.imextent[wi,:], margin=10)
    WISE.coextent_q3[wi,:], WISE.imextent_q3[wi,:] = get_extents_quadrant(wcs, cowcs, copoly, W, H, WISE, 
                                                                          wi, ps, 3, WISE.coextent[wi,:],
                                                                          WISE.imextent[wi,:], margin=10)
    WISE.coextent_q4[wi,:], WISE.imextent_q4[wi,:] = get_extents_quadrant(wcs, cowcs, copoly, W, H, WISE, 
                                                                          wi, ps, 4, WISE.coextent[wi,:],
                                                                          WISE.imextent[wi,:], margin=10)
    ###
    return WISE, True


def one_coadd(ti, band, W, H, pixscale, WISE,
              ps, wishlist, outdir, mp1, mp2, do_cube, plots2,
              frame0, nframes, force, medfilt, maxmem, do_dsky, checkmd5,
              bgmatch, center, minmax, rchi_fraction, do_cube1, epoch,
              before, after, recover_warped, do_rebin, try_download,
              force_outdir=False, just_image=False, warp_all=False,
              reference_dir=None, hi_lo_rej=False, output_masks=True):
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

    version = retrieve_git_version()

    if recover_warped:
        print 'will attempt to recover frames using per-quadrant polynomial warps ...'

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
    if not compare_moon_all:
        WISE = cut_to_epoch(WISE, epoch, before, after, ti=ti)

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
    WISE.use = np.ones(len(WISE), bool)
    WISE.moon_rej = np.zeros(len(WISE), bool)
    WISE.use *= (WISE.qual_frame > 0)
    print 'Cut out qual_frame = 0;', sum(WISE.use), 'remaining'

    WISE.use *= good_scan_mask(WISE)
    print 'Cut out bad scans from bad_scans.txt; ', sum(WISE.use), 'remaining'

    WISE.use *= (WISE.planets == 0)
    print 'Cut out planets != 0;', sum(WISE.use), 'remaining'
    if not recover_warped:
        WISE.use *= (WISE.nearby_planets == 0)
        print 'Cut out nearby planets != 0;', sum(WISE.use), 'remaining'

    if band in [3,4]:
        frames.use *= (frames.dtanneal > 2000.)
        debug('Cut out dtanneal <= 2000 seconds:', sum(frames.use), 'remaining')

    if band == 4:
        ok = np.array([np.logical_or(s < '03752a', s > '03761b')
                       for s in frames.scan_id])
        frames.use *= ok
        debug('Cut out bad scans in W4:', sum(frames.use), 'remaining')

    # Cut ones where the w?intmedian is NaN
    frames.use *= np.isfinite(frames.intmedian)
    debug('Cut out intmedian non-finite:', sum(frames.use), 'remaining')

    # this will need to be adapted/modified for the time-resolved coadds...
    # Cut on moon, based on (robust) measure of standard deviation
    if sum(WISE.moon_masked[WISE.use]):
        moon = WISE.moon_masked[WISE.use]
        nomoon = np.logical_not(moon)
        Imoon = np.flatnonzero(WISE.use)[moon]
        assert(sum(moon) == len(Imoon))
        print sum(nomoon), 'of', sum(WISE.use), 'frames are not moon_masked'
        nomoonstdevs = WISE.intmed16p[WISE.use][nomoon]
        med = np.median(nomoonstdevs)
        mad = 1.4826 * np.median(np.abs(nomoonstdevs - med))
        print 'Median', med, 'MAD', mad
        moonstdevs = WISE.intmed16p[WISE.use][moon]
        okmoon = (moonstdevs - med)/mad < 5.
        print sum(np.logical_not(okmoon)), 'of', len(okmoon), 'moon-masked frames have large pixel variance'
        if not recover_warped:
            WISE.use[Imoon] *= okmoon
        WISE.moon_rej[Imoon] = (~okmoon)
        print 'Cut to', sum(WISE.use), 'on moon'
        del Imoon
        del moon
        del nomoon
        del nomoonstdevs
        del med
        del mad
        del moonstdevs
        del okmoon

    if compare_moon_all:
        WISE = cut_to_epoch(WISE, epoch, before, after)

    print 'Frames:'
    for i,w in enumerate(WISE):
        print '  ', i, w.scan_id, '%4i' % w.frame_num, 'MJD', w.mjd
   
    if frame0 or nframes:
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
        for wise in WISE:
            intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, band, int_gz=int_gz)
            if not os.path.exists(intfn):
                print 'Need:', intfn
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
    WISE.coextent = np.zeros((len(WISE), 4), int)
    WISE.coextent_q1 = np.zeros((len(WISE), 4), int)
    WISE.coextent_q2 = np.zeros((len(WISE), 4), int)
    WISE.coextent_q3 = np.zeros((len(WISE), 4), int)
    WISE.coextent_q4 = np.zeros((len(WISE), 4), int)

    # *inclusive* coordinates of the bounding-box in the image
    # overlapping coadd
    WISE.imextent = np.zeros((len(WISE), 4), int)
    WISE.imextent_q1 = np.zeros((len(WISE), 4), int)
    WISE.imextent_q2 = np.zeros((len(WISE), 4), int)
    WISE.imextent_q3 = np.zeros((len(WISE), 4), int)
    WISE.imextent_q4 = np.zeros((len(WISE), 4), int)

    frames.imagew = np.zeros(len(frames), np.int32)
    frames.imageh = np.zeros(len(frames), np.int32)
    frames.intfn  = np.zeros(len(frames), object)
    frames.wcs    = np.zeros(len(frames), object)

    # count total number of coadd-space pixels -- this determines memory use
    pixinrange = 0.

    wdirs = get_l1b_dirs(yml=True, verbose=True)
    nu = 0
    NU = sum(frames.use)
    failedfiles = []
    for wi,wise in enumerate(frames):
        if not wise.use:
            continue
        nu += 1
        debug(nu, 'of', NU, 'scan', wise.scan_id, 'frame', wise.frame_num, 'band', band)

        found = False
        _phase = phase_from_scanid(wise.scan_id)
        for wdir in [wdirs[_phase], wdirs['missing']] + [None]:
            download = False
            if wdir is None:
                download = allow_download
                wdir = 'merge_p1bm_frm'

            intfn = get_l1b_file(wdir, wise.scan_id, wise.frame_num, band, int_gz=int_gz)

            if download and try_download:
                download_frameset_1band(wise.scan_id, wise.frame_num, band)
            if os.path.exists(intfn):
                print 'intfn', intfn
                try:
                    if not int_gz:
                        wcs = Sip(intfn)
                    else:
                        tmpname = (intfn.split('/'))[-1]
                        tmpname = tmpname.replace('.gz', '')
                        # add random stuff to tmpname to avoid collisions b/w simultaneous jobs
                        tmpname = str(random.randint(0, 1000000)).zfill(7) + '-' + tmpname
                        cmd_unzip_tmp = 'gunzip -c '+ intfn + ' > ' + tmpname
                        os.system(cmd_unzip_tmp)
                        wcs = Sip(tmpname)
                        # delete unzipped temp file
                        cmd_delete_tmp = 'rm ' +  tmpname
                        os.system(cmd_delete_tmp)
                except RuntimeError:
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                debug('does not exist:', intfn)
                continue
            if (os.path.exists(intfn.replace('-int-', '-unc-') + ('.gz' if not int_gz else '')) and
                os.path.exists(intfn.replace('-int-', '-msk-') + ('.gz' if not int_gz else ''))):
                found = True
                break
            else:
                print('missing unc or msk file')
                continue
        if not found:
            WISE.use[wi] = False
            print 'WARNING: Not found: scan', wise.scan_id, 'frame', wise.frame_num, 'band', band
            continue

        WISE, has_overlap = get_extents(wcs, cowcs, copoly, W, H, WISE, wi, ps)
        # Count total coadd-space bounding-box size -- this x 5 bytes
        # is the memory toll of our round-1 coadds, which is basically
        # the peak memory use.
        if has_overlap:
            WISE.intfn[wi] = intfn
            e = WISE.coextent[wi,:]
            pixinrange += (1+e[1]-e[0]) * (1+e[3]-e[2])
            print 'Total pixels in coadd space:', pixinrange

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

    debug('Frames to coadd after cuts:')
    ii = np.argsort(frames.mjd)
    for i in ii:
        w = frames[i]
        if not w.use:
            continue
        debug('  ', w.scan_id, '%4i' % w.frame_num, 'MJD', w.mjd,
              'ASC', w.ascending, 'DESC', w.descending, 'RA,Dec %.4f, %.4f' % (w.ra, w.dec))

    # construct metadata table of frames we'll try to recover
    recover = None
    if recover_warped:
        if np.sum(WISE.use & (WISE.moon_rej | (WISE.nearby_planets != 0))) != 0:
            recover = WISE[WISE.use & (WISE.moon_rej | (WISE.nearby_planets != 0))]

    # Now that we've got some information about the input frames, call
    # the real coadding code.  Maybe we should move this first loop into
    # the round 1 coadd...
    try:
        (coim,coiv,copp,con, coimb,coivb,coppb,conb,masks, cube, cosky,
         comin,comax,cominb,comaxb, warp_list, qmasks, rstats
         )= coadd_wise(ti.coadd_id, cowcs, WISE[WISE.use & ~(WISE.moon_rej | (WISE.nearby_planets != 0))], ps, 
                       band, mp1, mp2, do_cube, medfilt, plots2=plots2, do_dsky=do_dsky,
                       checkmd5=checkmd5, bgmatch=bgmatch, minmax=minmax,
                       rchi_fraction=rchi_fraction, do_cube1=do_cube1, recover=recover, do_rebin=do_rebin,
                       warp_all=warp_all, reference_dir=reference_dir, hi_lo_rej=hi_lo_rej)
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
    if np.sum(con == 0):
        coim[con == 0] = 0
        coimb[con == 0] = 0

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
    hdr.add_record(dict(name='UNW_DVER', value=3.3,
                        comment='unWISE data model version'))
    hdr.add_record(dict(name='UNW_DATE', value=datetime.datetime.now().isoformat(),
                        comment='unWISE run time'))
    hdr.add_record(dict(name='UNW_FR0', value=frame0, comment='unWISE frame start'))
    hdr.add_record(dict(name='UNW_FRN', value=nframes, comment='unWISE N frames'))
    hdr.add_record(dict(name='UNW_FRNR', value=nframes_random, comment='unWISE N random frames'))
    hdr.add_record(dict(name='UNW_MEDF', value=medfilt, comment='unWISE median filter sz'))
    hdr.add_record(dict(name='UNW_BGMA', value=bgmatch, comment='unWISE background matching?'))

    referen1, referen2 = header_reference_keywords(reference_dir)
    hdr.add_record(dict(name='REFEREN1', value=referen1, comment='reference coadd directory'))
    hdr.add_record(dict(name='REFEREN2', value=referen2, comment='reference coadd directory, continued'))

    # make sure there's always a numerical representation of epoch that can go into header
    if epoch is None:
        epoch_num = -1
    else:
        epoch_num = epoch

    hdr.add_record(dict(name='EPOCH', value=epoch_num, comment='epoch number'))

    WISE.included = np.zeros(len(WISE), np.uint8)
    WISE.sky1 = np.zeros(len(WISE), np.float32)
    WISE.sky2 = np.zeros(len(WISE), np.float32)
    WISE.zeropoint = np.zeros(len(WISE), np.float32)
    WISE.pa = np.zeros(len(WISE), np.float32)
    WISE.ascending   = np.zeros(len(WISE), np.uint8)
    WISE.npixoverlap = np.zeros(len(WISE), np.int32)
    WISE.npixpatched = np.zeros(len(WISE), np.int32)
    WISE.npixrchi    = np.zeros(len(WISE), np.int32)
    WISE.weight      = np.zeros(len(WISE), np.float32)

    if not warp_all:
        Iused = np.flatnonzero(WISE.use & ~(WISE.moon_rej | (WISE.nearby_planets != 0))) # hack !!!!!
        assert(len(Iused) == len(masks))
        parse_write_masks(outdir, tag, WISE, Iused, masks, int_gz, ofn, ti, output_masks=output_masks)
    else:
        parse_write_quadrant_masks(outdir, tag, WISE, masks, int_gz, ofn, ti, output_masks=output_masks)

    if recover_warped:
        parse_write_quadrant_masks(outdir, tag, WISE, qmasks, int_gz, ofn, ti, output_masks=output_masks)

    WISE.delete_column('wcs')

    # downcast datatypes, and work around fitsio's issues with
    # "bool" columns
    for c,t in [('included', np.uint8),
                ('use', np.uint8),
                ('moon_masked', np.uint8),
                ('moon_rej', np.uint8),
                ('imagew', np.int16),
                ('imageh', np.int16),
                ('coextent', np.int16),
                ('imextent', np.int16),
                ]:
        WISE.set(c, WISE.get(c).astype(t))

    if (warp_list is not None) and (not warp_all):
        update_included_bitmask(WISE, warp_list) # this is the --recover_warped case

    if warp_all:
        update_included_bitmask(WISE, masks)

    # might crash if WISE.use is all zeros ...
    kw_mjdmin = np.min((WISE[WISE.included > 0]).mjd)
    kw_mjdmax = np.max((WISE[WISE.included > 0]).mjd)

    hdr.add_record(dict(name='MJDMIN', value=kw_mjdmin, comment='minimum MJD among included L1b frames'))
    hdr.add_record(dict(name='MJDMAX', value=kw_mjdmax, comment='maximum MJD among included L1b frames'))

    hdr.add_record(dict(name='BAND', value=band, comment='WISE band'))

    # "Unmasked" versions
    ofn = prefix + '-img-u.fits'
    fitsio.write(ofn, coim.astype(np.float32), header=hdr, clobber=True, extname='coadded image, outliers patched')
    print 'Wrote', ofn

    if just_image:
        return 0

    ofn = prefix + '-invvar-u.fits'
    fitsio.write(ofn, coiv.astype(np.float32), header=hdr, clobber=True, extname='inverse variance, outliers patched')
    print 'Wrote', ofn
    ofn = prefix + '-std-u.fits'
    fitsio.write(ofn, copp.astype(np.float32), header=hdr, clobber=True, extname='sample standard deviation, outliers patched')
    print 'Wrote', ofn
    ofn = prefix + '-n-u.fits'
    n_u_type = np.int32 if (np.max(con) >= 2**15) else np.int16
    fitsio.write(ofn, con.astype(n_u_type), header=hdr, clobber=True, extname='integer frame coverage, outlier pixels patched')
    print 'Wrote', ofn

    # "Masked" versions
    ofn = prefix + '-img-m.fits'
    fitsio.write(ofn, (coimb*(conb != 0)).astype(np.float32), header=hdr, clobber=True, extname='coadded image, outliers removed')
    print 'Wrote', ofn
    ofn = prefix + '-invvar-m.fits'
    fitsio.write(ofn, (coivb*(conb != 0)).astype(np.float32), header=hdr, clobber=True, extname='inverse variance, outliers removed')
    print 'Wrote', ofn
    ofn = prefix + '-std-m.fits'
    fitsio.write(ofn, (coppb*(conb != 0)).astype(np.float32), header=hdr, clobber=True, extname='sample standard deviation, outliers removed')
    print 'Wrote', ofn
    ofn = prefix + '-n-m.fits'
    n_m_type = np.int32 if (np.max(conb) >= 2**15) else np.int16
    fitsio.write(ofn, conb.astype(n_m_type), header=hdr, clobber=True, extname='integer frame coverage, outlier pixels removed')
    print 'Wrote', ofn

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

    ofn = prefix + '-frames.fits'

    WISE.writeto(ofn)
    print 'Wrote', ofn
    # append warp summary table, write syntactically correct placeholder if no warps attempted
    warp_tab = gen_warp_table(warp_list)
    if warp_list is not None: print 'Appending warp summary table to ' + ofn
    fitsio.write(ofn, warp_tab)
    if rstats is not None:
        rstats = rstats.to_recarray()
        print 'Appending warp recovery summary statistics to ' + ofn
    else:
        rstats = RecoveryStats(None, None, None) # dummy
        rstats = rstats.to_recarray()
    fitsio.write(ofn, rstats)

    if output_masks:
        md = tag + '-mask'
        cmd = ('cd %s && tar czf %s %s && rm -R %s' %
               (outdir, md + '.tgz', md, md))
        print 'tgz:', cmd
        rtn,out,err = run_command(cmd)
        print out, err
        if rtn:
            print >>sys.stderr, 'ERROR: return code', rtn
            print >>sys.stderr, 'Command:', cmd
            print >>sys.stderr, out
            print >>sys.stderr, err
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

                intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, wise.band, int_gz=int_gz)
                try:
                    # what happens here when int_gz is true ???
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

def parse_write_masks(outdir, tag, WISE, Iused, masks, int_gz, ofn, ti, output_masks=True):
    # use the list of second round masks to update the WISE metadata
    # table and write out per-exposure outlier mask
    maskdir = os.path.join(outdir, tag + '-mask')
    if output_masks:
        if not os.path.exists(maskdir):
            os.mkdir(maskdir)
            
    for i,mm in enumerate(masks):
        if mm is None:
            continue

        ii = Iused[i]
        WISE.sky1       [ii] = mm.sky
        WISE.sky2       [ii] = mm.dsky
        WISE.zeropoint  [ii] = mm.zp
        WISE.pa         [ii] = mm.pa
        WISE.ascending  [ii] = mm.ascending
        WISE.npixoverlap[ii] = mm.ncopix
        WISE.npixpatched[ii] = mm.npatched
        WISE.npixrchi   [ii] = mm.nrchipix
        WISE.weight     [ii] = mm.w

        if not mm.included:
            continue

        WISE.included   [ii] = 1

        # Write outlier masks
        if output_masks:
            ofn = WISE.intfn[ii].replace('-int', '')
            ofn = os.path.join(maskdir, 'unwise-mask-' + ti.coadd_id + '-'
                           + os.path.basename(ofn) + ('.gz' if not int_gz else ''))
            w,h = WISE.imagew[ii],WISE.imageh[ii]
            fullmask = np.zeros((h,w), mm.omask.dtype)
            x0,x1,y0,y1 = WISE.imextent[ii,:]
            fullmask[y0:y1+1, x0:x1+1] = mm.omask
            fitsio.write(ofn, fullmask, clobber=True)
            print 'Wrote mask', (i+1), 'of', len(masks), ':', ofn

def _bounce_one_round2(*A):
    try:
        return _coadd_one_round2(*A)
    except:
        import traceback
        print('_coadd_one_round2 failed:')
        traceback.print_exc()
        raise

def _coadd_one_round2((ri, N, scanid, rr, cow1, cowimg1, cowimgsq1, tinyw,
                       plotfn, ps1, do_dsky, rchi_fraction, con1)):
    '''
    For multiprocessing, the function to be called for each round-2
    frame.
    '''
    (ri, N, scanid, rr, cow1, cowimg1, cowimgsq1, tinyw,
                       plotfn, ps1, do_dsky, rchi_fraction) = X
    if rr is None:
        return None

    included_round1 = rr.included_round1

    print 'Coadd round 2, image', (ri+1), 'of', N
    t00 = Time()
    mm = SecondRoundImage(quadrant=rr.quadrant)
    mm.npatched = rr.npatched
    mm.ncopix   = rr.ncopix
    mm.sky      = rr.sky
    mm.zp       = rr.zp
    mm.pa       = rr.pa
    mm.ascending = rr.ascending
    mm.w        = rr.w
    mm.included = 1

    cox0,cox1,coy0,coy1 = rr.coextent
    coslc = slice(coy0, coy1+1), slice(cox0, cox1+1)
    # Remove this image from the per-pixel std calculation...
    if included_round1:
        subw  = np.maximum(cow1[coslc] - rr.w, tinyw)
        subco = (cowimg1  [coslc] - (rr.w * rr.rimg   )) / subw
        subsq = (cowimgsq1[coslc] - (rr.w * rr.rimg**2)) / subw
    else:
        subw  = np.maximum(cow1[coslc], tinyw)
        subco = (cowimg1[coslc]) / subw
        subsq = (cowimgsq1[coslc]) / subw
    
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
    if not (np.all(np.isfinite(rchi))):
        print 'something very unusual is going on'
        return None

    assert(np.all(np.isfinite(rchi)))

    badpix = ((np.abs(rchi) >= 5.) & (con1[coslc] > 2))
    #print 'Number of rchi-bad pixels:', np.count_nonzero(badpix)

    mm.nrchipix = np.count_nonzero(badpix)

    # Bit 1: abs(rchi) >= 5
    badpixmask = badpix.astype(np.uint8)
    # grow by a small margin
    badpix = binary_dilation(badpixmask)
    # Bit 2: grown
    badpixmask = (badpixmask + (2 * badpix))
    # Add rchi-masked pixels to the mask
    # (clear bit 2)
    rr.rmask[badpix] = (rr.rmask[badpix] & (~2))
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
        print ('WARNING: dropping exposure %s: n rchi pixels %i / %i' %
               (scanid, mm.nrchipix, mm.ncopix))
        mm.included = 0

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
        H = int(H)
        W = int(W)
        self.coimg    = np.zeros((H,W))
        self.coimgsq  = np.zeros((H,W))
        self.cow      = np.zeros((H,W))
        self.con      = np.zeros((H,W), np.int32)
        self.coimgb   = np.zeros((H,W))
        self.coimgsqb = np.zeros((H,W))
        self.cowb     = np.zeros((H,W))
        self.conb     = np.zeros((H,W), np.int32)

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

def do_one_warp(rimg, wise, reference, debug=False, do_rebin=False):
    # return value needs to somehow indicate whether the warp
    # succeeded or failed
    assert(rimg.quadrant != -1)

    # return value should be None if unsuccessful, and a WarpQuadrant
    # object if successful

    # the warp can either fail because of insufficient 
    # number of pixels to fit warp or because chi2 value
    # of best fit warp isn't good enough
    binfac = (2 if do_rebin else 1)

    par = WarpMetaParameters(binfac=binfac) # set band keyword here ??

    npix_good = np.sum(rimg.rmask == 3) # check that 3 is right value
    order = par.npix2order(npix_good)
    if order is None:
        print 'Too few pixels :  ' + str(npix_good) + ', not computing warp'
        return None

    # don't rebin if polynomial correction is just an offset
    if order == 0:
        print 'not rebinning !!'
        binfac = 1
        do_rebin = False
        par = WarpMetaParameters(binfac=binfac)

    _t0 = _time()
    imref, sigref, nref = reference.extract_cutout(rimg)

    # assert that imref has same shape as rimg.rimg
    assert((rimg.rimg.shape[0] ==  imref.shape[0]) and (rimg.rimg.shape[1] == imref.shape[1]))
    # always want to ignore pixels with (rmask != 3) or with zero coverage in terms of -n-u
    # the latter should happen exceptionally rarely or never ...
    non_extreme_mask = mask_extreme_pix(imref, ignore=((rimg.rmask != 3) | (nref == 0)))

    x_l1b_im = np.zeros(rimg.rimg.shape)
    y_l1b_im = np.zeros(rimg.rimg.shape)

    x_l1b_im[rimg.y_coadd, rimg.x_coadd] = rimg.x_l1b
    y_l1b_im[rimg.y_coadd, rimg.x_coadd] = rimg.y_l1b

    if not do_rebin:
        pix_l1b_quad = rimg.rimg[non_extreme_mask]
        pix_ref = imref[non_extreme_mask]
        unc_ref = sigref[non_extreme_mask]

        x_fit = x_l1b_im[non_extreme_mask]
        y_fit = y_l1b_im[non_extreme_mask]
    else:
        images_out, mask_reb = pad_rebin_weighted([rimg.rimg, imref, x_l1b_im, y_l1b_im, sigref], 
                                                  non_extreme_mask.astype('byte'), binfac=binfac)
        goodmask = [mask_reb > 0.5]
        pix_l1b_quad = (images_out[0])[goodmask]
        pix_ref = (images_out[1])[goodmask]
        x_fit = (images_out[2])[goodmask]
        y_fit = (images_out[3])[goodmask]
        unc_ref = (images_out[4])[goodmask]/(binfac*np.sqrt(mask_reb[goodmask]))

    # some of these outputs are never used ...
    t0 = _time()
    coeff, xmed, ymed, x_l1b_quad, y_l1b_quad, isgood, chi2_mean, chi2_mean_raw, pred = compute_warp(pix_l1b_quad, pix_ref, 
                                                                                                     x_fit, 
                                                                                                     y_fit, unc_ref, 
                                                                                                     order=order)
    chi2_mean /= par.get_chi2_fac(binfac)
    chi2_mean_raw /= par.get_chi2_fac(binfac)

    dt = _time() - t0
    print 'time to fit warp = ' + str(dt) + ' seconds, number of pixels = ' + str(npix_good)
    warp = QuadrantWarp(rimg.quadrant, coeff, xmed, ymed, chi2_mean, chi2_mean_raw, 
                        order, non_extreme_mask, npix_good, wise.scan_id, wise.frame_num, 
                        debug=debug)
    _dt = _time() - _t0
    print 'total time in do_one_warp ' + str(_dt) + ' seconds, number of pixels = ' +str(npix_good)
    return warp

def recover_warped_frames(WISE, coadd, reference, cowcs, zp_lookup_obj, r1_coadd, do_rebin=True):
    # coadd is a coaddacc object, which should already have accumulated
    # the Moon-free exposures
    # reference holds relevant info about reference coadd, and is a ReferenceImage object
    # r1_coadd is a FirstRoundCoadd object
    assert(np.min(WISE.use) == 1)
    assert(np.min(WISE.moon_rej | (WISE.nearby_planets != 0)) == 1) # clean this up
    assert(np.sum(WISE.planets) == 0) # never try to recover when planet is inside FOV

    rchi_fraction = 0.05 # be more lenient

    nrec = len(WISE)
    print 'Attempting to recover ' + str(nrec) + ' Moon or planet contaminated frames'
    # call routine to generate per-quadrant list of FirstRoundImages
    # it will be best to compute/apply warp at time of each FirstRoundImage's creation i.e. within process_round1_quadrants

    # pretty sure I really do want to hardwire delete_xy_coords=True here...
    coadd, warp_list, qmasks, rstats = process_round1_quadrants(WISE, cowcs, zp_lookup_obj, r1_coadd=r1_coadd, 
                                                                delete_xy_coords=True, reference=reference, 
                                                                do_apply_warp=True, save_raw=False, coadd=coadd, do_rebin=do_rebin)
    gc.collect()
    return coadd, warp_list, qmasks, rstats

def coadd_wise(tile, cowcs, WISE, ps, band, mp1, mp2,
               do_cube, medfilt, plots2=False, table=True, do_dsky=False,
               checkmd5=False, bgmatch=False, minmax=False, rchi_fraction=0.01, do_cube1=False, 
               recover=None, do_rebin=True, warp_all=False, reference_dir=None, hi_lo_rej=False):
    L = 3
    W = int(cowcs.get_width())
    H = int(cowcs.get_height())
    # For W4, single-image ww is ~ 1e-10
    tinyw = 1e-16

    reference = (reference_image_from_dir(reference_dir, tile, band) if warp_all else None)
    warp_list = None # dummy return value which may or may not get updated
    rstats = None # dummy return value which may or may not get updated

    # Round-1 coadd:
    (rimgs, r1_coadd, rstats, cube1) = _coadd_wise_round1(
        cowcs, WISE, ps, band, table, L, tinyw, mp1, medfilt, checkmd5,
        bgmatch, do_cube1, reference=reference, recover=recover, hi_lo_rej=hi_lo_rej)

    if not warp_all:
        assert(len(rimgs) == len(WISE))

    if warp_all:
        warp_list = []
        for _rimg in rimgs:
            if _rimg.warp is not None:
                warp_list.append(_rimg.warp)

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
        num_round1_images = len(rimgs)
        while len(rimgs):
            ri += 1
            rr = rimgs.pop(0)
            if ps and plots2:
                plotfn = ps.getnext()
            else:
                plotfn = None
            if not warp_all:
                this_scan_id = WISE.scan_id[ri]
                this_frame_num = WISE.frame_num[ri]
            else:
                this_scan_id = rr.scan_id
                this_frame_num = rr.frame_num
            scanid = ('scan %s frame %i band %i' %
                     (this_scan_id, this_frame_num, band)) # really should clean this up ...
            mm = _coadd_one_round2(
                (ri, num_round1_images, scanid, rr, r1_coadd.cow1, r1_coadd.cowimg1, r1_coadd.cowimgsq1, tinyw,
                 plotfn, ps1, do_dsky, rchi_fraction, r1_coadd.con1))
            coadd.acc(mm, delmm=delmm)
            if mm is not None:
                mm.scan_id = this_scan_id
                mm.frame_num = this_frame_num
            masks.append(mm)
    else:
        # this will be screwed up if warp_all is True ...
        args = []
        N = len(WISE)
        for ri,rr in enumerate(rimgs):
            if ps and plots2:
                plotfn = ps.getnext()
            else:
                plotfn = None
            scanid = ('scan %s frame %i band %i' %
                      (WISE.scan_id[ri], WISE.frame_num[ri], band))
            args.append((ri, N, scanid, rr, r1_coadd.cow1, r1_coadd.cowimg1, r1_coadd.cowimgsq1, tinyw,
                         plotfn, ps1, do_dsky, rchi_fraction, r1_coadd.con1))
        masks = mp2.map(_bounce_one_round2, args)
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
            i += 1
            masks.append(mm)
            if mm is None:
                continue
            coadd.acc(mm, delmm=delmm)
            if i == inext:
                inext *= 2
                info('Accumulated', i, 'of', Nimgs, ':', Time()-t0)

    t0 = Time()
    debug('Before garbage collection:', Time()-t0)
    gc.collect()
    debug('After garbage collection:', Time()-t0)

    # this should be only for the case of full-depth warp recovery
    if (recover is not None) and (reference is None):
        # recover contains Moon-contaminated subset of rows from exposure metadata table
        coadd_copy = deepcopy(coadd)
        coimg,  coinvvar,  coppstd,  con, coimgb, coinvvarb, coppstdb, conb, cube = extract_round2_outputs(coadd_copy, tinyw)
        reference = ReferenceImage(coimg, coppstd, con)
        # does cowcs need to be **full** coadd WCS here ?? think so..
        zp_lookup_obj = ZPLookUp(band, poly=True)
        coadd, warp_list, qmasks, rstats = recover_warped_frames(recover, coadd, reference, cowcs, zp_lookup_obj, r1_coadd, do_rebin=do_rebin)
    else:
        qmasks = None # dummy return value

    coimg,  coinvvar,  coppstd,  con, coimgb, coinvvarb, coppstdb, conb, cube = extract_round2_outputs(coadd, tinyw)

    # think it's best to only do coadd-level sky subtraction
    # *after* attempting to recover Moon-contaminated frames
    coimg, coimgb, sky = subtract_coadd_sky(coimg, coimgb, con)

    coadd.finish()
    return (coimg,  coinvvar,  coppstd,  con,
            coimgb, coinvvarb, coppstdb, conb,
            masks, cube, sky,
            coadd.comin, coadd.comax, coadd.cominb, coadd.comaxb, warp_list, qmasks, rstats)

def extract_round2_outputs(coadd, tinyw):
    # coadd is an object of type coaddacc

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

    #### HACK ####
    if 0:
        fitsio.write('coimgsq.fits', coimgsq)
        fitsio.write('cow.fits', cow)
        fitsio.write('coimg.fits', coimg)
        fitsio.write('con.fits', con)

        fitsio.write('coimgsqb.fits', coimgsqb)
        fitsio.write('cowb.fits', cowb)
        fitsio.write('coimgb.fits', coimgb)
        fitsio.write('conb.fits', conb)
    ##############

    return (coimg,  coinvvar,  coppstd,  con, coimgb, 
            coinvvarb, coppstdb, conb, cube)

def subtract_coadd_sky(coimg, coimgb, con):
    # re-estimate and subtract sky from the coadd
    try:
        sky = estimate_mode(coimgb[con != 0]) # ignore zero coverage regions
        print 'Estimated coadd sky:', sky
        coimg  -= sky
        coimgb -= sky
    except np.linalg.LinAlgError:
        print('WARNING: Failed to estimate sky in coadd:')
        import traceback
        traceback.print_exc()
        sky = 0.

    return coimg, coimgb, sky

def _coadd_one_round1((i, N, wise, table, L, ps, band, cowcs, medfilt,
                       do_check_md5, zp_lookup_obj), store_xy_coords=False):
    '''
    For multiprocessing, the function called to do round 1 on a single
    input frame.
    '''
    (i, N, wise, table, L, ps, band, cowcs, medfilt) = X
    t00 = Time()
    debug('Coadd round 1, image', (i+1), 'of', N)
    intfn = wise.intfn
    uncfn = intfn.replace('-int-', '-unc-')
    if unc_gz and (not int_gz):
        uncfn = uncfn + '.gz'
    maskfn = intfn.replace('-int-', '-msk-')
    if mask_gz and (not int_gz):
        maskfn = maskfn + '.gz'
    debug('intfn', intfn)
    debug('uncfn', uncfn)
    debug('maskfn', maskfn)

    wcs = wise.wcs
    x0,x1,y0,y1 = wise.imextent
    wcs_full = wcs # going to put this into FirstRoundImage object
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

    if not use_zp_meta:
        zp = zp_lookup_obj.get_zp(ihdr['MJD_OBS'])
    else:
        zp = ihdr['MAGZP']

    zpscale = 1. / zeropointToScale(zp)
    print 'Zeropoint:', zp, '-> scale', zpscale
    pa = ihdr['PA'] # for john fowler

    asce = ascending(ihdr['INEVENTS'])

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

    rr = FirstRoundImage()
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
    scan_frame_int = int_from_scan_frame(wise.scan_id, wise.frame_num)
    np.random.seed(scan_frame_int)
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
    rr.pa = pa # for john fowler
    rr.ascending = asce
    rr.ncopix = len(Yo)
    rr.coextent = wise.coextent
    rr.cosubwcs = cosubwcs
    rr.wcs_full = wcs_full
    rr.cowcs_full = cowcs
    rr.scan_id = wise.scan_id
    rr.frame_num = wise.frame_num

    if store_xy_coords:
        rr.x_l1b = Xi
        rr.y_l1b = Yi
        rr.x_coadd = Xo
        rr.y_coadd = Yo

    debug(Time() - t00)
    return rr

def _coadd_wise_round1(cowcs, WISE, ps, band, table, L, tinyw, mp, medfilt,
                       checkmd5, bgmatch, cube1, reference=None, recover=None, hi_lo_rej=False):
                       
    '''
    Do round-1 coadd.
    '''
    W = int(cowcs.get_width())
    H = int(cowcs.get_height())
    coimg   = np.zeros((H,W))
    coimgsq = np.zeros((H,W))
    cow     = np.zeros((H,W))
    con1    = np.zeros((H,W))

    zp_lookup_obj = ZPLookUp(band, poly=True)

    rstats = None # dummy
    if reference is None:
        args = []
        for wi,wise in enumerate(WISE):
            args.append((wi, len(WISE), wise, table, L, ps, band, cowcs, medfilt,
                         checkmd5, zp_lookup_obj))
        rimgs = mp.map(_coadd_one_round1, args)
        del args
    else:
        # this is intended for the case of a time-resolved coadd with full-depth reference available
        print 'Warping all quadrants relative to reference image'
        rimgs, _ = process_round1_quadrants(WISE, cowcs, zp_lookup_obj, r1_coadd=None, 
                                            delete_xy_coords=True, reference=reference,
                                            do_apply_warp=True, save_raw=False, coadd=None, only_good_chi2=False, debug=False,
                                            do_rebin=True)
        if recover is not None:
            rimgs_rec, rstats = process_round1_quadrants(recover, cowcs, zp_lookup_obj, r1_coadd=None, 
                                                         delete_xy_coords=True, reference=reference,
                                                         do_apply_warp=True, save_raw=False, coadd=None, only_good_chi2=True, debug=False,
                                                         do_rebin=True)
            if rimgs_rec is not None:
                while len(rimgs_rec):
                    rr_rec = rimgs_rec.pop(0)
                    if rr_rec.warped:
                        if rimgs is None:
                            rimgs = []
                        rimgs.append(rr_rec)

    if rimgs is None:
        print 'No usable frames to coadd !!'
        assert(False)

    print 'Accumulating first-round coadds...'

    if hi_lo_rej:
        hilo = HiLo()

        _t0 = _time()
        print 'constructing min and max coadd images based on first round L1b images'
        for iii, rrr in enumerate(rimgs):
            if rrr is not None:
                hilo.update(rrr)
        print (_time() - _t0), ' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cube = None
    if cube1:
        cube = np.zeros((len([rr for rr in rimgs if rr is not None]), H, W),
                        np.float32)
        z = 0
    t0 = Time()
    for wi,rr in enumerate(rimgs):
        if rr is None:
            continue
        rr.included_round1 = True
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
        id = rr.scan_id + str(rr.frame_num).zfill(3)
        if hi_lo_rej:
            good = ( ((hilo.id_min)[slc] != id) & ((hilo.id_max)[slc] != id) )
        else:
            good = 1.0

        coimgsq[slc] += rr.w * good * (rr.rimg**2)
        coimg  [slc] += rr.w * good *  rr.rimg
        cow    [slc] += rr.w * good * (rr.rmask & 1)
        con1   [slc] += good * (rr.rmask & 1)

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

    r1_coadd = FirstRoundCoadd(coimg, cow, coppstd, coimgsq, con1)
    return rimgs, r1_coadd, rstats, cube

def get_wise_frames_for_dataset(dataset, band, racen, deccen,
                                randomize=False, cache=True, dirnm=None):
    fn = '%s-frames.fits' % dataset
    if dirnm is not None:
        fn = os.path.join(dirnm, fn)
    if os.path.exists(fn) and cache:
        print 'Reading', fn
        WISE = fits_table(fn)
    else:
        WISE = get_wise_frames(racen, deccen, band)
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
    parser.add_option('--threads1', dest='threads1', type=int, default=None,
                      help='Multithreading during round 1')           
    parser.add_option('-w', dest='wishlist', action='store_true',
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

    # adding this default value of before is appropriate for first year NEOWISER processing
    # to avoid special non-public Hyades data
    parser.add_option('--before', type=float, help='Keep only input frames before the given MJD',
                      default=100000.0)
    parser.add_option('--after',  type=float, help='Keep only input frames after the given MJD')

    parser.add_option('--int_gz', dest='int_gz', action='store_true', default=False,
                      help='Are L1b int images gzipped?')
    parser.add_option('--use_zp_meta', dest='use_zp_meta', action='store_true', default=False,
                      help='Should coadd use MAGZP metadata for zero points?')
    parser.add_option('--compare_moon_all', dest='compare_moon_all', action='store_true', default=False,
                      help='When making Moon cut, determine threshold using all available frames regardless of epoch?')
    parser.add_option('--recover_warped', dest='recover_warped', action='store_true', default=False,
                      help='Attempt to recover Moon-contaminated exposures?')
    parser.add_option('--no_warp_rebin', dest='do_rebin', action='store_false', default=True,
                      help='Turn of rebinning when fitting per-quadrant polynomial warps.')
    parser.add_option('--no_irsa_dl', dest='try_download', action='store_false', default=True,
                      help='Do not attempt to download missing L1b files on the fly from IRSA.')
    parser.add_option('--warp_all', dest='warp_all', action='store_true', default=False,
                      help='For time-resolved coadd, warp all exposures relative to external reference image?')
    parser.add_option('--reference_dir', dest='reference_dir', type=str, default=None, 
                      help='Directory containing reference image when --warp_all option activated')
    parser.add_option('--no_sanity_check', dest='no_sanity_check', action='store_true', default=False,
                      help='Skip sanity checks of whether the specified combinations of options make sense.')
    parser.add_option('--hi_lo_rej', dest='hi_lo_rej', action='store_true', default=False,
                      help='Include a min/max rejection stpe during first round coaddition.')
    parser.add_option('--no_output_masks', dest='output_masks', action='store_false', default=True,
                      help='Turn off writing of per-exposure mask outputs.')

    opt,args = parser.parse_args()

    if not opt.no_sanity_check:
        sanity_check_inputs(parser)

    global int_gz
    int_gz = opt.int_gz

    global use_zp_meta
    use_zp_meta = opt.use_zp_meta

    global compare_moon_all
    compare_moon_all = opt.compare_moon_all

    if opt.threads:
        mp2 = multiproc(opt.threads)
    else:
        mp2 = multiproc()
    if opt.threads1 is None:
        mp1 = mp2
    else:
        mp1 = multiproc(opt.threads1)

    arr = os.environ.get('PBS_ARRAYID')
    if arr is not None:
        arr = int(arr)

    radec = opt.ra is not None and opt.dec is not None

    if len(args) == 0 and arr is None and not (opt.allmd5 or radec or opt.tile or opt.preprocess):
        print 'No tile(s) specified'
        parser.print_help()
        return -1

    print('unwise_coadd.py starting: args:', sys.argv)
    #print('opt:', opt)
    #print(dir(opt))

    print 'Running on host: ' + str(os.environ.get('HOSTNAME'))
    print 'Running as user: ' + str(os.environ.get('USER'))
    mkl_num_threads = os.environ.get('MKL_NUM_THREADS')
    print 'MKL_NUM_THREADS: ' + (mkl_num_threads if (mkl_num_threads is not None) else '')

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
        dataset = ''
    if opt.tile is not None:
        dataset = ''

    if dataset == 'sequels':
        # SEQUELS
        r0,r1 = 120.0, 210.0
        d0,d1 =  45.0,  60.0
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
            T = get_atlas_tiles(r0,r1,d0,d1, W,H, opt.pixscale, coadd_id=opt.tile)
            T.writeto(fn)
            print 'Wrote', fn

    if opt.band is None:
        bands = [1,2]
    else:
        bands = list(opt.band)

    if opt.plotprefix is None:
        opt.plotprefix = dataset
    ps = PlotSequence(opt.plotprefix, format='%03i')
    if opt.pdf:
        ps.suffixes = ['png','pdf']

    if not opt.plots:
        ps = None

    WISE = get_wise_frames_for_dataset(dataset, opt.band[0], T.ra, T.dec)

    if opt.allmd5:
        Ibad = check_md5s(WISE)
        print 'Found', len(Ibad), 'bad MD5s'
        for i in Ibad:
            intfn = get_l1b_file(wisedir, WISE.scan_id[i], WISE.frame_num[i], WISE.band[i], int_gz=int_gz)
            print ('(wget -r -N -nH -np -nv --cut-dirs=4 -A "*w%i*" "http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p1bm_frm/%s")' %
                   (WISE.band[i], os.path.dirname(intfn).replace(wisedir + '/', '')))
        sys.exit(0)

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

    period = opt.period
    grid = opt.grid

    kwargs = vars(opt)
    print('kwargs:', kwargs)
    # rename
    for fr,to in [('dsky', 'do_dsky'),
                  ('cube', 'do_cube'),
                  ('cube1', 'do_cube1'),
                  ('download', 'allow_download'),
                  ('ascending', 'ascendingOnly'),
                  ('descending', 'descendingOnly'),
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
                    tile.coadd_id = orig_name + '_grid_%i_%i' % (x, y)
                    print('Doing coadd grid tile', tile.coadd_id, 'band', band, 'x,y', x,y)
                    kwcopy = kwargs.copy()
                    kwcopy['zoom'] = (x*grid, min((x+1)*grid, W),
                                      y*grid, min((y+1)*grid, H))
                    kwcopy.update(ps=ps, mp1=mp1, mp2=mp2)
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

    for tileid in tiles:
        band   = (opt.band)[0]
        tileid = tileid % arrayblock
        assert(tileid < len(T))
        print 'Doing coadd tile', T.coadd_id[tileid], 'band', band
        t0 = Time()

        medfilt = opt.medfilt
        if medfilt is None:
            if band in [3,4]:
                medfilt = 50
            else:
                medfilt = 0

        if one_coadd(T[tileid], band, W, H, opt.pixscale, WISE, ps,
                     opt.wishlist, opt.outdir, mp1, mp2,
                     opt.cube, opt.plots2, opt.frame0, opt.nframes, opt.force,
                     medfilt, opt.maxmem, opt.dsky, opt.md5, opt.bgmatch,
                     opt.center, opt.minmax, opt.rchi_fraction, opt.cube1,
                     opt.epoch, opt.before, opt.after, opt.recover_warped, opt.do_rebin, opt.try_download,
                     warp_all=opt.warp_all, reference_dir=opt.reference_dir, hi_lo_rej=opt.hi_lo_rej,
                     output_masks=opt.output_masks):
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
