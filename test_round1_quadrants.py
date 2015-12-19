from unwise_coadd import process_round1_quadrants
from astrometry.util.util import Tan
from astrometry.util.util import Sip
from zp_lookup import ZPLookUp
from astrometry.util.fits import fits_table
import numpy as np
from warp_utils import WarpMetaParameters
import matplotlib.pyplot as plt
import copy
from unwise_coadd import split_one_quadrant
import fitsio
from unwise_coadd import ReferenceImage
from warp_utils import evaluate_warp_poly
from warp_utils import render_warp
from warp_utils import apply_warp

def add_wcs_column(WISE):
    WISE.wcs = np.zeros(len(WISE), object)
    nexp = len(WISE) 
    for i in range(nexp):
        WISE.intfn[i] = (WISE.intfn[i]).replace(' ', '')
    for i in range(nexp):
        print WISE.intfn[i]
        WISE.wcs[i] = Sip(WISE.intfn[i])
    return WISE

def assemble_quadrant_objects(nmax=20, moon_rej=False, reference=None, band=1, only_good_chi2=False):
# choose a name for file from which WISE will be read
#    tabname = '/global/cscratch1/sd/ameisner/unwise_test_tiles/foo20/e0_moon/343/3433p000/unwise-3433p000-w'+str(band)+'-frames.fits'
    tabname = '/global/cscratch1/sd/ameisner/unwise_test_tiles/foo63/moon/343/3433p000/unwise-3433p000-w'+str(band)+'-frames.fits'
# this implies a choice of coadd..
    coaddname = '/global/cscratch1/sd/ameisner/unwise_test_tiles/foo63/moon/343/3433p000/unwise-3433p000-w'+str(band)+'-img-m.fits'

# read (or create) the coadd header WCS
    cowcs = Tan(coaddname)

# choice of frames table file name also implies a choice of band
# so use this choice to create a zp_lookup_obj using poly=True
    zp_lookup_obj = ZPLookUp(band, poly=True)

# possibly trim WISE to a smaller number of rows, so that testing
# runs  faster
    WISE = fits_table(tabname)
    if moon_rej:
        WISE = WISE[WISE.moon_rej == True]
    else:
        WISE = WISE[WISE.moon_rej == False]
    print WISE.intfn
    print WISE.use
    WISE.cut(WISE.use == 1)
    #print WISE.intfn
    nmax = min(nmax, len(WISE))
    WISE = WISE[0:nmax]
    WISE = add_wcs_column(WISE)
    print len(WISE), ' !!!!!!!!!!!!!!'
    
    #print WISE.intfn
    rimgs = process_round1_quadrants(WISE, cowcs, zp_lookup_obj, 
                                     reference=reference, do_apply_warp=True, save_raw=True, only_good_chi2=only_good_chi2)
    return rimgs, WISE

def create_reference(band=1):
    coaddname = '/global/cscratch1/sd/ameisner/unwise-coadds/fulldepth_zp/343/3433p000/unwise-3433p000-w'+str(band)+'-img-u.fits'
    print 'reading reference tile : ' + coaddname + ' ~~~~~~~~~~~~~~~~~~~~~~'
    imref = fitsio.read(coaddname)
    nref = fitsio.read(coaddname.replace('img', 'n'))
    stdref = fitsio.read(coaddname.replace('img', 'std'))

    ref = ReferenceImage(imref, stdref, nref)
    return ref

def plot_quadrant_results(nmax=20, moon_rej=True, band=1):
    reference = create_reference(band=band)
    rimgs_quad, WISE = assemble_quadrant_objects(nmax=nmax, moon_rej=moon_rej, 
                                                 reference=reference, band=band)

    for rimg_quad in rimgs_quad:
        # rimg_quad = apply_warp(rimg_quad)
        print '--------------------------------------'
        print len(rimg_quad.x_l1b), len(rimg_quad.y_l1b), len(rimg_quad.x_coadd), len(rimg_quad.y_coadd)
        print np.sum(rimg_quad.rmask[rimg_quad.y_coadd, rimg_quad.x_coadd] != 0)

        warp = rimg_quad.warp
        if warp is None:
            print 'no warp was computed'
        else:
            print 'warp was computed !!!!!!!!!!!!!!!!!!!!!!  order = ' + str(warp.order)
        plt.figure(figsize=(16,10))
   
        plt.subplot(2,4,2)
        quad_im_raw = (rimg_quad.rimg if (warp is None)  else rimg_quad.rimg_bak)
        plt.imshow(quad_im_raw, vmin=-10, vmax=40, interpolation='nearest', 
                   origin='lower', cmap='gray')
        plt.title('raw L1b quadrant', fontsize=8)

        rmask_reconstructed = np.zeros(rimg_quad.rimg.shape)
        rmask_reconstructed[rimg_quad.y_coadd, rimg_quad.x_coadd] = 1

        assert(np.sum(rmask_reconstructed != (rimg_quad.rmask != 0)) == 0)

        x_l1b_im = np.zeros(rimg_quad.rimg.shape)
        y_l1b_im = np.zeros(rimg_quad.rimg.shape)

        x_l1b_im[rimg_quad.y_coadd, rimg_quad.x_coadd] = rimg_quad.x_l1b
        y_l1b_im[rimg_quad.y_coadd, rimg_quad.x_coadd] = rimg_quad.y_l1b


        plt.subplot(2,4,1)
        imref, _, __ = reference.extract_cutout(rimg_quad)
        plt.imshow(imref*(rimg_quad.rmask != 0), cmap='gray', 
                   interpolation='nearest',
                   origin='lower', vmin=-10, vmax=40)
        plt.title('reference', fontsize=8)

        if warp is not None:
            dx = x_l1b_im - warp.xmed
            dy = y_l1b_im - warp.ymed
            #warp_image = evaluate_warp_poly(warp.coeff, dx, dy)
            #warp_image = np.zeros(rimg_quad.rimg.shape)
            #warp_image *= (rimg_quad.rmask != 0)

            # try new function
            warp_image = render_warp(rimg_quad)
            plt.subplot(2,4,3)
            plt.imshow(warp_image, cmap='gray', interpolation='nearest',
                       origin='lower', vmin=-10, vmax=40)
            plt.title("{:.4f}".format(warp.chi2mean_raw) + " , " + "{:.4f}".format(warp.chi2mean), fontsize=8)

            # now construct/plot the warp-corrected quadrant
            #corr = rimg_quad.rimg - warp_image
            plt.subplot(2,4,4)
            plt.imshow(rimg_quad.rimg, cmap='gray', interpolation='nearest', origin='lower', vmin=-10, vmax=40)
            plt.title('corrected L1b quad', fontsize=8)

            # masked reference image
            plt.subplot(2,4,5)
            plt.imshow(imref*warp.non_extreme_mask, vmin=-10, vmax=40, origin='lower', interpolation='nearest', cmap='gray')
            plt.title('masked reference', fontsize=8)

            # masked quadrant image
            plt.subplot(2,4,6)
            plt.imshow(rimg_quad.rimg_bak*warp.non_extreme_mask, vmin=-10, vmax=40, origin='lower', interpolation='nearest', cmap='gray')
            plt.title('masked L1b quad raw', fontsize=8)

            # non-extreme vs. extreme reference image pixel mask
            #plt.subplot(2,4,7)
            #plt.imshow(warp.non_extreme_mask, vmin=0, vmax=1, origin='lower', interpolation='nearest', cmap='gray')

        if False:
            plt.subplot(2,4,7)
            plt.imshow(rmask_reconstructed != 0, vmin=0, vmax=1,
                   interpolation='nearest', origin='lower', cmap='gray')
            plt.title('mask reconstructed from coadd x,y coords', fontsize=8)
            plt.subplot(2,4,8)
            plt.imshow(x_l1b_im, cmap='gray', interpolation='nearest',
                       origin='lower')
            plt.title('L1b x', fontsize=8)

            plt.subplot(2,4,5)
            plt.imshow(y_l1b_im, cmap='gray', interpolation='nearest',
                       origin='lower', vmin=np.min(rimg_quad.y_l1b),
                       vmax=np.max(rimg_quad.y_l1b))
            plt.title('L1b y', fontsize=8)
            plt.subplot(2,4,6)
            plt.imshow(rimg_quad.rmask != 0, vmin=0, vmax=1,
                       interpolation='nearest', origin='lower', cmap='gray')
            plt.title('rmask != 0', fontsize=8)

        plt.show()
        #plt.clf()
        # subplots : 
        # 1) just rimg for quadrant
        # 2) just rmask for quadrant
        # 3) reconstruction of rmask based on x_coadd, y_coadd
        # 4) image of x_l1b for quadrant
        # 5) image of y_l1b for quadrant
        # ---- LATER ----
        # 6) image of reference coadd for quadrant
        # 7) image of masked reference
        # 8) rendering of the polynomial warp
        # 9) image of warp-subtracted quadrant

def recovery_stats(nmax=20, moon_rej=True, band=1, plot=False):
    # loop over quadrants, computing warps and then 
    # computing basic statistics of how many pixels/quadrants were/weren't recovered

    reference = create_reference(band=band)
    rimgs_quad, WISE = assemble_quadrant_objects(nmax=nmax, moon_rej=moon_rej, 
                                                 reference=reference, band=band)

    n_attempted = 0 # number of quadrants for which warping was attempted
    n_success = 0 # number of quadrants for which warping succeeded
    n_fail = 0 # number of quadrants for which warping was attempted but failed
    n_skipped = 0 # number of quadrants for which warping wasn't even attempted
    ntot = len(rimgs_quad)

    par = WarpMetaParameters(band=band)
    chi2mean_vals = []
    for rimg_quad in rimgs_quad:
        if rimg_quad.warp is None:
            n_skipped += 1
        else:
            n_attempted += 1
            chi2mean_vals.append(rimg_quad.warp.chi2mean)
            success = (rimg_quad.warp.chi2mean < par.chi2_mean_thresh)
            if success:
                n_success += 1
            else:
                n_fail += 1
    print str(par.chi2_mean_thresh) + ' @@@@@@@@@@@@@@@@@@@'
    assert(ntot == (n_attempted+n_skipped))
    assert(n_attempted == (n_success + n_fail))
    if plot:
        plt.hist(np.array(chi2mean_vals), histtype='step', bins=np.arange(0,15,0.2))
        plt.plot([par.chi2_mean_thresh, par.chi2_mean_thresh],[0,10], c='r')
        plt.show()

    return (ntot, n_attempted, n_success, n_fail, n_skipped, chi2mean_vals)
