import warp_l1b_quadrants
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
from astrometry.util.util import Tan
from astrometry.util.util import Sip
import pyfits
from unwise_coadd import get_l1b_file
from unwise_coadd import zeropointToScale
import time
import fitsio

def test_all_fake():
    # this is totally broken ... need to implement
    # more of L1bWarpQuadrant
    l1b_image_fake = np.zeros((1016, 1016))
    l1b_image_fake[0:508, 0:508] = 100
    l1b_image_fake[0:508, 508:1016] = 200
    l1b_image_fake[508:1016, 508:1016] = 300
    l1b_image_fake[508:1016, 0:508] = 400

    coadd_image_fake = np.zeros((2048, 2048))
    l1b_wcs_fake = None
    coadd_wcs_fake = None
    l1b_mask_fake = np.zeros((1016, 1016))
    coadd_unc_fake = np.zeros((2048, 2048))

    warper = warp_l1b_quadrants.L1bQuadrantWarper(l1b_image_fake, 
                                                  l1b_mask_fake, 
                                                  coadd_image_fake,
                                                  coadd_unc_fake, 
                                                  l1b_wcs_fake, 
                                                  coadd_wcs_fake)
    return warper

def test_warp_fitting():
    # specifically test the warp polynomial fitting using fake data to make
     # sure it's sane

    sidelen = 508
    npix = sidelen*sidelen
    xbox = np.arange(npix) % sidelen
    ybox = np.arange(npix) / sidelen

    xmed = np.median(xbox)
    ymed = np.median(ybox)

    print xmed, ymed

    coadd_bg_level = 100.0
    quad_fake_coadd = np.zeros((sidelen, sidelen)) + coadd_bg_level

    l1b_offs = 300.0
    quad_fake_l1b = np.zeros((sidelen, sidelen)) + coadd_bg_level + l1b_offs
    quad_fake_l1b += np.random.normal(loc=0.0, scale=10, 
                                      size=(sidelen, sidelen))
    unc_ref = np.zeros((sidelen, sidelen)) + 10.

#    quad_fake_l1b[ybox, xbox] += 0.025*(xbox)
#    quad_fake_l1b[ybox, xbox] -= 0.025*(ybox)
#    quad_fake_l1b[ybox, xbox] += (2e-4)*(xbox-xmed)*(ybox-ymed)

    quad_fake_l1b[ybox, xbox] -= (2.0e-4)*((xbox-xmed)**2)
    quad_fake_l1b[ybox, xbox] += (2.0e-4)*((ybox-ymed)**2)

    #npix = len(np.ravel(quad_fake_coadd))

    #xbox = np.arange(npix) % sidelen
    #ybox = np.arange(npix) / sidelen

    warp = warp_l1b_quadrants.compute_warp(np.ravel(quad_fake_l1b), 
                                           np.ravel(quad_fake_coadd), 
                                           np.ravel(xbox), 
                                           np.ravel(ybox), 
                                           np.ravel(unc_ref))

    # figure out an appropriate color stretch
    vmin = scoreatpercentile(quad_fake_l1b, 5)
    vmax = scoreatpercentile(quad_fake_l1b, 95)

    plt.imshow(quad_fake_l1b, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title('L1b quadrant')
    plt.show()

    warp_image = warp_l1b_quadrants.render_warp(warp)

    plt.imshow(quad_fake_l1b-warp_image, cmap='gray', vmin=vmin-l1b_offs, 
               vmax=vmax-l1b_offs)
    plt.title('corrected L1b quadrant')
    plt.show()

    plt.imshow(warp_image, cmap='gray')
    plt.title('polynomial warp')
    plt.show()


    # this should be very close to coadd_bg_level
    print np.median(quad_fake_l1b-warp_image)

    return warp
#    print xmed, ymed
#    print coeff
#    print type(coeff), coeff.shape

# the right way to do test_warp_fitting is to construct a WarpParameters 
# object, then pass it to test_warp_fitting, then use the object's parameters
# to generate fake data, then see if the recovered polynomial makes sense

# another good way to do it would be to (in this file) create a *separate*
# object type for holding polynomial parameters (this avoids potential for
# circularity)

def test_wcs(opt='full'):
    coadd_id = '3433p000'
    fname_coadd = '/project/projectdirs/cosmo/data/unwise/unwise-coadds/343/3433p000/unwise-3433p000-w1-img-m.fits'

    if opt == 'full':
        # choose an example that's fully contained within the tile
        fname_l1b = '/project/projectdirs/cosmo/data/wise/merge/merge_p1bm_frm/0a/05200a/118/05200a118-w1-int-1b.fits'
        scan_id = '05200a'
        frame_num = 118
    elif opt == 'part':
        # choose an example that's partially contained within the tile
        fname_l1b = '/project/projectdirs/cosmo/data/wise/merge/merge_p1bm_frm/7b/05237b/140/05237b140-w1-int-1b.fits'
        scan_id = '05237b'
        frame_num = 140
    elif opt == 'none':
        # choose an example that has no overlap with the tile
        fname_l1b = '/project/projectdirs/cosmo/data/wise/merge/merge_p1bm_frm/2a/05232a/115/05232a115-w1-int-1b.fits'
        scan_id = '05232a'
        frame_num = 115

    sidelen = 1016
    npix = sidelen*sidelen
    xbox = np.arange(npix) % sidelen
    ybox = np.arange(npix) / sidelen

    xbox = xbox.reshape(sidelen, sidelen)
    ybox = ybox.reshape(sidelen, sidelen)

    wcs_l1b = Sip(fname_l1b) 
    wcs_coadd = Tan(fname_coadd)

    abox, dbox = wcs_l1b.pixelxy2radec(xbox + 1, ybox + 1)

    ok, x_coadd, y_coadd = wcs_coadd.radec2pixelxy(abox, dbox)

    x_coadd -= 1
    y_coadd -= 1

    print np.min(x_coadd),np.max(x_coadd)
    print np.min(y_coadd),np.max(y_coadd)

    print ok.dtype, ok.shape, np.unique(ok)

    print len(np.unique(xbox)), len(np.unique(ybox))
    print xbox.shape
    print ybox.shape
    print abox.shape
    print dbox.shape

    print np.min(abox), np.max(abox)
    print np.min(dbox), np.max(dbox)

def test_warper():
    # try to create a L1bQuadrantWarper instance
    
    # define l1b filename
    fname_l1b_int = '/project/projectdirs/cosmo/data/wise/merge/merge_p1bm_frm/0a/05200a/118/05200a118-w1-int-1b.fits'
    fname_l1b_msk =  '/project/projectdirs/cosmo/data/wise/merge/merge_p1bm_frm/0a/05200a/118/05200a118-w1-msk-1b.fits.gz'

    # define coadd filename
    coadd_id = '3433p000'
    fname_coadd_int = '/scratch1/scratchdirs/ameisner/unwise-coadds/fulldepth_zp/343/3433p000/unwise-3433p000-w1-img-u.fits'
    fname_coadd_unc =  '/scratch1/scratchdirs/ameisner/unwise-coadds/fulldepth_zp/343/3433p000/unwise-3433p000-w1-std-u.fits'

    # read in l1b image, assign to l1b_image
    hdus = pyfits.open(fname_l1b_int)
    l1b_image = hdus[0].data

    # read in l1b mask, assign to l1b_mask
    hdus = pyfits.open(fname_l1b_msk)
    l1b_mask = hdus[0].data

    # read in l1b wcs, assign to l1b_wcs
    l1b_wcs = Sip(fname_l1b_int)

    # read in coadd image, assign to coadd_image
    hdus = pyfits.open(fname_coadd_int)
    coadd_image = hdus[0].data

    # read in coadd uncertainty image, assign to coadd_unc
    hdus = pyfits.open(fname_coadd_unc)
    coadd_unc = hdus[0].data

    # read in coadd wcs, assign to coadd_wcs
    coadd_wcs = Tan(fname_coadd_int)

    my_warper = warp_l1b_quadrants.L1bQuadrantWarper(l1b_image, l1b_mask,
                                                     coadd_image, coadd_unc, 
                                                     l1b_wcs, coadd_wcs)
    return my_warper

def test_warper_many_exp(band=1, quad_num=1):

    band_str = str(band)
    coadd_id = '3433p000'

    # read in the unwise-????p???-w?-frames.fits table
    fname_frames = '/scratch1/scratchdirs/ameisner/unwise-coadds/fulldepth_zp/343/3433p000/unwise-3433p000-w' + band_str + '-frames.fits'
    hdus = pyfits.open(fname_frames)
    tab = hdus[1].data

    # number of exposures to loop over
    nexp = len(tab)

    # read in the coadd image and coadd uncertainty and coadd wcs

    fname_coadd_int = '/scratch1/scratchdirs/ameisner/unwise-coadds/fulldepth_zp/343/3433p000/unwise-3433p000-w1-img-u.fits'
    fname_coadd_unc =  '/scratch1/scratchdirs/ameisner/unwise-coadds/fulldepth_zp/343/3433p000/unwise-3433p000-w1-std-u.fits'
    fname_coadd_n =  '/scratch1/scratchdirs/ameisner/unwise-coadds/fulldepth_zp/343/3433p000/unwise-3433p000-w1-n-u.fits'

    hdus = pyfits.open(fname_coadd_int)
    coadd_image = hdus[0].data
    hdus = pyfits.open(fname_coadd_unc)
    coadd_unc = hdus[0].data
    coadd_wcs = Tan(fname_coadd_int)
    hdus = pyfits.open(fname_coadd_n)
    coadd_n = hdus[0].data

    for i in range(nexp):
        if not tab['moon_rej'][i]:
            continue

        print 'working on exposure ' + str(i+1) + ' of ' + str(nexp)
    #
    #   construct L1b file name
        basedir = '/project/projectdirs/cosmo/data/wise/merge/merge_p1bm_frm'
        fname_l1b = get_l1b_file(basedir, tab['scan_id'][i], tab['frame_num'][i], band)
        print fname_l1b
    #   read in l1b image and mask and wcs
        hdus = pyfits.open(fname_l1b)
        l1b_image = hdus[0].data

        h_l1b = fitsio.read_header(fname_l1b, 0)

    # convert l1b_image to be in vega nanomaggies !!!
        print str(h_l1b.get('MAGZP')) + ' >>>>>>>>>>'
        zpscale = 1. / zeropointToScale(h_l1b.get('MAGZP'))
        l1b_image = l1b_image*zpscale

        l1b_wcs = Sip(fname_l1b)

        fname_l1b_msk = (fname_l1b.replace('int', 'msk')) + '.gz'
        hdus = pyfits.open(fname_l1b_msk)
        l1b_mask = hdus[0].data

#        t0 = time.time()
    #   create a warper object with all of the stuff i read in
        my_warper = warp_l1b_quadrants.L1bQuadrantWarper(l1b_image, l1b_mask,
                                                     coadd_image, coadd_unc, 
                                                     coadd_n, l1b_wcs, coadd_wcs)
        if (my_warper.get_quadrant(quad_num)).npix_good >= 86000:
            # plot some stuff
            plt.figure(figsize=(16,3))
            coadd_cutout = np.zeros((508, 508))
            coadd_cutout[my_warper.get_quadrant(quad_num).y_fit, my_warper.get_quadrant(quad_num).x_fit] = my_warper.get_quadrant(quad_num).coadd_int_fit
            quad_cutout = np.zeros((508, 508))
            quad_cutout[my_warper.get_quadrant(quad_num).y_fit, my_warper.get_quadrant(quad_num).x_fit] = my_warper.get_quadrant(quad_num).quad_int_fit
            quad_cutout -= np.median(my_warper.get_quadrant(quad_num).quad_int_fit)
            print my_warper.get_quadrant(1).npix_good, my_warper.get_quadrant(2).npix_good, my_warper.get_quadrant(3).npix_good,my_warper.get_quadrant(4).npix_good
            warp = my_warper.get_quadrant(quad_num).warp
            warp_image = warp_l1b_quadrants.render_warp(warp)
            warp_image -= np.median(warp_image[warp_image != 0])

            pred_warp_image = np.zeros((508,508))
            pred_warp_image[warp.y_l1b_quad, warp.x_l1b_quad] = warp.pred
            pred_warp_image -= np.median(warp.pred)
            print str(warp.chi2_mean) + ' &&&&&&&&&&&&&&&&&&&&&&'
            plt.subplot(1,4,1)
            plt.imshow(quad_cutout, origin='lower', vmin=-10, vmax=40, interpolation='nearest', cmap='gray')
            plt.subplot(1,4,2)
            plt.imshow(coadd_cutout, origin='lower', vmin=-10, vmax=40, interpolation='nearest', cmap='gray')
            plt.subplot(1,4,3)
            plt.imshow(warp_image, origin='lower', vmin=-10, vmax=40, interpolation='nearest', cmap='gray')
            plt.subplot(1,4,4)
            #plt.imshow(pred_warp_image, origin='lower', vmin=-10, vmax=40, interpolation='nearest', cmap='gray')
            plt.imshow(quad_cutout-warp_image, origin='lower', vmin=-10, vmax=40, interpolation='nearest', cmap='gray')
            plt.show()

        #print str(dt) + ' !!!!!'
