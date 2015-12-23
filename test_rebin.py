import warp_utils
import pyfits
import matplotlib.pyplot as plt
import time
import numpy as np

def test_pad_rebin(xmin=0,xmax=500,ymin=0,ymax=500):
    # arbitrarily choose some coadd image to use as a testbed
    fname = '/global/cscratch1/sd/ameisner/unwise-coadds/fulldepth_zp/343/3433p000/unwise-3433p000-w1-img-u.fits'

    x_full = np.arange(2048*2048,dtype='int').reshape((2048,2048)) % 2048
    y_full = np.arange(2048*2048,dtype='int').reshape((2048,2048)) / 2048

    hdus = pyfits.open(fname)

    im_full = hdus[0].data
    im = im_full[ymin:ymax, xmin:xmax]
    x = x_full[ymin:ymax, xmin:xmax]
    y = y_full[ymin:ymax, xmin:xmax]

    # call mask_extreme_reference_pix to mask out the stars and simulate
    # having a mask
    msk = warp_utils.mask_extreme_pix(im)

    t0 = time.time()
    rebinned_images, rebinned_mask = warp_utils.pad_rebin_weighted([im,x,y], msk, binfac=2)
    dt = time.time()-t0
    print str(dt) + ' seconds !!!!!!!!!'

    return im, rebinned_images, rebinned_mask

    # call the routine that i actually want to test
#    pad_rebin_weighted(images, mask, binfac=2)
