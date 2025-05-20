import numpy as np
import time
from scipy.ndimage.interpolation import map_coordinates
from wise_l1b_maskinterp import wise_l1b_maskinterp
from warp_utils import compute_warp
from warp_utils import mask_extreme_pix
from warp_utils import WarpMetaParameters
from warp_utils import evaluate_warp_poly

def render_warp(warp):
    # warp input should be an object of type WarpParameters
    # should this function be a method that belongs to the warp object ?

    par = WarpMetaParameters()
    # par will tell you the L1b sidelength
    warp_image = np.zeros((par.sidelen_quad, par.sidelen_quad))

    dx = warp.x_l1b_quad - warp.xmed
    dy = warp.y_l1b_quad - warp.ymed

    warp_vals = evaluate_warp_poly(warp.coeff, dx, dy)
    warp_image[warp.y_l1b_quad, warp.x_l1b_quad] = warp_vals

    # return a 508 x 508 image of the warp
    return warp_image

class WarpParameters:
    # Object to hold a warp's polynomial coefficients and reference coords
    def __init__(self, coeff, xmed, ymed, x_l1b_quad, y_l1b_quad, 
                 outlier_flag, chi2_mean, chi2_mean_raw, pred):
        # maybe i should store the chi-squared in this object as well
        self.coeff = coeff
        self.xmed = xmed
        self.ymed = ymed
        self.x_l1b_quad = x_l1b_quad
        self.y_l1b_quad = y_l1b_quad
        self.npix = len(x_l1b_quad)
        self.outlier_flag = outlier_flag
        self.n_outlier = np.sum(outlier_flag)
        # 508 x 508 bitmask with 1 indicating overlap with reference coadd
        # and zero indicating no overlap with reference coadd
        # maybe it's not necessary to compute this in constructor ..
        self.coverage = self.coverage_from_xy()
        self.chi2_mean = chi2_mean
        self.chi2_mean_raw = chi2_mean_raw
        self.pred = pred # the warp values or pixels (x_l1b_quad, y_l1b_quad)

    def coverage_from_xy(self):
        par = WarpMetaParameters()
        # should this be a boolean image rather than 1's and 0's ??
        coverage = np.zeros((par.sidelen_quad, par.sidelen_quad), 
                            dtype='int16')
        coverage[self.y_l1b_quad, self.x_l1b_quad] = 1
        return coverage

    def get_outlier_mask(self):
        par = WarpMetaParameters()
        outlier_mask = np.zeros((par.sidelen_quad, par.sidelen_quad), 
                                dtype='i2')
        x_outlier = self.x_l1b_quad[self.outlier_flag]
        y_outlier = self.y_l1b_quad[self.outlier_flag]
        outlier_mask[y_outlier, x_outlier] = 1
        return outlier_mask

def coords_for_quadrant(quad_num, two_dim=False):
    par = WarpMetaParameters()
    sidelen_quad = par.sidelen_quad

    xmin = par.get_xmin_quadrant(quad_num)
    ymin = par.get_ymin_quadrant(quad_num)
    xbox = (np.arange(sidelen_quad*sidelen_quad) % sidelen_quad) + xmin
    ybox = (np.arange(sidelen_quad*sidelen_quad) / sidelen_quad) + ymin

    if two_dim:
        xbox = xbox.reshape(sidelen_quad, sidelen_quad)
        ybox = ybox.reshape(sidelen_quad, sidelen_quad)

    return xbox, ybox

class L1bWarpQuadrant():
    # object representing a single L1b quadrant
    def __init__(self, quad_num, l1b_image, l1b_mask, coadd_image_full, 
                 coadd_unc_full, l1b_wcs, coadd_wcs):
        self.quad_num = quad_num
        self.l1b_image = l1b_image
        # should l1b mask be the full mask or just a boolean summary of
        # the full bitmask
        self.l1b_mask = l1b_mask
        self.l1b_wcs = l1b_wcs
        self.coadd_wcs = coadd_wcs

        self.xbox, self.ybox = coords_for_quadrant(self.quad_num)

        # these get assigned by extract_coadd_quadrant below
    
        # x_overlap_l1b are x coords of the *L1b image* that are contained 
        # within this quadrant and overlap the coadd

        # x_overlap_quad are x coords of the *L1b quadrant* that are 
        # contained within this quadrant and overlap the coadd

        self.x_overlap_l1b = None
        self.y_overlap_l1b = None
        self.x_overlap_quad = None
        self.y_overlap_quad = None
        self.abox = None
        self.dbox = None

        # these two are for debugging, and get assigned in extract_l1b_quadrant
        self.coadd_x = None
        self.coadd_y = None

        self.quad_int, self.quad_msk = self.extract_l1b_quadrant()

        # this assigns x_overlap, y_overlap, abox, dbox
        self.coadd_int, self.coadd_unc = self.extract_coadd_quadrant(coadd_image_full, coadd_unc_full)

        self.npix_good = len(self.x_overlap_l1b)

        par = WarpMetaParameters()
        self.enough_pix = (self.npix_good >= par.npix_min)


        # these are the actual inputs to the warp fitting, and are assigned
        # only in the event that there are sufficient overlapping pixels
        self.quad_int_fit = None
        self.coadd_int_fit = None
        self.coadd_unc_fit = None
        self.x_fit = None
        self.y_fit = None
        self.extreme_pix_mask = None

        # self.warp is meant to hold the warp summary object, in the event
        # that there are indeed sufficient pixels of overlap to fit a warp
        self.warp = None
        if self.enough_pix:
            mask = mask_extreme_pix(self.coadd_int)
            self.extreme_pix_mask = mask # maybe clean this up...

            self.x_fit = self.x_overlap_quad[mask]
            self.y_fit = self.y_overlap_quad[mask]
            self.quad_int_fit = self.quad_int[self.y_fit, self.x_fit]
            self.coadd_int_fit = self.coadd_int[mask]
            self.coadd_unc_fit = self.coadd_unc[mask]

            
            coeff, xmed, ymed, x_l1b_quad, y_l1b_quad, isgood, chi2_mean, chi2_mean_raw, pred = compute_warp(self.quad_int_fit, 
                                                                                                             self.coadd_int_fit, 
                                                                                                             self.x_fit, self.y_fit, 
                                                                                                             self.coadd_unc_fit)
            self.warp = WarpParameters(coeff, xmed, ymed, x_l1b_quad, y_l1b_quad, 
                                       ~isgood, chi2_mean, chi2_mean_raw, pred)

    def extract_l1b_quadrant(self):
        par = WarpMetaParameters()

        # be careful re: which dimension represents x vs y in python indexing
        x_l = par.get_xmin_quadrant(self.quad_num)
        x_u = par.get_xmax_quadrant(self.quad_num)
        y_l = par.get_ymin_quadrant(self.quad_num)
        y_u = par.get_ymax_quadrant(self.quad_num)
        
        quad_int = self.l1b_image[y_l:(y_u+1), x_l:(x_u+1)]
        quad_msk = self.l1b_mask[y_l:(y_u+1), x_l:(x_u+1)]

        return quad_int, quad_msk

    def extract_coadd_quadrant(self, coadd_int_full, coadd_unc_full):
        self.abox, self.dbox = (self.l1b_wcs).pixelxy2radec(self.xbox + 1, 
                                                            self.ybox + 1)

        _, coadd_x, coadd_y = (self.coadd_wcs).radec2pixelxy(self.abox, 
                                                             self.dbox)
        coadd_x -= 1
        coadd_y -= 1

        # for debugging
        self.coadd_x = coadd_x
        self.coadd_y = coadd_y

        # not clear that i got imageh and imagew right, but the coadds are
        # square 2048 x 2048 so it shouldn't matter
        good = (coadd_x > 0) & (coadd_x < self.coadd_wcs.imageh)
        good = good & (coadd_y > 0) & (coadd_y < self.coadd_wcs.imagew)

        # do i want to flatten these?
        self.x_overlap_l1b = np.ravel(self.xbox[good])
        self.y_overlap_l1b = np.ravel(self.ybox[good])

        if len(self.x_overlap_l1b) == 0:
            return None, None

        par = WarpMetaParameters()
        self.x_overlap_quad = self.x_overlap_l1b-par.get_xmin_quadrant(self.quad_num)
        self.y_overlap_quad = self.y_overlap_l1b-par.get_ymin_quadrant(self.quad_num)

        sidelen_coadd = self.coadd_wcs.imageh

        x_coadd = np.arange(sidelen_coadd)
        y_coadd = np.arange(sidelen_coadd)

        # order of x, y ???
        coadd_int = map_coordinates(coadd_int_full,[coadd_y[good],
                                                    coadd_x[good]],
                                                    order=1, mode='nearest')
        coadd_unc = map_coordinates(coadd_unc_full, [coadd_y[good],
                                                    coadd_x[good]],
                                                    order=1, mode='nearest')
        return coadd_int, coadd_unc

class L1bQuadrantWarper:
    # object encapsulating the entire warping process
    def __init__(self, l1b_image, l1b_mask, coadd_image, coadd_unc, coadd_n,
                 l1b_wcs, coadd_wcs):
        # l1b_image and coadd_image should both be in vega nanomaggies !!!
        t0 = time.time()
        self.l1b_image = wise_l1b_maskinterp(l1b_image, l1b_mask)
        dt = time.time()-t0
        print str(dt) + ' ........................'
        self.l1b_mask = l1b_mask
        self.coadd_image = coadd_image
        self.coadd_unc = coadd_unc*np.sqrt(coadd_n) # HACK !!!!
        self.coadd_n = coadd_n
        self.l1b_wcs = l1b_wcs
        self.coadd_wcs = coadd_wcs
        self.quadrant1 = L1bWarpQuadrant(1, self.l1b_image, l1b_mask, 
                                         coadd_image, self.coadd_unc, 
                                         l1b_wcs, coadd_wcs)
        self.quadrant2 = L1bWarpQuadrant(2, self.l1b_image, l1b_mask, 
                                         coadd_image, self.coadd_unc, 
                                         l1b_wcs, coadd_wcs)
        self.quadrant3 = L1bWarpQuadrant(3, self.l1b_image, l1b_mask, 
                                         coadd_image, self.coadd_unc, 
                                         l1b_wcs, coadd_wcs)
        self.quadrant4 = L1bWarpQuadrant(4, self.l1b_image, l1b_mask, 
                                         coadd_image, self.coadd_unc,
                                         l1b_wcs, coadd_wcs)

    def get_quadrant(self, quad_num):
        if quad_num == 1:
            return self.quadrant1
        elif quad_num == 2:
            return self.quadrant2
        elif quad_num == 3:
            return self.quadrant3
        elif quad_num == 4:
            return self.quadrant4

    def get_num_warps(self):
        num_warps = int(self.quadrant1.enough_pix) + int(self.quadrant2.enough_pix) + int(self.quadrant3.enough_pix) + int(self.quadrant4.enough_pix)
        return num_warps

# optimize wise_l1b_maskinterp, right now it's dumbly inefficient

# add a verbose keyword to compute warp to dictate whether various
# printouts happen or not

# do i want to have a field in the quadrant class that stores the coadd
# as a rectangular image rather than a 1d array of values

# consider rebinning before polynomial fit in order to speed things up --- 
# this will create a huge headache in terms of indexing and edge cases

# add a function to the WarpParameters class to have a warp object
# convert itself into a fits-writable structure/table ?
