import numpy as np
import time
from scipy.ndimage.interpolation import map_coordinates
from wise_l1b_maskinterp import wise_l1b_maskinterp

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

def render_warp(warp):
    # warp input should be an object of type WarpParameters
    # should this function be a method that belongs to the warp object ?

    par = WarpMetaParameters()
    # par will tell you the L1b sidelength
    warp_image = np.zeros((par.sidelen_quad, par.sidelen_quad))

    dx = warp.x_l1b_quad - warp.xmed
    dy = warp.y_l1b_quad - warp.ymed

    warp_vals = (warp.coeff[0]) + \
                (warp.coeff[1])*dx + \
                (warp.coeff[2])*dy  + \
                (warp.coeff[3])*(dx*dy) + \
                (warp.coeff[4])*(dx**2) + \
                (warp.coeff[5])*(dy**2) + \
                (warp.coeff[6])*(dx**2)*dy + \
                (warp.coeff[7])*(dy**2)*dx + \
                (warp.coeff[8])*(dx**3) + \
                (warp.coeff[9])*(dy**3) + \
                (warp.coeff[10])*(dx**2)*(dy**2) + \
                (warp.coeff[11])*(dx**3)*dy + \
                (warp.coeff[12])*(dy**3)*dx + \
                (warp.coeff[13])*(dx**4) + \
                (warp.coeff[14])*(dy**4)
    warp_image[warp.y_l1b_quad, warp.x_l1b_quad] = warp_vals

    # return a 508 x 508 image of the warp
    return warp_image

def compute_warp(pix_l1b_quad, pix_ref, x_l1b_quad, y_l1b_quad, unc_ref):
    # pix_l1b_quad and pix_ref should be flattened, no need to have them  
    # actually be 2D images here

    diff = pix_l1b_quad - pix_ref
    npix = len(diff)

    xmed = np.median(x_l1b_quad)
    ymed = np.median(y_l1b_quad)

    dx = x_l1b_quad - xmed
    dy = y_l1b_quad - ymed

    print xmed
    print ymed

    t0 = time.time()
    X = np.column_stack( (np.ones(npix), 
                          dx, 
                          dy, 
                          dx*dy,
                          dx**2, 
                          dy**2, 
                          (dx**2)*dy,
                          (dy**2)*dx,
                          dx**3,
                          dy**3,
                          (dx**2)*(dy**2), 
                          (dx**3)*dy,
                          (dy**3)*dx,
                          dx**4,
                          dy**4) )

    dt = time.time()-t0
    print dt

    t0 = time.time()
    coeff, __, ___, ____ = np.linalg.lstsq(X, diff)
    dt = time.time()-t0
    print dt

    pred = np.dot(X, coeff)
    resid = (diff - pred)

    # try to mimic hogg_iter_linfit
    sig_thresh = 3.
    ms  = np.mean(resid**2)
    isgood = ((resid**2) < (sig_thresh**2)*ms)

    # redo the fit with outliers removed
    coeff, __, ___, ____ = np.linalg.lstsq(X[isgood], diff[isgood])

    print coeff, len(coeff) , ' !!!!!!!!!!!'

    #print np.min(pred), np.max(pred)

    # calculate the mean chi-squared
    # i think the mean chi-squared should be calculated including *all* pixels
    pred = np.dot(X, coeff)
    resid = (diff - pred)
    chi2_image = (resid/unc_ref)**2
    chi2_mean = np.mean(chi2_image)

    # should chi2_mean_raw be calculated after requiring that
    # reference quadrant and l1b quadrant be made to have matching medians?
    chi2_mean_raw = np.mean(((pix_l1b_quad - pix_ref)/(unc_ref))**2)
    print chi2_mean, '~~~~~~~'

    warp = WarpParameters(coeff, xmed, ymed, x_l1b_quad, y_l1b_quad, 
                          ~isgood, chi2_mean, chi2_mean_raw, pred)
    return warp

def mask_extreme_pix(image, ignore=None):
    # ignore is meant to be a btimask flagging pixels that should be ignored
    # the use case i have in mind is to ignore any zero value pixels, which
    # could be there in the case of missing data


    # variables image and ignore should have same dimensions, maybe should check

    # pct_vals is list of values to actually use in computing 5th and 95th
    # percentiles
    pct_vals = image if ignore is None else image[~ignore] 
    hi_lo = np.percentile(pct_vals, [5, 95])
    lo = hi_lo[0]
    hi = hi_lo[1]

    # good = non-extreme = True, bad = extreme = False
    extreme_pix_mask = ~((image < lo) | (image > hi))

    # also mark "iginored" pixels as bad, just to be safe?
    if ignore is not None:
        extreme_pix_mask = ((extreme_pix_mask) & (~ignore))

    return extreme_pix_mask

class WarpMetaParameters:
    # object holding various special numbers
    def __init__(self):
        self.npix_min = 86000 # roughly one third of L1b quadrant
        self.sidelen_quad = 508 # this is wrong for W4 ...
        self.l1b_sidelen = 1016 # this is wrong for W4 ...
        self.warp_order = 4 # order of per-quadrant polynomial correction

        # worst goodness-of-fit for a quadrant to be considered recovered
        self.chi2_mean_thresh = 4.5

#                           Q1    Q2   Q3    Q4
        self.xmin_list = [ 508,    0,   0,  508]
        self.xmax_list = [1015,  507, 507, 1015]
        self.ymin_list = [ 508,  508,   0,    0]
        self.ymax_list = [1015, 1015, 507,  507]

    def get_xmin_quadrant(self, quad_num):
        return self.xmin_list[quad_num - 1]

    def get_xmax_quadrant(self, quad_num):
        return self.xmax_list[quad_num - 1]

    def get_ymin_quadrant(self, quad_num):
        return self.ymin_list[quad_num - 1]

    def get_ymax_quadrant(self, quad_num):
        return self.ymax_list[quad_num - 1]

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

        # self.warp is meant to hold the warp summary object, in the event
        # that there are indeed sufficient pixels of overlap to fit a warp
        self.warp = None
        if self.enough_pix:
            mask = mask_extreme_pix(self.coadd_int)

            self.x_fit = self.x_overlap_quad[mask]
            self.y_fit = self.y_overlap_quad[mask]
            self.quad_int_fit = self.quad_int[self.y_fit, self.x_fit]
            self.coadd_int_fit = self.coadd_int[mask]
            self.coadd_unc_fit = self.coadd_unc[mask]

            
            self.warp = compute_warp(self.quad_int_fit, self.coadd_int_fit, 
                                     self.x_fit, self.y_fit, 
                                     self.coadd_unc_fit)

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
        # l1b_image and coadd_imae should both be in vega nanomaggies !!!
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

# optimize wise_l1b_maskinterp, right now it's dumbly inefficient

# add a verbose keyword to compute warp to dictate whether various
# printouts happen or not

# do i want to have a field in the quadrant class that stores the coadd
# as a rectangular image rather than a 1d array of values

# consider rebinning before polynomial fit in order to speed things up --- 
# this will create a huge headache in terms of indexing and edge cases

# add a function to the WarpParameters class to have a warp object
# convert itself into a fits-writable structure/table ?
