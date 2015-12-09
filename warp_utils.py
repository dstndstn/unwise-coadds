import numpy as np
import time

def evaluate_warp_poly(coeff, dx, dy):
    warp_vals = (coeff[0]) + \
                (coeff[1])*dx + \
                (coeff[2])*dy  + \
                (coeff[3])*(dx*dy) + \
                (coeff[4])*(dx**2) + \
                (coeff[5])*(dy**2) + \
                (coeff[6])*(dx**2)*dy + \
                (coeff[7])*(dy**2)*dx + \
                (coeff[8])*(dx**3) + \
                (coeff[9])*(dy**3) + \
                (coeff[10])*(dx**2)*(dy**2) + \
                (coeff[11])*(dx**3)*dy + \
                (coeff[12])*(dy**3)*dx + \
                (coeff[13])*(dx**4) + \
                (coeff[14])*(dy**4)
    return warp_vals

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
    print chi2_mean_raw,  '~~~~~~~', chi2_mean, '~~~~~~~'

    return (coeff, xmed, ymed, x_l1b_quad, y_l1b_quad, 
            isgood, chi2_mean, chi2_mean_raw, pred)

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
    def __init__(self, band=1):
        self.npix_min = 86000 # roughly one third of L1b quadrant
        self.sidelen_quad = 508 # this is wrong for W4 ...
        self.l1b_sidelen = 1016 # this is wrong for W4 ...
        self.warp_order = 4 # order of per-quadrant polynomial correction
        self.band = band

        # worst goodness-of-fit for a quadrant to be considered recovered
        self.chi2_mean_thresh = (2.5 if (band == 1) else 3.25)

        # these values are zero indexed !!
        #                   Q1    Q2   Q3    Q4
        self.xmin_list = [ 508,    0,   0,  508]
        self.xmax_list = [1015,  507, 507, 1015]
        self.ymin_list = [ 508,  508,   0,    0]
        self.ymax_list = [1015, 1015, 507,  507]

    def get_xmin_quadrant(self, quad_num, one_indexed=False):
        return (self.xmin_list[quad_num - 1] + int(one_indexed))

    def get_xmax_quadrant(self, quad_num, one_indexed=False):
        return (self.xmax_list[quad_num - 1] + int(one_indexed))

    def get_ymin_quadrant(self, quad_num, one_indexed=False):
        return (self.ymin_list[quad_num - 1] + int(one_indexed))

    def get_ymax_quadrant(self, quad_num, one_indexed=False):
        return (self.ymax_list[quad_num - 1] + int(one_indexed))
