import numpy as np
import time
import os
import fitsio
from unwise_utils import _rebin, get_dir_for_coadd

class QuadrantWarp():
    def __init__(self, quadrant, coeff, xmed, ymed, chi2mean, chi2mean_raw, order, 
                 non_extreme_mask, npix, scan_id, frame_num, debug=False):
        self.quadrant = quadrant # this is an integer ??
        self.coeff = coeff
        self.xmed = xmed
        self.ymed = ymed
        self.chi2mean = chi2mean
        self.chi2mean_raw = chi2mean_raw
        self.order = order
        if debug:
            self.non_extreme_mask = non_extreme_mask
        else:
            self.non_extreme_mask = None
        self.npix = int(npix) # number of pixels used in fit, including those rejected in iterative fit
        self.scan_id = scan_id
        self.frame_num = frame_num

class ReferenceImage():
    def __init__(self, image, std, n):
        # these need to be images corresponding to the *entire* reference tile
        self.image = image # the reference image itself
        self.n = n # the integer coverage, should be from *n-u.fits
        self.sigma = self.patch_zero_sigma(std*np.sqrt(n), n) # 1 sigma error to be used when computing warp chi2 values

    def extract_cutout(self, rimg):
        # rimg should be the FirstRoundImage object
        x0_coadd = rimg.coextent[0]
        x1_coadd = rimg.coextent[1] + 1
        y0_coadd = rimg.coextent[2]
        y1_coadd = rimg.coextent[3] + 1

        return (self.image[y0_coadd:y1_coadd, x0_coadd:x1_coadd],  
                self.sigma[y0_coadd:y1_coadd, x0_coadd:x1_coadd],
                self.n[y0_coadd:y1_coadd, x0_coadd:x1_coadd])

    def patch_zero_sigma(self, sigma, n):
        # HACK !!
        nbad = np.sum((sigma == 0) & (n != 0))
        if (nbad != 0):
            ok = patch_image(sigma, ((sigma != 0) | (n == 0)))
            assert(ok)
            print 'Patched ' + str(nbad) + ' bad reference uncertainty pixels'
            return sigma
        else:
            return sigma

    def write(self, outname):
        # for debugging
        fitsio.write(outname, self.image)
        fitsio.write(outname, self.n)
        fitsio.write(outname, self.sigma)

def evaluate_warp_poly(coeff, dx, dy):
    par = WarpMetaParameters()
    order = par.coeff2order(coeff)

    warp_vals = coeff[0]

    if order > 0:
        warp_vals += ( (coeff[1])*dx + \
                       (coeff[2])*dy )

    if order > 1:
        dx2 = (dx*dx)
        dy2 = (dy*dy)
        warp_vals += ( (coeff[3])*(dx*dy) + \
                       (coeff[4])*dx2 + \
                       (coeff[5])*dy2 )
    if order > 2:
        dx3 = (dx2*dx)
        dy3 = (dy2*dy)
        warp_vals += ( (coeff[6])*(dx2)*dy + \
                       (coeff[7])*(dy2)*dx + \
                       (coeff[8])*dx3 + \
                       (coeff[9])*dy3 )
    if order > 3:
        warp_vals += ( (coeff[10])*(dx2)*(dy2) + \
                       (coeff[11])*(dx3)*dy + \
                       (coeff[12])*(dy3)*dx + \
                       (coeff[13])*(dx2*dx2) + \
                       (coeff[14])*(dy2*dy2) )
    return warp_vals

def poly_design_matrix(dx, dy, order):
    assert((order >= 1) and (order <= 4))

    npix = len(dx)
    # construct X through first order
    X = np.column_stack( (np.ones(npix), 
                          dx, 
                          dy) )
    # if order > 1 construct and column stack the second order terms
    if (order > 1):
        dx2 = (dx*dx)
        dy2 = (dy*dy)
        X = np.column_stack( (X,
                              dx*dy,
                              dx2, 
                              dy2) )

    # if order > 2 construct and column stack the third order terms
    if (order > 2):
        dx3 = (dx2*dx)
        dy3 = (dy2*dy)
        X = np.column_stack( (X,
                              (dx2)*dy,
                              (dy2)*dx,
                              dx3,
                              dy3) )

    # if order > 3 construct and column stack the fourth order terms
    if (order > 3):
        X = np.column_stack( (X,
                              (dx2)*(dy2), 
                              (dx3)*dy,
                              (dy3)*dx,
                              dx2*dx2,
                              dy2*dy2) )
    return X

def compute_warp(pix_l1b_quad, pix_ref, x_l1b_quad, y_l1b_quad, unc_ref,
                 order=4, verbose=False):
    # pix_l1b_quad and pix_ref should be flattened, no need to have them  
    # actually be 2D images here

    assert((order >= 0) and (order <= 4))
    # should this actually use <= 0 rather than == 0 ?
    assert(np.sum(unc_ref == 0) == 0)
    assert(len(x_l1b_quad) > 0)
    assert(len(y_l1b_quad) > 0)
    assert(len(x_l1b_quad) == len(y_l1b_quad))

    diff = pix_l1b_quad - pix_ref
    npix = len(diff)

    xmed = np.median(x_l1b_quad)
    ymed = np.median(y_l1b_quad)

    par = WarpMetaParameters()
    if order > 0:
        dx = x_l1b_quad - xmed
        dy = y_l1b_quad - ymed

        X = poly_design_matrix(dx, dy, order)

        t0 = time.time()
        coeff = np.linalg.lstsq(X, diff)[0]
        if verbose: print (time.time()-t0)

        pred = np.dot(X, coeff)
        resid = (diff - pred)

        # try to mimic hogg_iter_linfit
        resid2 = (resid**2)
        ms  = np.mean(resid2)
        isgood = (resid2 < (par.sig_thresh**2)*ms)

        # redo the fit with outliers removed
        coeff = np.linalg.lstsq(X[isgood], diff[isgood])[0]
        pred = np.dot(X, coeff)
    else:
        # zeroth order case
        pred = np.median(diff)
        coeff = np.array([pred])
        isgood = np.ones(npix, dtype=bool) # ?? hack

    assert(order == par.coeff2order(coeff))

    if verbose: print coeff, len(coeff) , ' !!!!!!!!!!!'

    # calculate the mean chi-squared
    # i think the mean chi-squared should be calculated including *all* pixels
    resid = (diff - pred)
    chi2_image = (resid/unc_ref)**2
    chi2_mean = np.mean(chi2_image)

    # should chi2_mean_raw be calculated after requiring that
    # reference quadrant and l1b quadrant be made to have matching medians?
    chi2_mean_raw = np.mean(((pix_l1b_quad - pix_ref)/(unc_ref))**2)
    if verbose: print chi2_mean_raw,  '~~~~~~~', chi2_mean, '~~~~~~~'

    return (coeff, xmed, ymed, x_l1b_quad, y_l1b_quad, 
            isgood, chi2_mean, chi2_mean_raw, pred)

def render_warp(rimg_quad):
    # rimg_quad is a FirstRoundImage object representing a quadrant
    assert(rimg_quad.quadrant != -1)

    sh = rimg_quad.rimg.shape

    warp = rimg_quad.warp
    if warp is not None:
        dx = rimg_quad.x_l1b - warp.xmed
        dy = rimg_quad.y_l1b - warp.ymed
        warp_image = np.zeros(sh)

        warp_vals = evaluate_warp_poly(warp.coeff, dx, dy)
        warp_image[rimg_quad.y_coadd, rimg_quad.x_coadd] = warp_vals
        warp_image *= (rimg_quad.rmask != 0)
    else:
        # not sure if this is the right thing to do
        warp_image = np.zeros(sh)

    return warp_image

def apply_warp(rimg_quad, band, save_raw=False, only_good_chi2=False):
    # input is a FirstRoundImage object -- a modified version of this
    # object will be returned
    
    # don't ever want to doubly subtract a polynomial warp correction
    assert(rimg_quad.warped == False)
    # if rimg_quad.warp is None, or if the warp correction didn't achieve a
    # good enough chi-squared, return the input object itself
    par = WarpMetaParameters(band=band)

    if (rimg_quad.warp is None) or (only_good_chi2 and (rimg_quad.warp.chi2mean > par.chi2_mean_thresh)):
        return rimg_quad

    # save uncorrected image to rimg_bak
    rimg_bak = rimg_quad.rimg
    # call render_warp to get the warp image
    warp_image = render_warp(rimg_quad)
    # subtract the warp image from rimg_quad.rimg
    rimg_quad.rimg = (rimg_quad.rimg - warp_image)
    rimg_quad.warped = True
    print 'Subtracted polynomial warp from quadrant'

    assert(np.sum(rimg_quad.rimg != rimg_bak) != 0)

    # save_raw=False is a way to conserve RAM in event that saving
    # copy of uncorrected quadrant image is unnecessary
    if save_raw:
        rimg_quad.rimg_bak = rimg_bak

    return rimg_quad

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
    def __init__(self, band=1, binfac=1):
        self.npix_min = 86000 # roughly one third of L1b quadrant
        self.sidelen_quad = 508 # this is wrong for W4 ...
        self.l1b_sidelen = 1016 # this is wrong for W4 ...
        self.warp_order = 4 # order of per-quadrant polynomial correction
        self.band = band
        self.binfac = binfac # should be either 1 or 2 for now

        # worst goodness-of-fit for a quadrant to be considered recovered
        self.chi2_mean_thresh = 2.5 # could make this band-dependent

        # outlier threshold (in standard deviations) for iterative poly fitx
        self.sig_thresh = 3.0

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

    def npix2order(self, npix):
        # convert from number of pixels to appropriate warp polynomial order
        tol = 0.002 # somewhat arbitrary
        assert(npix <= ((1.0+tol)*(self.sidelen_quad**2)))
        assert(npix >= 0)

        if (npix >= 45000):
            return 4
        elif (npix >= 20000):
            return 3
        elif (npix >= 13000):
            return 2
        elif (npix >= 5000):
            return 1
        elif (npix >= 50):
            return 0
        else:
            return None

    def coeff2order(self, coeff):
        # determine polynomial order based on number of coefficients
        ncoeff = len(coeff)

        # clean this up by using a dictionary
        if ncoeff == 1:
            return 0 # zeroth order
        elif ncoeff == 3:
            return 1 # first order
        elif ncoeff == 6:
            return 2 # second order
        elif ncoeff == 10:
            return 3 # third order
        elif ncoeff == 15:
            return 4 # fourth order

    def get_chi2_fac(self, binfac):
        facs = {1 : 1.0, 2 : 1.315, 4 : 1.0} # binfac=4 value 1.0 is a dummy
        return facs[binfac]

def gen_warp_table(warp_list):
    # generate a table summarizing the successfully derived/applied warps
    # based on a list of QuadrantWarp objects

    # assume that warp_list contains no None entries

    nwarp = (0 if warp_list is None else len(warp_list))
    arr_out = np.zeros((nwarp,), 
                       dtype=[('scan_id','a6'),
                              ('frame_num','int'),
                              ('quad_num','uint8'),
                              ('order','uint8'),
                              ('coeff','(15,)float64'),
                              ('x_ref', 'float'),
                              ('y_ref', 'float'),
                              ('chi2_mean', 'float32'),
                              ('chi2_mean_raw', 'float32'),
                              ('npix', 'int')])

    if warp_list is None:
        return arr_out # syntactically correct but no rows

    for i,warp in enumerate(warp_list):
        arr_out['scan_id'][i] = warp.scan_id
        arr_out['frame_num'][i] = warp.frame_num
        arr_out['quad_num'][i] = warp.quadrant
        arr_out['order'][i] = warp.order
        coeff = np.zeros(15, dtype=np.float64)
        coeff[0:len(warp.coeff)] = warp.coeff
        arr_out['coeff'][i] = coeff
        arr_out['x_ref'][i] = warp.xmed
        arr_out['y_ref'][i] = warp.ymed
        arr_out['chi2_mean'][i] = warp.chi2mean
        arr_out['chi2_mean_raw'][i] = warp.chi2mean_raw
        arr_out['npix'][i] = warp.npix
        

    return arr_out

def update_included_bitmask(WISE, warp_list):
    # make the included column of WISE metadata table into a bitmask
    # indicating which quadrants were recovered
    # note that warp_list can also be a list of SecondRoundImage objects
    # provided that the SecondRoundImage objects have .scan_id and .frame_num set

    for warp in warp_list:
        assert(warp.quadrant != -1)
        if hasattr(warp, 'included'):
            if not warp.included:
                continue
        val = int(2**(warp.quadrant))
        # WISE input should be modified in calling scope
        WISE.included[(WISE.scan_id == warp.scan_id) & (WISE.frame_num == warp.frame_num)] |= val

def parse_write_quadrant_masks(outdir, tag, WISE, qmasks, int_gz, ofn, ti):

    if qmasks is None:
        return

    print 'Updating metadata based on quadrant SecondRoundImage objects.'

    # appropriately update the WISE metadata table
    for qmask in qmasks:
        # find relevant row in metadata table
        exp_mask = [(WISE.scan_id == qmask.scan_id) & (WISE.frame_num == qmask.frame_num)]
        assert(np.sum(exp_mask) == 1)

        WISE.sky1[exp_mask] = qmask.sky
        WISE.sky2[exp_mask] = qmask.dsky
        WISE.zeropoint[exp_mask] = qmask.zp
        WISE.npixoverlap[exp_mask] += qmask.ncopix
        WISE.npixpatched[exp_mask] += qmask.npatched
        WISE.npixrchi[exp_mask] += qmask.nrchipix
        WISE.weight[exp_mask] = qmask.w

    # call merge_write_quadrant_masks to actually write the bitmask images
    merge_write_quadrant_masks(outdir, tag, WISE, qmasks, int_gz, ofn, ti)

def merge_write_quadrant_masks(outdir, tag, WISE, qmasks, int_gz, ofn, ti):
    # figure out the list of unique scan_id, frame_num pairs
    # loop over each (scan_id, frame_num) pair

    if len(qmasks) == 0:
        return

    expid = [(qmask.scan_id + str(qmask.frame_num).zfill(3)) for qmask in qmasks]
    expid = np.array(expid)
    expid_u = np.unique(expid)

    width = int(np.max(WISE.imagew))
    height = int(np.max(WISE.imageh))
    masktype = qmasks[0].omask.dtype

    maskdir = os.path.join(outdir, tag + '-mask')
    if not os.path.exists(maskdir):
        os.mkdir(maskdir)

    for i, id_u in enumerate(expid_u):
        w_id = (np.where(expid == id_u))[0]
        nquad = len(w_id) # number of quadrants recovered from this exposure
        assert((nquad > 0) and (nquad <= 4))

        fullmask = np.zeros((height, width), masktype) 
        intfn = ''
        for ix in w_id:
            qmask = qmasks[ix]
            # figure out the image extent for the relevant quadrant
            imextent_q, intfn = lookup_meta_quadrant(qmask.scan_id, qmask.frame_num, qmask.quadrant, WISE)
            # fullmask[some indices] = qmask.omask
            x0,x1,y0,y1 = imextent_q
            fullmask[y0:y1+1, x0:x1+1] = qmask.omask

        # construct file name
        ofn = intfn.replace('-int', '')
        ofn = os.path.join(maskdir, 'unwise-mask-' + ti.coadd_id + '-'
                           + os.path.basename(ofn) + ('.gz' if not int_gz else ''))
        assert(not os.path.exists(ofn))

        fitsio.write(ofn, fullmask)
        print 'Wrote quadrant-based mask', (i+1), 'of', len(expid_u), ':', ofn

def lookup_meta_quadrant(scan_id, frame_num, quad_num, WISE):
    # meant to be helper for merge_write_quadrant_masks function above
    row = WISE[(WISE.scan_id == scan_id) & (WISE.frame_num == frame_num)][0]

    intfn = (row.intfn).replace(' ','')

    if quad_num == 1:
        imextent = row.imextent_q1
    elif quad_num == 2:
        imextent = row.imextent_q2
    elif quad_num == 3:
        imextent = row.imextent_q3
    elif quad_num == 4:
        imextent = row.imextent_q4

    return imextent, intfn

class RecoveryStats():
    def __init__(self, n_attempted, n_succeeded, n_skipped):
        self.n_attempted = n_attempted
        self.n_succeeded = n_succeeded
        self.n_skipped = n_skipped

    def to_recarray(self):
        dummy = (self.n_attempted is None)
        nrow = (0 if dummy else 1)

        arr_out = np.zeros((nrow,), 
                           dtype=[('n_attempted','int'),
                                  ('n_succeeded','int'),
                                  ('n_skipped','int')])
        if dummy:
            return arr_out

        arr_out['n_attempted'][0] = self.n_attempted
        arr_out['n_succeeded'][0] = self.n_succeeded
        arr_out['n_skipped'][0] = self.n_skipped

        return arr_out

    def _print(self):
        print 'number of warps attempted: ' + str(self.n_attempted)
        print 'number of warps succeeded: ' + str(self.n_succeeded)
        print 'number of warps skipped: ' + str(self.n_skipped)

def pad_rebin_weighted(images, mask, binfac=2):
    # pad a set of images so that each can be rebinned by integer binfac 
    # then do the rebinning; images input is a list of 2d numpy arrays

    # mask should be just zeros and ones

    # only binfac = 2 tested so far

    sh = images[0].shape

    spill_x = sh[1] % binfac
    spill_y = sh[0] % binfac

    pad_x = ((binfac-spill_x) if spill_x else 0)
    pad_y = ((binfac-spill_y) if spill_y else 0)

    sh_pad = (sh[0] + pad_y, sh[1] + pad_x)
    sh_reb = (sh_pad[0] // binfac, sh_pad[1] // binfac)

    if pad_x or pad_y:
    # pad the mask
        mask_pad = np.zeros(sh_pad, dtype=float) # want float during averaging
        mask_pad[0:sh[0], 0:sh[1]] = mask
    else:
        mask_pad = mask

    # calculate rebinned weight using the mask
    mask_reb = _rebin(mask_pad, sh_reb)

    # loop over the images

    images_out = []
    for im in images:
        if pad_x or pad_y:
            im_pad = np.zeros(sh_pad)
            im_pad[0:sh[0], 0:sh[1]] = im
        else:
            im_pad = im
    #     rebin image*(padded mask)
        im_reb = _rebin(im_pad*mask_pad, sh_reb)
    #     divide by (rebinned_weight + (rebinned_weight == 0))
        im_reb = im_reb / (mask_reb + (mask_reb == 0).astype('int'))
        images_out.append(im_reb) # slow ?
    
    return images_out, mask_reb

def reference_image_from_dir(basedir, coadd_id, band, verbose=True):
    # basedir is base directory, e.g. '$SCRATCH/unwise-coadds'
    # portion of $SCRATCH/unwise-coadds/000/0000p000/unwise-0000p000-w1-img-u.fits
    # could have problems if/when certain full-depth outputs get gzipped

    dir = get_dir_for_coadd(basedir, coadd_id)
    intfn = os.path.join(dir, 'unwise-' + coadd_id + '-w' + str(band) + '-img-u.fits')
    uncfn = intfn.replace('-img-u.fits', '-std-u.fits')
    nfn = intfn.replace('-img-u.fits', '-n-u.fits')

    if verbose:
        print 'Creating reference image from files: '
        print intfn
        print uncfn
        print nfn

    image = fitsio.read(intfn)
    std = fitsio.read(uncfn)
    n = fitsio.read(nfn)
    ref = ReferenceImage(image, std, n)
    return ref
