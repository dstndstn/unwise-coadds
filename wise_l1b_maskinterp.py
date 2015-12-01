import numpy as np
from astrometry.util.miscutils import patch_image
from scipy.ndimage.morphology import binary_dilation

def wise_l1b_maskinterp(l1b_image, l1b_mask, nointerp=False):
    # nointerp keyword meant to be a way to just return the mask

    # deal with cosmic rays
    kpix = 3
    kern = np.ones((kpix, kpix), dtype='int')
    crmask = (np.bitwise_and(l1b_mask, 2**28) != 0)
    crmask_dilate = binary_dilation(crmask, structure=kern).astype(crmask.dtype)

    # mask for saturated pixels
    satmask = (np.bitwise_and(l1b_mask, 523264) != 0)

    badmask  = ((np.bitwise_and(l1b_mask, 255) != 0) | crmask_dilate | satmask) 
    goodmask = np.isfinite(l1b_image)
    goodmask = (goodmask & (~badmask))

    if nointerp:
        return goodmask
    else:
    # interpolate over NaN mask, CR mask, saturated pixels and other bad pixels
        _l1b_image = l1b_image.copy()
        ok = patch_image(_l1b_image, goodmask.copy())
        return _l1b_image
    
