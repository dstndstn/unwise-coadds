import numpy as np
from astrometry.util.miscutils import patch_image

def wise_l1b_maskinterp(l1b_image, l1b_mask):
    # for now just interpolate of nan's

    goodmask = np.isfinite(l1b_image)
    _l1b_image = l1b_image.copy()
    ok = patch_image(_l1b_image, goodmask.copy())
    return _l1b_image
    
