import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import numpy as np
import pylab as plt

import os
from astrometry.util.fits import *
from astrometry.libkd.spherematch import *

#fn = os.path.join(os.environ['PHOTO_RESOLVE'], 'window_flist.fits')
# resolve-2010-05-23-cut: cut to PRIMARY fields.
fn = 'window_flist-cut.fits'
print 'window', fn
T = fits_table(fn)
print 'Read', len(T)
T.cut(T.rerun == '301')
print 'Cut to', len(T), 'rerun 301'
print 'image status (r):', np.unique(T.image_status[:,2])
print 'calib status (r):', np.unique(T.calib_status[:,2])
print 'score range:', T.score.min(), T.score.max()
print 'xbin, ybin:', np.unique(T.xbin), np.unique(T.ybin)
#T.cut(T.score >= 0.5)
#print 'Cut to', len(T), 'on score'

W = fits_table('wise-frames/wise_allsky_4band_p3as_cdd.fits')
print 'Read', len(W), 'Atlas tiles'

#r = ((2048 * 2.75)/3600. + np.hypot(13., 9.)/60.) / 2.
r = ((np.hypot(2048, 2048) * 2.75)/3600. + np.hypot(13.6, 9.9)/60.) / 2. + 0.1
print 'Matching radius:', r, 'degrees'

I,J,d = match_radec(T.ra, T.dec, W.ra, W.dec, r)
print len(I), 'matches'

JJ = np.unique(J)
print len(JJ), 'unique atlas tiles'

plt.clf()
plt.plot(W.ra, W.dec, 'b.')
plt.axis([360, 0, -30, 90])
plt.savefig('foot1.png')

W.cut(JJ)

I = match_radec(W.ra, W.dec, T.ra, T.dec, r, indexlist=True)
from astrometry.sdss.common import munu_to_radec_deg
from astrometry.util.miscutils import polygons_intersect
from unwise_coadd import get_coadd_tile_wcs
# Bit o' margin (pixels)
margin = 20.
lo,hi = 0.5 - margin, 2048.5 + margin
poly1 = np.array([[lo,lo], [lo,hi], [hi,hi], [hi,lo]])
poly2 = np.zeros((4,2))

keep = []
for i,ilist in enumerate(I):
    print 'Tile', i, W.coadd_id[i],
    if ilist is None:
        continue
    wcs = get_coadd_tile_wcs(W.ra[i], W.dec[i])

    ilist = np.array(ilist)
    ok,x,y = wcs.radec2pixelxy(T.ra[ilist], T.dec[ilist])
    # center of SDSS field within tile?
    if np.any(ok * (x >= lo) * (x <= hi) * (y >= lo) * (y <= hi)):
        print 'Center within tile'
        keep.append(i)
        continue

    gotone = False
    for ii in ilist:
        m0,m1 = T.mu_start[ii], T.mu_end[ii]
        n0,n1 = T.nu_start[ii], T.nu_end[ii]
        node,incl = T.node[ii], T.incl[ii]
        mu,nu = np.array([m0,m0,m1,m1]), np.array([n0,n1,n1,n0])
        r,d = munu_to_radec_deg(mu, nu, node, incl)
        ok,x,y = wcs.radec2pixelxy(r, d)
        print 'Field', T.run[ii], T.camcol[ii], T.field[ii], 'xy', x,y
        # corner within tile?
        if np.any((x >= lo) * (x <= hi) * (y >= lo) * (y <= hi)):
            print 'Corner within tile'
            gotone = True
            break
        poly2[:,0] = x
        poly2[:,1] = y
        if polygons_intersect(poly1, poly2):
            print 'Polygons intersect'
            gotone = True
            break
    if gotone:
        keep.append(i)
    else:
        print 'No intersection: tile', W.coadd_id[i]

W.cut(np.array(keep))
print 'Cut to', len(W)

I = np.lexsort((W.ra, W.dec))
W.cut(I)

A = fits_table()
A.ra  = W.ra
A.dec = W.dec
A.coadd_id = np.array([c.replace('_ab41','') for c in W.coadd_id])
A.writeto('sdss-atlas.fits')

plt.clf()
plt.plot(A.ra, A.dec, 'b.')
plt.axis([360, 0, -30, 90])
plt.savefig('foot2.png')

