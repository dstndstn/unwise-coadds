import matplotlib
matplotlib.use('Agg')
import pylab as plt

import os
import datetime

from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.plotutils import *
from astrometry.util.starutil_numpy import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *

from unwise_coadd import *

# linear interp
from scipy.interpolate import interp1d

'''
http://minorplanetcenter.net/iau/MPEph/MPEph.html

text2fits.py cerberus.txt cerberus.fits -f jjjsddfffffff
'''

ps = PlotSequence('cerberus')

TT = []
for bands in [2,3,4]:
    T = fits_table('wise-metadata-%iband-b.fits' % bands)
    T.bands = np.zeros(len(T), np.uint8) + bands
    print bands, 'band: MJD', T.mjd.min(), T.mjd.max(), 'N frames', len(T)
    TT.append(T)
T = merge_tables(TT)

# 4band: 55203.07308 55414.439617

print len(T), 'frames'
print 'MJD', T.mjd.min(), T.mjd.max()

#E = fits_table('cerberus.fits')
E = fits_table('pluto.fits')
# Och, you're kidding me, decimal HOURS?!
E.ra *= 15.
E.mjd = np.array([datetomjd(datetime.datetime(yr, mo, day, 0,0,0))
                  for yr,mo,day in zip(E.year, E.month, E.day)])
print 'E mjd', E.mjd.min(), E.mjd.max()

plt.clf()
plt.plot(E.ra[0], E.dec[0], 'bo')
plt.plot(E.ra, E.dec, 'b-')
ps.savefig()

I,J,d = match_radec(T.ra, T.dec, E.ra, E.dec, 1.)
T.cut(I)
print len(T), 'near mover'

EE = E[np.unique(J)]
print len(EE), 'ephemeris times'

plt.clf()
plt.plot(T.ra, T.dec, 'r.')
plt.plot(EE.ra, EE.dec, 'b-')
ps.savefig()

mI = []
mJ = []

plt.clf()
plt.subplot(1,2,1)
plt.plot(T.mjd, T.ra, 'r.')
plt.plot(EE.mjd, EE.ra, 'b.')
plt.xlabel('MJD')
plt.ylabel('RA')
plt.subplot(1,2,2)
plt.plot(T.mjd, T.dec, 'r.')
plt.plot(EE.mjd, EE.dec, 'b.')
plt.xlabel('MJD')
plt.ylabel('Dec')
ps.savefig()

II = set()
for j,(r,d,mjd) in enumerate(zip(EE.ra, EE.dec, EE.mjd)):
    I,J,d = match_radec(T.ra, T.dec, r, d, 1.)
    if len(I) == 0:
        continue
    K = np.flatnonzero(np.abs(T.mjd[I] - mjd) < 1.)
    if len(K) == 0:
        continue
    I = I[K]
    II.update(I)
    mI.append(I)
    mJ.append([j]*len(I))
II = np.array(list(II))
mI = np.hstack(mI)
mJ = np.hstack(mJ)
print 'Total of', len(II), 'unique frames'
print 'And', len(np.unique(mJ)), 'ephemeris points'

plt.clf()
plt.plot(T.ra[II], T.dec[II], 'r.')
plt.plot(EE.ra[mJ], EE.dec[mJ], 'b.')
ps.savefig()


mjd0 = 55200

plt.clf()
plt.subplot(1,2,1)
plt.plot(T.mjd[mI] -mjd0, T.ra[mI], 'r.')
plt.plot(EE.mjd[mJ]-mjd0, EE.ra[mJ], 'b.')
plt.xlabel('MJD - %i' % mjd0)
plt.ylabel('RA')
plt.subplot(1,2,2)
plt.plot(T.mjd[mI] -mjd0, T.dec[mI], 'r.')
plt.plot(EE.mjd[mJ]-mjd0, EE.dec[mJ], 'b.')
plt.xlabel('MJD')
plt.ylabel('Dec')
ps.savefig()

# First crossing ~ MJD0 + 180
#I = np.flatnonzero(np.abs(EE.mjd[mJ] - (mjd0 + 180)) < 20)

# Pluto
I = np.flatnonzero(np.abs(EE.mjd[mJ] - (mjd0 + 75)) < 20)

# Second crossing ~ MJD0 + 305
#I = np.flatnonzero(np.abs(EE.mjd[mJ] - (mjd0 + 305)) < 5)

mJ1 = mJ[I]
J = np.unique(mJ1)
print len(J), 'ephemeris points in meeting #1'

I1,J1,d = match_radec(T.ra[II], T.dec[II], EE.ra[J], EE.dec[J], 1.)
UI1 = np.unique(II[I1])
print len(UI1), 'frames near meeting #1'

Era  = interp1d(EE.mjd, EE.ra )
Edec = interp1d(EE.mjd, EE.dec)

ii = []
rr,dd = [],[]
for i,t in zip(UI1, T[UI1]):
    sz = 1016
    wcs = Tan(t.w1crval1, t.w1crval2, t.w1crpix1, t.w1crpix2,
              t.w1cd1_1, t.w1cd1_2, t.w1cd2_1, t.w1cd2_2,
              sz, sz)
    print 'MJD', t.mjd, type(t.mjd)
    mjd = float(t.mjd)
    ra  = Era (mjd)
    dec = Edec(mjd)
    ra  = float(ra )
    dec = float(dec)
    print 'RA,Dec at frame MJD:', ra, dec, type(ra),type(dec)
    #ok,x,y = wcs.radec2pixelxy(ra, dec)
    if wcs.is_inside(ra, dec):
        ii.append(i)
        rr.append(ra)
        dd.append(dec)
        
ii = np.array(ii)
rr = np.array(rr)
dd = np.array(dd)
print len(ii), 'frames actually contain the ephemeris'


if len(ii) > 100:
    I = np.random.permutation(len(ii))[:100]
    ii = ii[I]
    rr = rr[I]
    dd = dd[I]
    print 'Cut to random subset of', len(ii)

    print 'MJD range:', T.mjd[ii].min(), T.mjd[ii].max()
    print 'RA range', rr.min(), rr.max()
    print 'Dec range', dd.min(), dd.max()
    
pixscale = 2.75
# Cerberus
# ra = (345. + 348.5) / 2.
# dec = (46.0 + 46.4) / 2.
# dra  = 2.0
# ddec = 0.4

# Pluto
ra =  275.5
dec = -18.3
dra  = 0.2
ddec = 0.2

W = int(np.cos(np.deg2rad(dec)) * dra * 3600. / pixscale)
H = int(ddec*3600./pixscale)

targetwcs = Tan(ra, dec, W/2., H/2.,
                -pixscale/3600., 0., 0., pixscale/3600., W, H)
print 'Target WCS:', targetwcs


ps.skipto(5)

#plt.figure(figsize=(18,5.5))
plt.figure(figsize=(W/100., H/100.+0.5))
plt.subplots_adjust(left=0, right=1, bottom=0, top=1.-0.5/(H/100.+0.5))

for band in [1,2,3,4]:
    
    coadd = np.zeros((H,W), np.float32)
    comax = np.zeros((H,W), np.float32)
    coaddsq = np.zeros((H,W), np.float32)
    con   = np.zeros((H,W), np.uint8)
    
    for t in T[ii]:
        wisedir = 'wise-frames'
        fn = get_l1b_file(wisedir, t.scan_id, t.frame_num, band)
        print 'filename', fn
        if not os.path.exists(fn):
            cmd = ('wget -r -N -nH -np -nv --cut-dirs=4 -A "*w%i*" "http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p1bm_frm/%s/"' % (band, os.path.dirname(fn).replace(wisedir+'/','')))
            print cmd
            os.system(cmd)
        wcs = Sip(fn)
        img = fitsio.read(fn)
        #print 'Image', img.shape
    
        maskfn = fn.replace('-int-', '-msk-')
        maskfn = maskfn + '.gz'
        mask = fitsio.read(maskfn)
        #print 'Mask', mask.dtype
    
        uncfn = fn.replace('-int-', '-unc-')
        uncfn = uncfn + '.gz'
        unc = fitsio.read(uncfn)
        
        badbits = [0,1,2,3,4,5,6,7, 9, 
                   10,11,12,13,14,15,16,17,18,
                   21,26,27,28]
        maskbits = sum([1<<bit for bit in badbits])
        goodmask = ((mask & maskbits) == 0)
        goodmask[unc == 0] = False
        goodmask[np.logical_not(np.isfinite(img))] = False
        goodmask[np.logical_not(np.isfinite(unc))] = False
    
        sig1 = median_f(unc[goodmask])
        #print 'sig1:', sig1
        
        if band in [3,4]:
            medfilt = 50
            mf = np.zeros_like(img)
            ok = median_smooth(img, np.logical_not(goodmask), int(medfilt), mf)
            img -= mf
    
        # add some noise to smooth out "dynacal" artifacts
        # (only to the 'image' used to estimate sky level!)
        fim = img[goodmask]
        fim += np.random.normal(scale=sig1, size=fim.shape)
        sky = estimate_sky_2(fim)
        #print 'Estimated sky:', sky
        
        try:
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(targetwcs, wcs, [], 3)
        except OverlapError:
            print 'OverlapError'
            continue
        #print len(Yo), 'pixels overlap'
    
        I = np.flatnonzero(goodmask[Yi,Xi])
        #print len(I), 'unmasked'
        Yo = Yo[I]
        Xo = Xo[I]
        Yi = Yi[I]
        Xi = Xi[I]
        
        coadd[Yo,Xo] += (img[Yi,Xi] - sky)
        coaddsq[Yo,Xo] += (img[Yi,Xi] - sky)**2
        comax[Yo,Xo]  = np.maximum(comax[Yo,Xo], img[Yi,Xi] - sky)
        con  [Yo,Xo] += 1.
    coadd /= np.maximum(con, 1)
    costd = np.maximum(0., coaddsq - coadd**2)
    
    mn,mx = [np.percentile(coadd[np.isfinite(coadd)], p) for p in [25,99]]
    print 'Coadd range', mn,mx
    
    plt.clf()
    dimshow(coadd, vmin=mn, vmax=mx)
    plt.title('W%i' % band)
    ps.savefig()
    ax = plt.axis()
    ok,xx,yy = targetwcs.radec2pixelxy(rr, dd)
    plt.plot(xx-1, yy-1, 'ro', mec='r', mfc='none')
    plt.axis(ax)
    ps.savefig()
    
    im = comax - coadd
    imn,imx = [np.percentile(im[np.isfinite(im)], p) for p in [25,99]]
    plt.clf()
    dimshow(comax - coadd, vmin=imn, vmax=imx)
    plt.title('W%i max' % band)
    ps.savefig()
    ax = plt.axis()
    ok,xx,yy = targetwcs.radec2pixelxy(rr, dd)
    plt.plot(xx-1, yy-1, 'ro', mec='r', mfc='none')
    plt.axis(ax)
    ps.savefig()
    
    
    # im = costd
    # imn,imx = [np.percentile(im[np.isfinite(im)], p) for p in [25,99]]
    # plt.clf()
    # dimshow(im, vmin=imn, vmax=imx)
    # ps.savefig()
    # std = np.median(costd)
    # im = costd / np.hypot(np.sqrt(np.maximum(0., coadd)), std)
    # imn,imx = [np.percentile(im[np.isfinite(im)], p) for p in [25,99]]
    # plt.clf()
    # dimshow(im, vmin=imn, vmax=imx)
    # ps.savefig()
    # ax = plt.axis()
    # ok,xx,yy = targetwcs.radec2pixelxy(rr, dd)
    # plt.plot(xx, yy, 'ro', mec='r', mfc='none')
    # plt.axis(ax)
    # ps.savefig()
