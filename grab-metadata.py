'''
Create WISE-l1b-metadata-neowiser2.fits file, and WISE-index-L1b fragment,
for NEOWISE-R year2 data for Michael Cushing in the Hyades region.

'''


from glob import glob
from astrometry.util.fits import *
import fitsio

#fns = glob('neowiser2-frames/0b/56970b/*/*-w2-int-1b.fits.gz')
fns = glob('neowiser2-frames/*/*/*/*-w2-int-1b.fits.gz')
fns.sort()
print len(fns), 'files'

T = fits_table()
T.ra  = []
T.dec = []
T.scan_id = []
T.scangrp = []
T.frame_num = []
T.band = []
T.mjd = []

for fn in fns:
    print 'Reading', fn
    hdr = fitsio.read_header(fn)
    # the L1b WCS puts the reference point in the center.

    ra  = hdr['CRVAL1']
    dec = hdr['CRVAL2']

    T.ra.append(ra)
    T.dec.append(dec)
    T.band.append(hdr['BAND'])
    T.scan_id.append(hdr['SCAN'].strip())
    T.scangrp.append(hdr['SCANGRP'].strip())
    T.frame_num.append(hdr['FRNUM'])
    T.mjd.append(hdr['MJD_OBS'])

T.to_np_arrays()

# FAKE!
T.moon_masked = np.array(['00'] * len(T))
T.w1intmedian = np.zeros(len(T), np.float32)
T.w2intmedian = np.zeros(len(T), np.float32)
T.w1intstddev = np.zeros(len(T), np.float32)
T.w2intstddev = np.zeros(len(T), np.float32)
T.w1intmed16ptile = np.zeros(len(T), np.float32)
T.w2intmed16ptile = np.zeros(len(T), np.float32)
T.qual_frame = np.zeros(len(T), np.int16) + 10
T.dtanneal = np.zeros(len(T), np.int32) + 1000000

fn = 'neo2/WISE-l1b-metadata-neowiser2.fits'
T.writeto(fn)
print 'Wrote', fn

W = fits_table('wise-frames/WISE-index-L1b.fits')
W = merge_tables([W, T], columns='fillzero')
fn = 'neo2/WISE-index-L1b+.fits'
W.writeto(fn, columns=['scan_id', 'frame_num', 'band', 'ra', 'dec', 'mjd', 'scangrp'])
print 'Wrote', fn

# WISE-index-L1b.fits:
#    1 SCAN_ID          6A
#    2 FRAME_NUM        I
#    3 BAND             I
#    4 RA               D
#    5 DEC              D
#    6 MJD              D
#    7 SCANGRP          2A



    




