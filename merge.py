import os
import sys
import numpy as np

import fitsio

from astrometry.util.fits import *

def floatcompress(fn, truncbits):
    I,hdr = fitsio.read(fn, header=True)
    print I.shape, I.dtype
    assert(I.dtype == np.float32)
    F = np.zeros(I.shape, np.uint32)
    F.data[:] = I.data
    # IEEE -- ASSUME little-endian
    signbit = (1 << 31)
    expbits = (0xff << 23)
    manbits = 0x7fffff
    # print 'sign %08x' % signbit
    # print 'exp  %08x' % expbits
    # print 'man  %08x' % manbits
    # print '0x%x' % (signbit + expbits + manbits)
    signvals = (F & signbit)
    expvals  = (F & expbits)
    manvals  = (F & manbits)

    # mad = np.median(np.abs(I[:-5:10,:-5:10] - I[5::10,5::10]).ravel())
    # # convert to Gaussian -- sqrt(2) because our diffs are the
    # # differences of deviations of two pixels.
    # sig1 = 1.4826 * mad / np.sqrt(2.)
    # print 'sigma', sig1

    keepbits = np.uint32(0xffffffff - ((1 << truncbits)-1))
    #print 'keepbits', keepbits
    F[:] = (signvals | expvals | (manvals & keepbits))
    #approx = np.empty_like(I)
    #approx.data[:] = F.data
    I.data[:] = F.data
    approx = I
    #diff = I - approx
    # print 'Truncbits:', truncbits
    # print 'abs max diff:', np.max(np.abs(diff))
    # print 'mean abs diff:', np.mean(np.abs(diff))
    # print 'abs max relative diff:', np.max(np.abs(diff[I != 0] / I[I != 0]))
    # print 'mean abs diff:', np.mean(np.abs(diff[I != 0] / I[I != 0]))
    # print 'sigma:', sig1
    return approx, hdr



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start at row number...')
    parser.add_argument('-n', type=int, default=0, help='Run this many rows...')
    parser.add_argument('--dry-run', dest='dryrun', action='store_true')
    parser.add_argument('--skip-existing', dest='skip', action='store_true')
    parser.add_argument('-b', dest='bands', action='append', type=int, default=[],
                        help='Add WISE band (default: 1,2)')
    parser.add_argument('tiles', metavar='tile', type=str, nargs='+',
                        help='Individual tiles to run')
    opt = parser.parse_args()

    if len(opt.bands) == 0:
        opt.bands = [1,2,3,4]
    # Allow specifying bands like "123"
    bb = []
    for band in opt.bands:
        for s in str(band):
            bb.append(int(s))
    opt.bands = bb

    #indirs = ['data/unwise', 'data/unwise-nersc']
    #outdir = 'data/unwise-comp-4'
    #indirs = ['data/unwise',]
    indirs = ['data/unwise-4', 'data/unwise', 'data/unwise-nersc']
    outdir = 'data/unwise-comp'
    bands = opt.bands
    # (pattern, drop-bits, compress)
    pats = [('unwise-%s-w%i-img-m.fits', 0, False),
            ('unwise-%s-w%i-img-u.fits', 0, False),
            ('unwise-%s-w%i-invvar-m.fits', 11, True),
            ('unwise-%s-w%i-invvar-u.fits', 11, True),
            ('unwise-%s-w%i-std-m.fits', 11, True),
            ('unwise-%s-w%i-std-u.fits', 11, True),
            ('unwise-%s-w%i-n-m.fits', 0, True),
            ('unwise-%s-w%i-n-u.fits', 0, True),
            ('unwise-%s-w%i-frames.fits', 0, False),
            ]
    gzargs = ''

    if len(opt.tiles):
        tiles = opt.tiles
    else:
        T = fits_table('allsky-atlas.fits')
        T.cut(np.argsort(T.coadd_id))
        tiles = T.coadd_id

        if opt.start:
            T = T[opt.start:]
        if opt.n:
            T = T[:opt.n]

    for i,coadd in enumerate(tiles):
        for band in bands:
    
            print
            print 'Row', opt.start + i, 'coadd', coadd, 'band', band
            print
    
            outpath = os.path.join(outdir, coadd[:3], coadd)
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            # Skip inputs that don't exist
            exists = False
            fns = []
            for indir in indirs:
                fn = os.path.join(indir, coadd[:3], coadd, pats[0][0] % (coadd, band))
                if os.path.exists(fn):
                    exists = True
                    break
                fns.append(fn)
            if not exists:
                print 'Input file does not exist:', fns
                continue
    
            maskdir = 'unwise-%s-w%i-mask' % (coadd, band)
            for indir in indirs:
                fn = maskdir + '.tgz'
                fn = os.path.join(indir, coadd[:3], coadd, fn)
                print 'Checking', fn
                if os.path.exists(fn):
                    print 'exists'
                    break
                fn = maskdir
                fn = os.path.join(indir, coadd[:3], coadd, fn)
                print 'Checking', fn
                if os.path.exists(fn):
                    print 'exists'
                    break
            assert(os.path.exists(fn))
    
            outfn = os.path.join(outpath, maskdir + '.tgz')
            absoutfn = os.path.abspath(outfn)
            if not fn.endswith('.tgz'):
                # tar
                dirfn = os.path.dirname(fn)
                cmd = '(cd %s && tar czf %s %s)' % (dirfn, absoutfn, maskdir)
            else:
                # copy
                cmd = 'cp %s %s' % (fn, outfn)

            if opt.skip and os.path.exists(outfn):
                print 'Already exists:', outfn
                continue
            print cmd
            if not opt.dryrun:
                rtn = os.system(cmd)
                if rtn:
                    sys.exit(rtn)

            for pat, truncbits, compress in pats:
                for indir in indirs:
                    fn = os.path.join(indir, coadd[:3], coadd, pat % (coadd, band))
                    print 'Looking for', fn
                    if os.path.exists(fn):
                        break
                assert(os.path.exists(fn))
                outfn = os.path.join(outpath, pat % (coadd, band))
                print 'Writing to', outfn
                if truncbits:
                    data,hdr = floatcompress(fn, truncbits)
                    hdr.add_record(dict(name='UNW_TRNC', value=truncbits,
                                        comment='Floating-point bits truncated'))
                    if not opt.dryrun:
                        fitsio.write(outfn, data, header=hdr, clobber=True)
                    cmd = 'gzip -f %s %s' % (gzargs, outfn)
                elif compress:
                    cmd = 'gzip -f %s -c %s > %s.gz' % (gzargs, fn, outfn)
                else:
                    cmd = 'cp %s %s' % (fn, outfn)
                print cmd
                if not opt.dryrun:
                    rtn = os.system(cmd)
                    if rtn:
                        sys.exit(rtn)
