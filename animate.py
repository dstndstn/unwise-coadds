import os
import sys

import matplotlib
matplotlib.use('Agg')
import pylab as plt

from astrometry.util.multiproc import multiproc
from astrometry.util.file import *

from unwise_coadd import *

def trymakedirs(fn, dir=False):
    if dir is True:
        dirnm = fn
    else:
        dirnm = os.path.dirname(fn)
    if not os.path.exists(dirnm):
        try:
            os.makedirs(dirnm)
        except:
            pass

def main():
    import optparse

    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('--threads', dest='threads', type=int, help='Multiproc',
                      default=1)

    parser.add_option('--outdir', '-o', dest='outdir', default='unwise-coadds',
                      help='Output directory: default %default')

    parser.add_option('--size', dest='size', default=2048, type=int,
                      help='Set output image size in pixels; default %default')
    parser.add_option('--width', dest='width', default=0, type=int,
                      help='Set output image width in pixels; default --size')
    parser.add_option('--height', dest='height', default=0, type=int,
                      help='Set output image height in pixels; default --size')

    parser.add_option('--pixscale', dest='pixscale', type=float, default=2.75,
                      help='Set coadd pixel scale, default %default arcsec/pixel')
    #parser.add_option('--cube', dest='cube', action='store_true',
    #                  default=False, help='Save & write out image cube')
    parser.add_option('--ra', dest='ra', type=float, default=None,
                      help='Build coadd at given RA center')
    parser.add_option('--dec', dest='dec', type=float, default=None,
                      help='Build coadd at given Dec center')
    parser.add_option('-b', '--band', dest='bands', action='append', type=int, default=[],
                      help='Add WISE band (default: 1,2)')

    parser.add_option('--year', action='store_true', default=False)

    opt,args = parser.parse_args()

    mp = multiproc(opt.threads)

    radec = opt.ra is not None and opt.dec is not None
    if not radec:
        print 'Must specify --ra and --dec'
        sys.exit(-1)

    W = H = opt.size
    if opt.width:
        W = opt.width
    if opt.height:
        H = opt.height

    if len(opt.bands) == 0:
        opt.bands = [1,2]
    # Allow specifying bands like "123"
    bb = []
    for band in opt.bands:
        for s in str(band):
            bb.append(int(s))
    opt.bands = bb
    print 'Bands', opt.bands

    create_animations(opt.ra, opt.dec, W, H, pixscale=opt.pixscale,
                      bands=opt.bands, yearly=opt.year, mp=mp, outdir=opt.outdir)


def create_animations(ra, dec, W, H, pixscale=2.75, bands=[1,2],
                      yearly=False,
                      diffim=True,
                      sdiffim=True,
                      mp=None,
                      outdir='.',
                      ):

    if mp is None:
        mp = multiproc(1)

    dataset_tag = '%04i%s%03i' % (int(ra*10.),
                                      'p' if dec >= 0. else 'm',
                                      int(np.abs(dec)*10.))
    dataset = ('custom-%s' % dataset_tag)
    print 'Setting custom dataset', dataset
    cosd = np.cos(np.deg2rad(dec))
    r0 = ra - (pixscale * W/2.)/3600. / cosd
    r1 = ra + (pixscale * W/2.)/3600. / cosd
    d0 = dec - (pixscale * H/2.)/3600.
    d1 = dec + (pixscale * H/2.)/3600.

    WISE = get_wise_frames_for_dataset(dataset, r0,r1,d0,d1)

    T = fits_table()
    T.coadd_id = np.array([dataset])
    T.ra = np.array([ra])
    T.dec = np.array([dec])
    tile = T[0]

    ps = None
    randomize = False
    force = False
    medfilt = False
    dsky = False
    bgmatch = False
    cube = False
    cube1 = False
    rchi_fraction = 0.01

    if yearly:
        ebreaks = [56000]
        etag = 'y%i'
    else:
        ebreaks = get_epoch_breaks(WISE.mjd)
        print len(ebreaks), 'epoch breaks'
        etag = 'e%i'

    ebreaks = [0] + ebreaks + [1000000]

    gifs = []
    eims = []

    for ei,(elo,ehi) in enumerate(zip(ebreaks, ebreaks[1:])):
        ims = []
        for band in bands:
            WI = WISE[(WISE.mjd >= elo) * (WISE.mjd < ehi) * (WISE.band == band)]
            if len(WI) == 0:
                ims.append(None)
                continue
            out = os.path.join(outdir, etag % ei)
            outfn = os.path.join(out, 'unwise-%s-w%i-img-u.fits' % (dataset, band))
            if not os.path.exists(outfn):
                trymakedirs(out, dir=True)
                print 'Band', band, 'Epoch', ei
                if one_coadd(tile, band, W, H, pixscale, WI, ps,
                             False, out, mp, mp,
                             cube, False, None, None, None, force,
                             medfilt, 0, dsky, False, bgmatch,
                             False, False,  
                             rchi_fraction, cube1,
                             None, None, None, force_outdir=True):
                    os.unlink(out)
                    break
            print 'read', outfn
            ims.append(fitsio.read(outfn))
            print ims[-1].shape, ims[-1].min(), ims[-1].max()
        eims.append(ims)

        if bands == [1,2]:
            w1,w2 = ims
            if w1 is None or w2 is None:
                continue
            lo,hi = -10.,100.

            w1 = (w1 - lo) / (hi - lo)
            w2 = (w2 - lo) / (hi - lo)
            assert(w1.shape == w2.shape)
            h,w = w1.shape
            rgb = np.zeros((h,w,3), np.float32)
            rgb[:,:,0] = w2
            rgb[:,:,1] = (w1+w2)/2.
            rgb[:,:,2] = w1
            rgb = np.round(np.clip(rgb, 0., 1.)*255).astype(np.uint8)
            fn = os.path.join(outdir, '%s-%s.jpg' % (dataset_tag, etag % ei))
            plt.imsave(fn, rgb, origin='lower')
            print 'Wrote', fn
            giffn = os.path.join(outdir, '%s-%s.gif' % (dataset_tag, etag % ei))
            cmd = 'jpegtopnm %s | pnmquant 256 | ppmtogif > %s' % (fn, giffn)
            print cmd
            os.system(cmd)
            gifs.append(giffn)

    anim = os.path.join(outdir, 'anim-%s.gif' % dataset_tag)
    cmd = 'gifsicle -o %s -d 50 -l %s' % (anim, ' '.join(gifs))
    print cmd
    os.system(cmd)

    if diffim:
        # Difference image vs coadd of all images
        allims = []
        out = os.path.join(outdir, 'all')
        trymakedirs(out, dir=True)
        for band in bands:
            outfn = os.path.join(out, 'unwise-%s-w%i-img-u.fits' % (dataset, band))
            if not os.path.exists(outfn):
                print 'Band', band, 'Epoch', ei
                one_coadd(tile, band, W, H, pixscale, WISE, ps,
                          False, out, mp, mp,
                          cube, False, None, None, None, force,
                          medfilt, 0, dsky, False, bgmatch,
                          False, False,  
                          rchi_fraction, cube1,
                          None, None, None, force_outdir=True)
            print 'read', outfn
            allims.append(fitsio.read(outfn))
            print allims[-1].shape, allims[-1].min(), allims[-1].max()
    
        gifs = []
    
        if bands == [1,2]:
            w1all,w2all = allims
            if w1all is None or w2all is None:
                return 0
    
            for ei,(w1,w2) in enumerate(eims):
                lo,hi = -50.,50.
    
                w1 = (w1 - w1all - lo) / (hi - lo)
                w2 = (w2 - w2all - lo) / (hi - lo)
                print 'diff range', w1.min(), w1.max(), w2.min(), w2.max()
                h,w = w1.shape
                rgb = np.zeros((h,w,3), np.float32)
                rgb[:,:,0] = w2
                rgb[:,:,1] = (w1+w2)/2.
                rgb[:,:,2] = w1
                rgb = np.round(np.clip(rgb, 0., 1.)*255).astype(np.uint8)
                fn = os.path.join(outdir, 'diff-%s-%s.jpg' % (dataset_tag, etag % ei))
                plt.imsave(fn, rgb, origin='lower')
                print 'Wrote', fn
                giffn = os.path.join(outdir, 'diff-%s-%s.gif' % (dataset_tag, etag % ei))
                cmd = 'jpegtopnm %s | pnmquant 256 | ppmtogif > %s' % (fn, giffn)
                print cmd
                os.system(cmd)
                gifs.append(giffn)
    
            anim = os.path.join(outdir, 'danim-%s.gif' % dataset_tag)
            cmd = 'gifsicle -o %s -d 50 -l %s' % (anim, ' '.join(gifs))
            print cmd
            os.system(cmd)
        
            # Relative difference images.
            for ei,(w1,w2) in enumerate(eims):
                sig1 = 30.
                w1 = (w1 - w1all) / np.maximum(sig1, w1all)
                w2 = (w2 - w2all) / np.maximum(sig1, w2all)
                lo,hi = -3.,3.
                w1 = (w1 - lo) / (hi - lo)
                w2 = (w2 - lo) / (hi - lo)
                print 'diff range', w1.min(), w1.max(), w2.min(), w2.max()
                h,w = w1.shape
                rgb = np.zeros((h,w,3), np.float32)
                rgb[:,:,0] = w2
                rgb[:,:,1] = (w1+w2)/2.
                rgb[:,:,2] = w1
                rgb = np.round(np.clip(rgb, 0., 1.)*255).astype(np.uint8)
                fn = os.path.join(outdir, 'reldiff-%s-%s.jpg' % (dataset_tag, etag % ei))
                plt.imsave(fn, rgb, origin='lower')
                print 'Wrote', fn


    if sdiffim:
        # Sequential difference images
        for ei,((w1z,w2z),(w1,w2)) in enumerate(zip(eims, eims[1:])):

            #lo,hi = -50.,50.
            #lo,hi = -100.,100.
            lo,hi = -200.,200.

            w1 = (w1 - w1z - lo) / (hi - lo)
            w2 = (w2 - w2z - lo) / (hi - lo)
            h,w = w1.shape
            rgb = np.zeros((h,w,3), np.float32)
            rgb[:,:,0] = w2
            rgb[:,:,1] = (w1+w2)/2.
            rgb[:,:,2] = w1
            rgb = np.round(np.clip(rgb, 0., 1.)*255).astype(np.uint8)
            fn = os.path.join(outdir, 'sdiff-%s-%s.jpg' % (dataset_tag, etag % ei))
            plt.imsave(fn, rgb, origin='lower')
            print 'Wrote', fn
            #giffn = os.path.join(outdir, 'diff-%s-%s.gif' % (etag % ei, dataset_tag))
            #cmd = 'jpegtopnm %s | pnmquant 256 | ppmtogif > %s' % (fn, giffn)
            #print cmd
            #os.system(cmd)
            #gifs.append(giffn)


    return            

if __name__ == '__main__':
    sys.exit(main())
