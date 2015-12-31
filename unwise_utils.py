import os
import numpy as np
from astrometry.util.util import Tan

def get_l1b_file(basedir, scanid, frame, band, int_gz=False):
    scangrp = scanid[-2:]
    fname = os.path.join(basedir, scangrp, scanid, '%03i' % frame, 
                        '%s%03i-w%i-int-1b.fits' % (scanid, frame, band))
    if int_gz:
        fname += '.gz'
    return fname

def tile_to_radec(tileid):
    assert(len(tileid) == 8)
    ra = int(tileid[:4], 10) / 10.
    sign = -1 if tileid[4] == 'm' else 1
    dec = sign * int(tileid[5:], 10) / 10.
    return ra,dec

def int_from_scan_frame(scan_id, frame_num):
    val_str = scan_id[0:5] + str(frame_num).zfill(3)
    val = int(val_str)
    return val

# from tractor.basics.NanoMaggies
def zeropointToScale(zp):
    '''
    Converts a traditional magnitude zeropoint to a scale factor
    by which nanomaggies should be multiplied to produce image
    counts.
    '''
    return 10.**((zp - 22.5)/2.5)

def retrieve_git_version():
    from astrometry.util.run_command import run_command
    code_dir = os.path.dirname(os.path.realpath(__file__))
    cwd = os.getcwd()
    do_chdir = (cwd[0:len(code_dir)] != code_dir)
    if do_chdir:
        os.chdir(code_dir)
    rtn,version,err = run_command('git describe')
    if do_chdir:
        os.chdir(cwd)
    if rtn:
        raise RuntimeError('Failed to get version string (git describe):' + ver + err)
    version = version.strip()
    print '"git describe" version info:', version
    return version

def get_dir_for_coadd(outdir, coadd_id):
    # base/RRR/RRRRsDDD/unwise-*
    return os.path.join(outdir, coadd_id[:3], coadd_id)

def get_epoch_breaks(mjds):
    mjds = np.sort(mjds)

    # define an epoch either as a gap of more than 3 months
    # between frames, or as > 6 months since start of epoch.
    start = mjds[0]
    ebreaks = []
    for lastmjd,mjd in zip(mjds, mjds[1:]):
        if (mjd - lastmjd >= 90.) or (mjd - start >= 180.):
            ebreaks.append((mjd + lastmjd) / 2.)
            start = mjd
    print 'Defined epoch breaks', ebreaks
    print 'Found', len(ebreaks), 'epoch breaks'
    return ebreaks

def get_coadd_tile_wcs(ra, dec, W=2048, H=2048, pixscale=2.75):
    '''
    Returns a Tan WCS object at the given RA,Dec center, axis aligned, with the
    given pixel W,H and pixel scale in arcsec/pixel.
    '''
    cowcs = Tan(ra, dec, (W+1)/2., (H+1)/2.,
                -pixscale/3600., 0., 0., pixscale/3600., W, H)
    return cowcs

def _rebin(a, shape):
    # stolen from stackoverflow ...
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def download_frameset_1band(scan_id, frame_num, band):
    
    wdir = '_' # dummy

    intfn = get_l1b_file(wdir, scan_id, frame_num, band)
    intfnx = intfn.replace(wdir+'/', '')

    # Try to download the file from IRSA.
    cmd = (('(wget -r -N -nH -np -nv --cut-dirs=4 -A "*w%i*" ' +
            '"http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p1bm_frm/%s/")') %
           (band, os.path.dirname(intfnx)))
    print
    print 'Trying to download file:'
    print cmd
    print
    os.system(cmd)
    print

def planet_name2bit(_name):
    name = _name.lower()
    name2bit = {'mars' : 0,
                'jupiter' : 1,
                'saturn' : 2}
    return name2bit[name]

def is_fulldepth(parser):
    # based on input arguments to unwise_coadd.py, decide whether
    # the coadd being run is fulldepth or not

    fulldepth = True

    opt, args = parser.parse_args()

    if opt.epoch is not None:
        fulldepth = False
    if opt.before != parser.defaults['before']:
        fulldepth = False
    if opt.after != parser.defaults['after']:
        fulldepth = False
    if opt.frame0 != parser.defaults['frame0']:
        fulldepth = False
    if opt.nframes != parser.defaults['nframes']:
        fulldepth = False

    return fulldepth

def sanity_check_inputs(parser):
    # with the newly added polynomial warping options, it's possible
    # for the user to specify a set of inputs that don't really make sense
    # taken together -- this routine will halt execution if non-sane inputs
    # provided

    opt, args = parser.parse_args()

    if opt.warp_all or (opt.reference_dir is not None):
        assert((opt.reference_dir is not None) and opt.warp_all and (opt.epoch is not None))
        assert((opt.ra is None) and (opt.dec is None))
        assert((parser.defaults['after'] == opt.after) and (parser.defaults['before'] == opt.before))
        assert(opt.tile is not None)

        # TODO : still need to check whether specified reference_dir has necessary files

    fulldepth = is_fulldepth(parser)
    if opt.recover_warped and (not fulldepth):
        assert(opt.warp_all and (opt.reference_dir is not None))
        # if arrived here then reference_dir will already have been checked for necessary files
