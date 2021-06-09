import os
import numpy as np
from astrometry.util.util import Tan
import fitsio
from time_limit import time_limit, TimeoutException
import yaml
import pprint
from astrometry.util.starutil_numpy import degrees_between
import bisect

def good_scan_mask(frames):
    # given an input table of frames e.g. the variable WISE in 
    # unwise_coadd.py, make and return a row-matched boolean mask indicating
    # which scans are bad, based on the file $UNWISE_META_DIR/bad_scans.txt
    # the frames table must include a .scan_id column

    # do not assume any particular sorting of scan_id values listed in 
    # bad_scans.txt, or in the frames table
    # do assume one valid scan_id value per line in bad_scans.txt

    # what if bad_scans.txt is empty or non-existent ?

    # what does each of True/False mean in the output ?
    # --> True means good, False means bad

    fname = os.path.join(os.environ.get('UNWISE_META_DIR'), 'bad_scans.txt')

    # start with all true
    mask = np.logical_not(np.zeros(len(frames), dtype=bool))
    if not os.path.exists(fname):
        return mask

    f = open(fname, 'r')

    # this should be fine even if the file is empty
    for line in f:
        bad_scan_id = line[:-1]
        assert(len(bad_scan_id) == 6)
        mask = np.logical_and(mask, (frames.scan_id != bad_scan_id))

    return mask

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

def get_epoch_breaks(mjds, subdivide=False):
    mjds = np.sort(mjds)

    # define an epoch either as a gap of more than 3 months
    # between frames, or as > 6 months since start of epoch.
    start = mjds[0]
    ebreaks = []
    starts = [np.min(mjds)]
    ends = []
    for lastmjd,mjd in zip(mjds, mjds[1:]):
        if (mjd - lastmjd >= 90.) or (mjd - start >= 180.):
            ebreaks.append((mjd + lastmjd) / 2.)
            start = mjd
            
            starts.append(mjd)
            ends.append(lastmjd)
    ends.append(np.max(mjds))
    # meant to avoid having excessively long epochs near ecl poles
    if subdivide:
        new_breaks_all = []
        # figure out the length of each epoch
        mjd_bdy = [min(mjds)]
        mjd_bdy.extend(ebreaks)
        mjd_bdy.append(max(mjds))
        n_epoch = len(ebreaks)+1
        for i in range(n_epoch):
            mjd_min = starts[i]
            mjd_max = ends[i]
            dt = mjd_max-mjd_min
            if int(round(dt/10.0)) > 1:
                print 'excessively long epoch : ', dt, ' days'
            # create new epoch breaks that subdivide the 
                n_new_breaks = int(round(dt/10.0)) - 1
                new_dt = dt/(n_new_breaks + 1)
                new_breaks = [(mjd_min + new_dt*(j+1)) for j in range(n_new_breaks)]
                new_breaks_all.extend(new_breaks)
        for k in range(len(new_breaks_all)):
            bisect.insort_left(ebreaks, new_breaks_all[k])

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

def readfits_dodge_throttle(fname, nfail_max=20, tmax=0.15, header=False):
    success = 0

    nfail = 0
    while (success == 0):

        assert(nfail < nfail_max)

        try:
            with time_limit(tmax):
                out = fitsio.read(fname, header=header)
                success = 1
        except TimeoutException, msg:
            nfail += 1
            print "file read timed out!"

    return out

def phase_from_scanid(scan_id):
    # will need to be updated as NEOWISE-R year 2 and year 3 data become available
    # not vectorized
    scan_int = int(scan_id[0:5])

    scan_letter = scan_id[5]

    if (scan_int >= 88734):
        phase = 'neo5'
        return phase

    if (scan_letter == 'r'):
        if (scan_int > 12253):
            phase = 'neo7'
        elif (scan_int > 1089):
            phase = 'neo6'
        else:
            phase = 'neo5'
        return phase

    if (scan_letter == 's'):
        if scan_int > 12253:
            phase = 'neo7'
        else:
            phase = 'neo6'
        return phase
    
    if scan_int < 7101:
        phase = '4band'
    elif scan_int == 7101: 
        scan_letter = scan_id[5]
        if scan_letter == 'a': 
            phase = '4band'
        else:
            phase = '3band'
    elif scan_int <= 8744:
        phase = '3band'
    elif scan_int <= 12514:
        phase = '2band'
    elif scan_int <= 55289:
        phase = 'neo1'
    elif scan_int < 66418:
        phase = 'neo2'
    elif scan_int == 66418:
        scan_letter = scan_id[5]
        if scan_letter == 'a':
            phase = 'neo2'
        else: 
            phase = 'neo3'
    elif scan_int < 77590:
        phase = 'neo3'
    elif scan_int == 77590:
        scan_letter = scan_id[5]
        if scan_letter == 'a':
            phase = 'neo3'
        else:
            phase = 'neo4'
    else:
        phase = 'neo4'
    return phase

def header_reference_keywords(reference_dir):
    # FITS convention stupidity, this won't handle arbitrarily long 
    # reference_dir properly...
    if reference_dir is not None:
        _reference_dir = os.path.abspath(reference_dir)
        if len(_reference_dir) > 68:
            referen1 = _reference_dir[0:68]
            referen2 = _reference_dir[68:]
        else:
            referen1 = _reference_dir
            referen2 = ''
    else:
        referen1 = referen2 = ''

    return referen1, referen2

def get_l1b_dirs(yml=False, verbose=False):
    if not yml:
        wdirs = { '4band' : '/global/cfs/cdirs/cosmo/data/wise/allsky/4band_p1bm_frm', 
                  '3band' : '/global/cfs/cdirs/cosmo/data/wise/cryo_3band/3band_p1bm_frm', 
                  '2band' : '/global/cfs/cdirs/cosmo/data/wise/postcryo/2band_p1bm_frm',
                  'neo1' : '/global/cfs/cdirs/cosmo/data/wise/neowiser/p1bm_frm',
                  'neo2' : '/global/cfs/cdirs/cosmo/staging/wise/neowiser2/neowiser/p1bm_frm',
                  'neo3' : '/global/projecta/projectdirs/cosmo/staging/wise/neowiser/p1bm_frm',
                  'neo4' : '/global/cfs/cdirs/cosmo/staging/wise/neowiser4/neowiser/p1bm_frm', 
                  'neo5' : '/global/cfs/cdirs/cosmo/staging/wise/neowiser5/neowiser/p1bm_frm',
                  'neo6' : '/global/cfs/cdirs/cosmo/staging/wise/neowiser6/neowiser/p1bm_frm',
                  'neo7' : '/global/cfs/cdirs/cosmo/staging/wise/neowiser7/neowiser/p1bm_frm',
                  'missing' : 'merge_p1bm_frm' }
    else:
        fname = os.path.join(os.environ.get('UNWISE_META_DIR'), 'l1b_dirs.yml')
        print 'Reading L1b top-level directory locations from ' + fname
        wdirs = yaml.safe_load(open(fname))

    if verbose:
        print 'L1b top-level directories are: '
        pprint.PrettyPrinter().pprint(wdirs)

    return wdirs

def is_nearby(ra, dec, racen, deccen, margin, fast=True):
    if not fast:
        dangle = degrees_between(ra, dec, racen, deccen)
        return (dangle <= margin)
    else:
        # do a binary search
        ind = np.searchsorted(dec, [deccen-margin, deccen+margin])
        ind0 = (ind[0])[0]
        ind1 = (ind[1])[0]
        dangle = degrees_between(ra[ind0:ind1], dec[ind0:ind1], racen, deccen)
        nearby = np.zeros(len(ra),dtype=bool)
        nearby[ind0:ind1] = (dangle <= margin)
        return nearby

def ascending(inevents):
    # parse INEVENTS L1b header keyword to determine whether a scan is
    # ascending or descending

    # input needs to be a string !!
    if 'ASCE' in inevents:
        return 1
    if 'DESC' in inevents:
        return 0

    # have not considered the case in which both ASCE and DESC are
    # in INEVENTS -- code assumes that this situation never happens

    # 2 is a dummy value for the case in which neither ASCE nor DESC
    # is present in inevents
    return 2
