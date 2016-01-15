import os
import fitsio
import sys
from unwise_coadd import wdirs
import time
import numpy as np
from unwise_utils import phase_from_scanid, get_l1b_file
import optparse

def check_l1b_files(indstart, nproc, band, int_gz=False):
    tab = fitsio.read(os.path.join(os.environ.get('UNWISE_META_DIR'), 'WISE-index-L1b.fits'),ex=1)
    print len(tab)
    tab = tab[tab['band'] == band]
    print len(tab)
    if indstart >= len(tab):
        print 'NO FILES TO CHECK !!!!!!!!'

    nmax = len(tab) - indstart # fence-posting ?

    nproc = min(nmax, nproc)
    # only check file existence, don't try to check for corruption

    dl_good = np.zeros(nproc, dtype=bool)
    t0 = time.time()
    for i in range(indstart, indstart+nproc):
        phase = phase_from_scanid(tab['scan_id'][i])
        basedir = wdirs[phase]
        int_gz = (phase == 'neo2')
        intfn = get_l1b_file(basedir, tab['scan_id'][i], tab['frame_num'][i], band, 
                             int_gz=int_gz)
        if (i % 1000) == 0:
            print i, '  ', intfn
        if not os.path.exists(intfn):
             print 'int missing !!!!!', intfn
             dl_good[i-indstart] = False
             continue
        if not os.path.exists(intfn.replace('-int-', '-msk-') + ('.gz' if not int_gz else '')):
             print 'msk missing !!!!!', intfn
             dl_good[i-indstart] = False
             continue
        if not os.path.exists(intfn.replace('-int-', '-unc-') + ('.gz' if not int_gz else '')):
             print 'unc missing !!!!!', intfn
             dl_good[i-indstart] = False
             continue
        dl_good[i-indstart] = True
    dt = time.time() - t0
    print dt, ' seconds'
    print np.sum(dl_good), len(dl_good)

    arr_out = np.zeros((len(dl_good),), dtype=[('scan_id','a6'), ('frame_num','int'), ('dl_good','int'), ('band', 'int')])
    arr_out['dl_good'] = dl_good.astype(int)
    arr_out['scan_id'] = tab['scan_id'][indstart:(indstart+nproc)]
    arr_out['frame_num'] = tab['frame_num'][indstart:(indstart+nproc)]
    arr_out['band'] = band
    outname = 'dl_good_w' + str(band) + '_' + str(indstart).zfill(8) + '.fits'
    print 'writing output : ' + outname
    fitsio.write(outname, arr_out)

def check_10k(ind, band):
    # check for the output file's existence
    # if it's there, return
    chunksize = 10000
    indstart = ind*chunksize
    outname = 'dl_good_w' + str(band) + '_' + str(indstart).zfill(8) + '.fits'
    if os.path.exists(outname):
        return False
    
    check_l1b_files(indstart, chunksize, band, int_gz=False)
    return True

def check_10k_many(band, delay=3):
    # loop over chunks of 10k framesets

    nchunk = 600 # be more careful about this
    for i in range(nchunk):
        print 'working on chunk ' + str(i) + ' of ' + str(nchunk)
        processed = check_10k(i, band)
        if processed:
            time.sleep(delay)

def main():
    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('--band', dest='band', default=None, type=int,
                      help='WISE band, should be integer, 1 or 2')
    # could add another parser option for delay

    opt,args = parser.parse_args()
    check_10k_many(opt.band)


if __name__ == '__main__':
    sys.exit(main())
