import fitsio
import numpy as np
from unwise_coadd import get_wise_frames
from unwise_utils import get_epoch_breaks

def process_one_tile(racen, deccen, band):
    # call get_wise_frames
    frames = get_wise_frames(racen, deccen, band)
    # call get_epoch_breaks using the MJD values where qual_frame is good

    ebreaks = get_epoch_breaks(frames[frames.qual_frame > 0].mjd)

    # return the number of epochs
    n_epoch = len(ebreaks) + 1
    return n_epoch

def process_many_tiles(band, indstart, nproc):
    # read in atlas file
    tab = fitsio.read('/n/fink1/ameisner/unwise/allsky-atlas.fits')

    ntile = len(tab)
    # determine indend
    indend = min(indstart+nproc, ntile) # watch out for fence-posting

    tab = tab[indstart:indend]

    n_epoch_list = np.zeros(len(tab), dtype=int)

    # loop over tile id's
    for i in range(len(tab)):
    # call process_one_tile
        n_epoch = process_one_tile(tab['ra'][i], tab['dec'][i], band)
        # print tile name and number of epochs so that they can be included in log file
        print tab['coadd_id'][i], n_epoch
        n_epoch_list[i] = n_epoch
    
    # write out the per-tile number of epochs
    outname = 'num_epoch_'+str(indstart).zfill(5)+'_'+str(indend-1).zfill(5)+'.fits'
    fitsio.write(outname, n_epoch_list)
