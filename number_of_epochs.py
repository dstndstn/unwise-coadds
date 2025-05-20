import fitsio
import numpy as np
from unwise_coadd import get_wise_frames
from unwise_utils import get_epoch_breaks
from astrometry.util.fits import fits_table
import matplotlib.pyplot as plt

def process_one_tile(racen, deccen, band, subdivide=False):
    # call get_wise_frames
    frames = get_wise_frames(racen, deccen, band)
    # call get_epoch_breaks using the MJD values where qual_frame is good

    ebreaks = get_epoch_breaks(frames[frames.qual_frame > 0].mjd, 
                               subdivide=subdivide)

    plot = False
    if plot:
        nmjd = len(frames[frames.qual_frame > 0].mjd)
        plt.subplot(2,1,1)
        plt.scatter(frames[frames.qual_frame > 0].mjd, np.zeros(nmjd), c='k', edgecolor='none')
        for i in range(len(ebreaks)):
            plt.plot([ebreaks[i], ebreaks[i]], [-0.5,0.5], c='r')

        plt.xlim((55200, 55600))
        plt.ylim((-0.5, 0.5))
        plt.yticks([])

        plt.title('number of epoch breaks = ' + str(len(ebreaks)))

        plt.subplot(2,1,2)
        plt.scatter(frames[frames.qual_frame > 0].mjd, np.zeros(nmjd), c='k', edgecolor='none')
        for i in range(len(ebreaks)):
            plt.plot([ebreaks[i], ebreaks[i]], [-0.5,0.5], c='r')

        plt.xlim((56625, 57025))
        plt.ylim((-0.5, 0.5))
        plt.yticks([])
        plt.xlabel('MJD')
        plt.show()

        plt.scatter(frames[frames.qual_frame > 0].mjd, np.zeros(nmjd), c='k', edgecolor='none')
        for i in range(len(ebreaks)):
            plt.plot([ebreaks[i], ebreaks[i]], [-0.5,0.5], c='r')

        plt.xlim((55200, 57025))
        plt.ylim((-0.5, 0.5))
        plt.yticks([])
        plt.xlabel('MJD')
        plt.title('number of epoch breaks = ' + str(len(ebreaks)))
        plt.show()

    # return the number of epochs
    n_epoch = len(ebreaks) + 1
    return n_epoch

def process_many_tiles(band, indstart, nproc, subdivide=False, fname='/n/fink1/ameisner/unwise/allsky-atlas.fits'):

    # fname is meant to be the name of the -atlas.fits file containing the
    # list of tiles that are of interest ... it defaults to the file with
    # all 18240 tiles

    # read in atlas file
    tab = fitsio.read(fname)

    ntile = len(tab)
    # determine indend
    indend = min(indstart+nproc, ntile) # watch out for fence-posting

    tab = tab[indstart:indend]

    n_epoch_list = np.zeros(len(tab), dtype=int)

    # loop over tile id's
    for i in range(len(tab)):
    # call process_one_tile
        n_epoch = process_one_tile(tab['RA'][i], tab['DEC'][i], band, subdivide=subdivide)
        # print tile name and number of epochs so that they can be included in log file
        print tab['COADD_ID'][i], n_epoch
        n_epoch_list[i] = n_epoch
    
    # write out the per-tile number of epochs
    outname = 'num_epoch_'+str(indstart).zfill(5)+'_'+str(indend-1).zfill(5)+'.fits'
    fitsio.write(outname, n_epoch_list)
