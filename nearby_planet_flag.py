import numpy as np
import ephem
import fitsio
import os
import numpy.lib.recfunctions as recfunctions
from astrometry.util.starutil_numpy import degrees_between
import matplotlib.pyplot as plt

#read in some other file with list of planet sighting (ra,dec) ????? what file

#define radius within which to flag images as "neaby" to a bright planet -- 
# think it should be at least 2.1 deg, maybe somewhat more for padding

#for each planet
#    downselect index-l1b file to only those rows w/ (ra,dec) within a certain 
#    angular radius of frames that have planet sighting
#           ---> this will help to minimize the number of positions pyephem 
#                needs to compute
#    compute this planet's separation from each of the potentially relevant 
#    frames
#    save a table of (scan_id, frame_num, ang_sep, mjd) for ang_sep < thresh 
#    (scan_id, frame_num for matching to metadata structures later on, mjd for 
#    debugging)

def _zfill3(string):
    return string.zfill(3)

def get_all_unique_framesets():
    # read in the full L1b frame index and return a version
    # with only the unique (scan_id, frame_num) pairs
    fdir = '/project/projectdirs/cosmo/data/wise/merge/merge_p1bm_frm'
    fname = 'WISE-index-L1b.fits'
    fname = os.path.join(fdir, fname)

    tab = fitsio.read(fname, ext=1, header=False)

    # construct padded string array for frame_num,i think padding is actually
    # not necessary but oh well
    
    vfunc = np.vectorize(_zfill3)

    framenum_str_not_padded = (tab['FRAME_NUM']).astype(str)
    framenum_str_padded = vfunc(framenum_str_not_padded)

    #concatenate padded frame_num strings with scan_id strings
    #np.core.defchararray.add
    framesets = np.core.defchararray.add(tab['SCAN_ID'], framenum_str_padded)
    
    # do numpy.unique, grabbing the unique indices as well
    _, ind_u = np.unique(framesets, return_index=True)

    # return full table at unique indices ? think i should be able to drop
    # band at least, and also scan group

    tab = tab[ind_u]
    # should sort this by declination

    ind_dec_sort = np.argsort(tab['DEC'])
    return tab[ind_dec_sort]

def get_planet_framesets():
    # get the list of framesets where the planet falls *inside* of the FOV
    # this should be a short list (how short though?)

    dir = os.environ['UNWISE_META_DIR']
    fnames = ['WISE-l1b-metadata-4band.fits',
              'WISE-l1b-metadata-3band.fits',
              'WISE-l1b-metadata-2band.fits',
              'WISE-l1b-metadata-neowiser.fits']

    fnames.reverse()
    arr_out = None
    for fname in fnames:
        fname = os.path.join(dir, fname)
        print 'Reading : ' + fname
        tab = fitsio.read(fname, ext=1, header=False)

        tab = recfunctions.append_fields(tab, '_FRAME_NUM', 
                       data=(tab['FRAME_NUM']).astype('int64'),dtypes='>i8')
        tab = tab[['SCAN_ID', '_FRAME_NUM', 'RA', 'DEC', 'PLANETS']]
        print tab.dtype
        mask = (tab['PLANETS'] != 0)
        if np.sum(mask):
            tab = tab[mask]
            if arr_out is None:
                arr_out = tab
            else:
                arr_out = np.append(arr_out, tab)

    arr_out.dtype.names = 'SCAN_ID', 'FRAME_NUM', 'RA', 'DEC', 'PLANETS'
    return arr_out

def compute_separations_nearby():
    # for those frames within 5 degrees of any planet sighting, compute
    # the actual angular separation
    all_frames = get_all_unique_framesets()
    planet_frames = get_planet_framesets()

    n_planet_sightings = len(planet_frames)
    n_frames_all = len(all_frames)

    m = ephem.Mars()
    j = ephem.Jupiter()
    s = ephem.Saturn()

    deg_per_rad = 57.2957763672

    decpad = 5 # 3.5
    nearby_frames = None
    for i in range(n_planet_sightings):
        # use dec binary search (search_sorted?) to figure out the
        # relatively small number of frames that could be nearby this planet
        # sighting
        deccen = planet_frames['DEC'][i]
        dec_l = deccen - decpad
        dec_u = deccen + decpad
        indices = np.searchsorted(all_frames['DEC'], [dec_l, dec_u])
        print indices, dec_l, dec_u
        n_expected = np.sum((all_frames['DEC'] > dec_l) & (all_frames['DEC'] < dec_u))
        # pad out the indices from binary search just to be safe
        ind_l = indices[0] # max(indices[0] - 100, 0)
        ind_u = indices[1] # min(indices[1] + 100, n_frames_all)

        subframes = all_frames[ind_l:ind_u]
        print len(subframes), n_expected

        assert(n_expected == len(subframes))
        print 'number of difference angles to compute ' + str(ind_u-ind_l)
        # use dustin's astrometry.net difference angle code to compute
        # difference angles
        dangle = degrees_between(planet_frames['RA'][i],
                                 planet_frames['DEC'][i],
                                 subframes['RA'], subframes['DEC'])
        print str(len(dangle)) + ', ' + str(np.sum(dangle < 5)) + ' !!!!!!!!!!!!!!!!!'

        subframes = subframes[dangle < 5] # 4 deg
        print subframes.dtype, len(subframes)

        # initialize numpy array to hold a flag marking which exposures
        # are actually nearby (< 2.5 deg away from) the planet
        subframes_planet_flag = np.zeros(len(subframes), dtype='int')

        #    figure out the relevant planet (assume just one for now)
        planet_flag_val = planet_frames['PLANETS'][i]
        if planet_flag_val == 1:
            p = m # mars
            name = 'mars'
        elif planet_flag_val == 2:
            p = j # jupiter
            name = 'jupiter'
        elif planet_flag_val == 4:
            p = s # saturn
            name = 'saturn'
        offs = 15019.5 # days
        plot = False

        for k in range(len(subframes)):
        #    compute astrometric geocentric ra,dec of relevant planet
        #    at this row's MJD
            date = ephem.Date(subframes['MJD'][k]-offs)
            p.compute(date)
            #print date, subframes['MJD'][k]
            dd = degrees_between(p.a_ra*deg_per_rad, p.a_dec*deg_per_rad, subframes['RA'][k], 
                                 subframes['DEC'][k])
            if dd < 2.5:
                subframes_planet_flag[k] = planet_frames['PLANETS'][i]
        # need to do another structure appending type of deal to save
        # the relevant rows of all_frames table

        if plot:
            plt.scatter(subframes['RA'], subframes['DEC'],edgecolor='none',s=10)
            plt.scatter(planet_frames['RA'][i], planet_frames['DEC'][i], s=50, c='k', edgecolor='none')
            plt.scatter(subframes['RA'][subframes_planet_flag != 0], 
                        subframes['DEC'][subframes_planet_flag != 0], s=10, c='r', edgecolor='none')
            plt.show()
        print np.sum(subframes_planet_flag != 0), '  ', np.unique(subframes_planet_flag), '   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'

        subframes = recfunctions.append_fields(subframes, 'PLANETS', 
                                               data=subframes_planet_flag,dtypes='>i8')
        if nearby_frames is None:
            nearby_frames = subframes[subframes_planet_flag != 0]
        else:
            nearby_frames = np.append(nearby_frames, subframes[subframes_planet_flag != 0])

    return nearby_frames

def merge_nearby_frames(nearby_frames):
    # create a summary table of the output from compute_separations_nearby that has one row per
    # unique (scan_id, frame_num) pair, assume never more than one nearby planet
    

    vfunc = np.vectorize(_zfill3)

    framenum_str_not_padded = (nearby_frames['FRAME_NUM']).astype(str)
    framenum_str_padded = vfunc(framenum_str_not_padded)

    framesets = np.core.defchararray.add(nearby_frames['SCAN_ID'], framenum_str_padded)

    _, ind_u = np.unique(framesets, return_index=True)
    #nearby_frames_u = nearby_frames[ind_u]

    nearby_frames_u = nearby_frames[ind_u]
    #n_u = len(nearby_frames_u)

    #planet_flags_u = np.zeros(n_u, dtype='int8')
    #for i in range(n_u):
    #    imask = ((nearby_frames['SCAN_ID'] == nearby_frames_u['SCAN_ID'][i]) & (nearby_frames['FRAME_NUM'] == nearby_frames_u['FRAME_NUM'][i]))
    #    planet_flags_all = np.unique(nearby_frames['PLANETS'][imask])
    #    planet_flags_u[i] = np.sum(planet_flags_all).astype('int8')

    #return planet_flags_u
    return nearby_frames_u

def check_nearby_frames(nearby_frames_u):
    # check that list of frames with nearby planets is indeed a superset of
    # the list of frames which contain planets within the FOV
    planet_frames = get_planet_framesets()

    n_planet_fov = len(planet_frames)
    for i in range(n_planet_fov):
        imask = (planet_frames['SCAN_ID'][i] == nearby_frames_u['SCAN_ID']) & (planet_frames['FRAME_NUM'][i] == nearby_frames_u['FRAME_NUM'])
        assert(np.sum(imask) == 1)
        assert(nearby_frames_u[imask]['PLANETS'][0] == planet_frames['PLANETS'][i])
        #assert(np.sum(imask) == 1)

def write_nearby_frames(outname='nearby_planet_frames.fits'):
    # write FITS tables listing the framesets
    # these will be used to update the metadata tables with an extra column
    # containing a "nearby planet" flag

    nearby_frames = compute_separations_nearby()
    nearby_frames_u = merge_nearby_frames(nearby_frames)

    # bother to sort this in any way ?
    fitsio.write(outname, nearby_frames_u)
