from scipy.interpolate import interp1d
import pyfits
import os
import numpy as np

def create_zp_interpolator(band, phase):
    # construct relevant file name

    # phase should NOT be '4band', should be one of '3band', '2band', 'neowiser'

    fdir = os.environ['UNWISE_META_DIR']
    fname = os.path.join(fdir, 'zp_lookup_w' + str(band) + '.fits')

    hdus = pyfits.open(fname)

    exten_dict = {'3band'    : 1,
                  '2band'    : 2,
                  'neowiser' : 3}

    print 'Reading zero point lookup table for phase : ' + phase + ', W' + str(band)

    tab = hdus[exten_dict[phase]].data
    ntab = len(tab)
    mjds = tab['mjd']
    zps = tab['zp']

    zps_pad = np.zeros(ntab + 2)
    mjds_pad = np.zeros(ntab + 2)

    mjds_pad[0] = 0.
    mjds_pad[-1] = 100000. # arbitrary large number
    mjds_pad[1:-1] = mjds

    zps_pad[0] = zps[0]
    zps_pad[-1] = zps[-1]
    zps_pad[1:-1] = zps

    interp = interp1d(mjds_pad, zps_pad, kind='linear')
    return interp

def get_phase_mjd(mjd):
    if (mjd <= 55414.4396170):
        return '4band'
    elif (mjd <= 55468.777510):
        return '3band'
    elif (mjd <= 55593.460432):
         return '2band'
    else:
         return 'neowiser'

class ZPLookUp:
    """Look up W1/W2 zero point values as a function of time"""

    def __init__(self, band):
        self.zp_4band = {1 : 20.752, 2 : 19.596}
        self.band = band
        self.zp_3band = create_zp_interpolator(self.band, '3band')
        self.zp_2band = create_zp_interpolator(self.band, '2band')
        self.zp_neowiser = create_zp_interpolator(self.band, 'neowiser')

    def zp_interpolator_phase(self, phase):
        if (phase == '3band'):
            return self.zp_3band

        elif (phase == '2band'):
            return self.zp_2band
        else:
            return self.zp_neowiser

    def get_zp(self, mjd):
        # if MJD is during 4band phase, then just return the constant
        # 4band phase values
        phase = get_phase_mjd(mjd)
        if phase == '4band':
            return self.zp_4band[self.band]
        else:
            interp = self.zp_interpolator_phase(phase)
            zp = interp(mjd)
            return zp
