from scipy.interpolate import interp1d
import pyfits
import os
import numpy as np
import scipy.special as ss

def create_zp_interpolator(band, phase):

    # phase should NOT be '4band'
    # should be one of '3band', '2band', 'neowiser', 'neowiser2', 'neowiser3', 'neowiser4', 'neowiser5', 'neowiser6', 'neowiser7', 'neowiser8', 'neowiser9', 'neowiser10'
    assert (phase == '3band') or (phase == '2band') or (phase == 'neowiser') or (phase == 'neowiser2') or (phase == 'neowiser3') or (phase == 'neowiser4') or (phase == 'neowiser5') or (phase == 'neowiser6') or (phase == 'neowiser7') or (phase == 'neowiser8') or (phase == 'neowiser9') or (phase == 'neowiser10')

    # construct relevant file name
    fdir = os.environ['UNWISE_META_DIR']
    fname = os.path.join(fdir, 'zp_lookup_w' + str(band) + '.fits')

    hdus = pyfits.open(fname)

    par = ZPMetaParameters(phase)
    print 'Reading zero point lookup table for phase : ' + phase + ', W' + str(band)

    tab = hdus[par.exten].data
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
    # should really absorb these special numbers somewhere...
    if (mjd <= 55414.4396170):
        return '4band'
    elif (mjd <= 55468.777510):
        return '3band'
    elif (mjd <= 55593.460432):
         return '2band'
    elif (mjd <= 57004.30997070):
         return 'neowiser'
    elif (mjd <= 57368.8109894100):
         return 'neowiser2'
    elif (mjd <= 57735.5000532063):
         return 'neowiser3'
    elif (mjd <= 58100.2976637800):
         return 'neowiser4'
    elif (mjd <= 58465.3333506900):
         return 'neowiser5'
    elif (mjd <= 58830.4934658700):
         return 'neowiser6'
    elif (mjd <= 59196.4970553900):
         return 'neowiser7'
    elif (mjd <= 59561.0493839840):
         return 'neowiser8'
    elif (mjd <= 59926.5081357310):
         return 'neowiser9'
    else:
         return 'neowiser10'

class ZPMetaParameters:
    def __init__(self, phase):
        exten_dict = {'3band' : 1,
                      '2band' : 2,
                      'neowiser' : 3,
                      'neowiser2' : 4,
                      'neowiser3' : 5,
                      'neowiser4' : 6,
                      'neowiser5' : 7,
                      'neowiser6' : 8,
                      'neowiser7' : 9,
                      'neowiser8' : 10,
                      'neowiser9' : 11,
                      'neowiser10' : 12}
        mjd_cen_dict = {'3band' : 55441,
                        '2band' : 55531,
                        'neowiser' : 56822,
                        'neowiser2' : 57187,
                        'neowiser3' : 57552,
                        'neowiser4' : 57918,
                        'neowiser5' : 58282,
                        'neowiser6' : 58647,
                        'neowiser7' : 59013,
                        'neowiser8' : 59378,
                        'neowiser9' : 59743,
                        'neowiser10' : 60108}
        zp_mjd_min_dict = {'3band' : 55414.9410170000,
                           '2band' : 55469.2786560000,
                           'neowiser' : 56640.2772121900,
                           'neowiser2' : 57004.8109894100,
                           'neowiser3' : 57369.9769432600,
                           'neowiser4' : 57736.0000532063,
                           'neowiser5' : 58103.7999558900,
                           'neowiser6' : 58468.8364068300,
                           'neowiser7' : 58834.0073457600,
                           'neowiser8' : 59199.9970553900,
                           'neowiser9' : 59564.5504547600,
                           'neowiser10' : 59930.050029600}
        zp_mjd_max_dict = {'3band' : 55467.9410170000,
                           '2band' : 55593.1195440000,
                           'neowiser' : 57004.0435914400,
                           'neowiser2' : 57368.8109894100,
                           'neowiser3' : 57734.9769432600,
                           'neowiser4' : 58099.8988816363,
                           'neowiser5' : 58460.7999558900,
                           'neowiser6' : 58825.8364068300,
                           'neowiser7' : 59191.0073457600,
                           'neowiser8' : 59556.9970553900,
                           'neowiser9' : 59921.5504547600,
                           'neowiser10' : 60287.050029600}
        self.exten = exten_dict[phase]
        self.mjd_cen = mjd_cen_dict[phase]
        # the nominal MJD of earliest per-day zero point tabulated
        self.zp_mjd_min = zp_mjd_min_dict[phase]
        # the nominal MJD of latest per-day zero point tabulated
        self.zp_mjd_max = zp_mjd_max_dict[phase]
        
class TaperWeight:
    """Polynomial tapering parameters"""

    def __init__(self, cutoff_low, cutoff_hi, phase, is_first=False, is_last=False):
        self.cutoff_low = cutoff_low
        self.cutoff_hi = cutoff_hi
        self.phase = phase
        self.center = (cutoff_low + cutoff_hi)/2.
        self.is_first = bool(is_first)
        self.is_last = bool(is_last)
        self.slope = 1./(self.center - self.cutoff_low)

    def compute_weight(self, mjd):
        # for now this will not be vectorized ...
        _mjd = max(min(mjd, self.cutoff_hi), self.cutoff_low)
        weight = min(1. - (self.center - _mjd)*self.slope , 1. - (_mjd - self.center)*self.slope)

        par = ZPMetaParameters(self.phase)

        if self.is_first:
            if _mjd <= par.zp_mjd_min:
                weight = 1.
        elif self.is_last:
            if _mjd >= par.zp_mjd_max:
                weight = 1.
        return weight

class TaperedPolynomial:
    """A set of polynomial coefficients and corresponding tapering parameters"""
    def __init__(self, coeff, taper_weight, phase):
        self.order = np.sum(np.isfinite(coeff)) - 1
        self.coeff = coeff[0:(self.order + 1)]
        self.taper_weight = taper_weight
        self.phase = phase

    def compute(self, mjd):
        # return the tapering weight and polynomial value as a tuple
        par = ZPMetaParameters(self.phase)

        _mjd = max(min(mjd, par.zp_mjd_max), par.zp_mjd_min)
        dt = _mjd - par.mjd_cen

        poly_val = 0.
        for i in range(0, self.order + 1):
            poly_val = poly_val + self.coeff[i]*(dt**i)

        weight = (self.taper_weight).compute_weight(_mjd)
        return poly_val, weight

class TaperedErf:
    """Error function parameters and corresponding tapering parameters"""
    def __init__(self, params, taper_weight, phase):
        self.mjd_cen = params[0]
        self.sigma = params[1]
        self.ampl = params[2]
        self.offs = params[3]
        self.taper_weight = taper_weight
        self.phase = phase

    def compute(self, mjd):
        # return the tapering weight and erf value as a tuple
        par = ZPMetaParameters(self.phase)

        _mjd = max(min(mjd, par.zp_mjd_max), par.zp_mjd_min)

        x = (_mjd - self.mjd_cen)/self.sigma
        erf_val = (self.ampl)*ss.erf(x) + self.offs

        weight = (self.taper_weight).compute_weight(_mjd)
        return erf_val, weight

def read_zp_poly(band, phase):
    # read in the appropriate table of piecewise polynomials
    fdir = os.environ['UNWISE_META_DIR']
    fname = os.path.join(fdir, 'zp_poly_coeff_w' + str(band) + '.fits')

    print 'Reading zero point polynomial coefficients for phase : ' + phase + ', W' + str(band)

    hdus = pyfits.open(fname)
    print fname + '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
    par = ZPMetaParameters(phase)
    tab = hdus[par.exten].data

    return tab
        

class PiecewisePolynomialInterpolator:
    """Custom interpolator using tapered piecewise polynomials"""
    # inputs should be band, phase
    def __init__(self, band, phase):
        self.band = band
        self.phase = phase
        self.poly_data = read_zp_poly(self.band, self.phase)
        self.npoly = len(self.poly_data)

    def compute_zp(self, mjd):
        # for each row in that table, use TaperedPolynomial to compute
        # polynomial value and corresponding weight
        weight_tot = 0.
        poly_tot = 0.

        for i in range(self.npoly):
            taper_weight = TaperWeight(self.poly_data['cutoff_low'][i], 
                                       self.poly_data['cutoff_hi'][i], 
                                       self.phase, 
                                       is_first=self.poly_data['is_first'][i],
                                       is_last=self.poly_data['is_last'][i])
            if self.poly_data['order'][i] == -1:
                t = TaperedErf(self.poly_data['coeff'][i], taper_weight, self.phase)
            else:
                t = TaperedPolynomial(self.poly_data['coeff'][i], taper_weight, self.phase)

            poly_val, weight = t.compute(mjd)
            poly_tot += poly_val*weight
            weight_tot += weight

        # calculate zero point as sum(weight*poly)/sum(weight)
        zp = poly_tot/weight_tot
        return zp

class ZPLookUp:
    """Look up W1/W2 zero point values as a function of time"""

    def __init__(self, band, poly=False):
        self.zp_4band = {1 : 20.752, 2 : 19.596}
        self.band = band
        self.poly = poly
        self.zp_3band = (create_zp_interpolator(self.band, '3band') if not self.poly else 
                         PiecewisePolynomialInterpolator(self.band, '3band'))
        self.zp_2band = (create_zp_interpolator(self.band, '2band') if not self.poly else 
                         PiecewisePolynomialInterpolator(self.band, '2band'))
        self.zp_neowiser = (create_zp_interpolator(self.band, 'neowiser') if not self.poly else 
                            PiecewisePolynomialInterpolator(self.band, 'neowiser'))
        self.zp_neowiser2 = (create_zp_interpolator(self.band, 'neowiser2') if not self.poly else 
                             PiecewisePolynomialInterpolator(self.band, 'neowiser2'))
        self.zp_neowiser3 = (create_zp_interpolator(self.band, 'neowiser3') if not self.poly else 
                             PiecewisePolynomialInterpolator(self.band, 'neowiser3'))
        self.zp_neowiser4 = (create_zp_interpolator(self.band, 'neowiser4') if not self.poly else 
                             PiecewisePolynomialInterpolator(self.band, 'neowiser4'))
        self.zp_neowiser5 = (create_zp_interpolator(self.band, 'neowiser5') if not self.poly else 
                             PiecewisePolynomialInterpolator(self.band, 'neowiser5'))
        self.zp_neowiser6 = (create_zp_interpolator(self.band, 'neowiser6') if not self.poly else 
                             PiecewisePolynomialInterpolator(self.band, 'neowiser6'))
        self.zp_neowiser7 = (create_zp_interpolator(self.band, 'neowiser7') if not self.poly else
                             PiecewisePolynomialInterpolator(self.band, 'neowiser7'))
        self.zp_neowiser8 = (create_zp_interpolator(self.band, 'neowiser8') if not self.poly else
                             PiecewisePolynomialInterpolator(self.band, 'neowiser8'))
        self.zp_neowiser9 = (create_zp_interpolator(self.band, 'neowiser9') if not self.poly else
                             PiecewisePolynomialInterpolator(self.band, 'neowiser9'))
        self.zp_neowiser10 = (create_zp_interpolator(self.band, 'neowiser10') if not self.poly else
                              PiecewisePolynomialInterpolator(self.band, 'neowiser10'))

    def zp_interpolator_phase(self, phase):
        if (phase == '3band'):
            return self.zp_3band
        elif (phase == '2band'):
            return self.zp_2band
        elif (phase == 'neowiser'):
            return self.zp_neowiser
        elif (phase == 'neowiser2'):
            return self.zp_neowiser2
        elif (phase == 'neowiser3'):
            return self.zp_neowiser3
        elif (phase == 'neowiser4'):
            return self.zp_neowiser4
        elif (phase == 'neowiser5'):
            return self.zp_neowiser5
        elif (phase == 'neowiser6'):
            return self.zp_neowiser6
        elif (phase == 'neowiser7'):
            return self.zp_neowiser7
        elif (phase == 'neowiser8'):
            return self.zp_neowiser8
        elif (phase == 'neowiser9'):
            return self.zp_neowiser9
        else:
            return self.zp_neowiser10

    def get_zp(self, mjd):
        # if MJD is during 4band phase, then just return the constant
        # 4band phase values
        phase = get_phase_mjd(mjd)
        if phase == '4band':
            return self.zp_4band[self.band]
        else:
            interp = self.zp_interpolator_phase(phase)
            if not self.poly:
                zp = interp(mjd)
            else:
                zp = interp.compute_zp(mjd)

            return zp
