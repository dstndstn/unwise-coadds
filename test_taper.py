import pyfits
import zp_lookup
import matplotlib.pyplot as plt
import numpy as np

def test_taper(band, phase, poly_row):

    band_str = str(band)
    fname = 'etc/zp_poly_coeff_w' + band_str + '.fits'

    par = zp_lookup.ZPMetaParameters(phase)

    hdus = pyfits.open(fname)
    tab = hdus[par.exten].data

    tw = zp_lookup.TaperWeight(tab['cutoff_low'][poly_row], 
                               tab['cutoff_hi'][poly_row],
                               phase,
                               is_first=tab['is_first'][poly_row],
                               is_last=tab['is_last'][poly_row])
    return tw

def test_w1(phase='3band', poly_row=0):
    tw = test_taper(phase, poly_row)

    dt = 0.1
    if phase == '3band':
        mjd_test = np.arange(55400, 55480, dt)
    if phase == '2band':
        mjd_test = np.arange(55460, 55605, dt)
    if phase == 'neowiser':
        mjd_test = np.arange(56630, 57015, dt)

    nmjd = len(mjd_test)
    weight_vals = []
    for i in range(nmjd):
        weight_val = tw.compute_weight(mjd_test[i])
        weight_vals.append(weight_val)

    weight_vals = np.array(weight_vals)
    print np.min(weight_vals), np.max(weight_vals)

    plt.plot(mjd_test, np.array(weight_vals))
    plt.ylim((-0.1, 1.1))
    plt.show()

def test_zp(band, phase):
    dt = 0.1
    if phase == '3band':
        mjd_test = np.arange(55400, 55480, dt)
    if phase == '2band':
        mjd_test = np.arange(55460, 55605, dt)
    if phase == 'neowiser':
        mjd_test = np.arange(56630, 57015, dt)

    zp_lookup_obj = zp_lookup.PiecewisePolynomialInterpolator(band, phase)

    old_obj = zp_lookup.create_zp_interpolator(band, phase)

    nmjd = len(mjd_test)
    zp_vals = []
    zp_vals_old = []
    for i in range(nmjd):
        zp_vals.append(zp_lookup_obj.compute_zp(mjd_test[i]))
        zp_vals_old.append(old_obj(mjd_test[i]))

    zp_vals = np.array(zp_vals)
    zp_vals_old = np.array(zp_vals_old)

    plt.plot(mjd_test, zp_vals)
    plt.plot(mjd_test, zp_vals_old)
    plt.show()
