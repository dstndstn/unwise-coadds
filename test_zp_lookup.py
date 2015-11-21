from zp_lookup import ZPLookUp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mjds = np.arange(55200, 57000, 0.1)

print np.min(mjds),np.max(mjds)

band = 2
z = ZPLookUp(band)

nmjd = len(mjds)
zps = np.zeros(nmjd)

for i, mjd in enumerate(mjds):
    zps[i] = z.get_zp(mjd)

plt.plot(mjds, zps)

interp_3band = z.zp_3band
interp_2band = z.zp_2band
interp_neowiser = z.zp_neowiser

plt.scatter(interp_3band.x, interp_3band.y, c='k')
plt.scatter(interp_2band.x, interp_2band.y, c='k')
plt.scatter(interp_neowiser.x, interp_neowiser.y, c='k')

plt.xlim((np.min(mjds), np.max(mjds)))

formatter = mpl.ticker.ScalarFormatter(useOffset=False)
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)


plt.show()

