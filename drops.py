from astrometry.util.fits import *
from astrometry.util.multiproc import *

from animate import create_animations

def _bounce_anims((args, kwargs)):
    try:
        create_animations(*args, **kwargs)
    except:
        import traceback
        print 'Exception processing:', args, kwargs
        traceback.print_exc()


mp = multiproc(4)

# fn = 'idrops.fits'
# htmlfn = 'idrops.html'
# dirnm = 'idrops'

fn = 'zdrops.fits'
htmlfn = 'zdrops.html'
dirnm = 'zdrops'

T = fits_table(fn)


#sys.exit(0)

args = []
for ra,dec in zip(T.ra, T.dec):
    args.append(((ra, dec, 100, 100),
                 dict(pixscale=0.4, yearly=True, diffim=False, outdir=dirnm)))
mp.map(_bounce_anims, args)


html = '<html><body><table>\n<tr><th>RA</th><th>Dec</th><th>Epoch 1</th><th>Epoch 2</th><th>Difference</th><th>Anim</th></tr>\n'
for ra,dec in zip(T.ra, T.dec):
    dataset_tag = '%04i%s%03i' % (int(ra*10.),
                                  'p' if dec >= 0. else 'm',
                                  int(np.abs(dec)*10.))
    html += '<tr>'
    html += '<td><a id="%s">%.4f</td><td>%.4f</td>' % (dataset_tag, ra,dec)
    html += '<td><img src="%s-y0.jpg"></td><td><img src="%s-y1.jpg"></td><td><img src="sdiff-%s-y0.jpg"></td>' % ((dataset_tag,)*3)
    html += '<td><a href="anim-%s.gif">gif</a></td></tr>\n' % dataset_tag

html += '</table></body></html>\n'
f = open(htmlfn, 'w')
f.write(html)
f.close()
print 'Wrote', htmlfn
