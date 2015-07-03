import os
import sys
import optparse

parser = optparse.OptionParser('%prog [options] dir [dir, ...]')

parser.add_option('-b', dest='bands', action='append', type=int, default=[],
                  help='Add band (default: 1234)')

opt,args = parser.parse_args()

if len(opt.bands) == 0:
    opt.bands = [1,2,3,4]
# Allow specifying bands like "123"
bb = []
for band in opt.bands:
    for s in str(band):
        bb.append(int(s))
opt.bands = bb
print 'Bands', opt.bands

if len(args) == 0:
    print 'No WISE directories specified.'
    parser.print_help()
    sys.exit(-1)

os.chdir('/clusterfs/riemann/raid000/bosswork/boss/wise_frames')
for a in args:
    for b in opt.bands:
        cmd = ('wget -r -N -nH -np -nv --cut-dirs=4 -A "*w%i*" "http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p1bm_frm/%s/"' %
               (band, a))
        print cmd
        os.system(cmd)
