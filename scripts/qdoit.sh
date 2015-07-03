#! /bin/bash

# carver
module load subversion/1.7.2
which svn
svn --version
module list

export PATH=/usr/common/usg/subversion/1.7.2/bin:${PATH}
which svn
svn --version

cd ${HOME}/unwise
while [ $# -gt 0 ]; do
  n="$1"
  echo "Running $n"
  python -u unwise_coadd.py --outdir data/unwise-4 --dataset allsky --maxmem 7 $n > data/logs-unwise-4/$n.log 2> data/logs-unwise-4/$n.err
  shift
done


