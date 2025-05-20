#! /bin/bash


export UNWISE_SYMLINK_DIR=/global/cfs/cdirs/cosmo/work/wise/etc/etc_neo11
export UNWISE_META_DIR=$UNWISE_SYMLINK_DIR

python unwise_coadd.py --tile 0000p000 --band 1

