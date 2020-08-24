#!/bin/bash
cd /lustre03/project/6004866/msadikov/programmes/phonon_projections/scripts/tmp
# OpenMp Environment
export OMP_NUM_THREADS=1
mpirun  -n 1 anaddb < /lustre03/project/6004866/msadikov/programmes/phonon_projections/scripts/tmp/run.files > /lustre03/project/6004866/msadikov/programmes/phonon_projections/scripts/tmp/run.log 2> /lustre03/project/6004866/msadikov/programmes/phonon_projections/scripts/tmp/run.err
