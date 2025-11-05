#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=23:59:00
#$ -p -5
#$ -N Volume3DBranchOnly
#$ -m ae
#$ -M liuzihua1004@gmail.com

export APPTAINER_CACHEDIR=/gs/bs/tga-lab_okmn/zliu/apptainer/cache

apptainer exec \
  -B /gs/bs/tga-lab_okmn/zliu/zliu/:/gs \
  -B /home \
  --nv -w /gs/bs/tga-lab_okmn/zliu/apptainer/stereogs \
  bash train_volume3d_branch_only.sh