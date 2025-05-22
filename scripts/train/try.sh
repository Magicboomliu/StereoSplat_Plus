#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=23:59:00
#$ -p -5
#$ -N VSRD_TEST
#$ -m ae
#$ -M liuzihua1004@gmail.com


# load envs
module load cuda/11.8.0 cudnn/9.0.0 ffmpeg/6.1.1
module load nccl/2.20.5
module load intel-mpi/2021.11  openmpi/5.0.2-gcc 
module load forge/23.1.2  intel-vtune/2024.0


export APPTAINER_CACHEDIR=/gs/bs/tga-lab_okmn/zliu/apptainer/cache

apptainer exec \
  -B /gs/bs/tga-lab_okmn/zliu/zliu/:/gs \
  -B /home \
  --nv -w /gs/bs/tga-lab_okmn/zliu/apptainer/stereogs \
  bash train_kitti360_stereo.sh
