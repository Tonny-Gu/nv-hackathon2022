#!/bin/bash


#SBATCH --gpus-per-node=4
#SBATCH -n 4
#SBATCH --reservation=HACKATHON
#SBATCH --time=01:00:00


module load openmpi/4.0.3-cuda11.2.2 gcc/9.2.0 dl
module load nccl/2.10.3.1
module use /sw/csi/modulefiles/applications
module load arm-forge

ddt --connect
