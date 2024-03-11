#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --time=40:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch

module purge

singularity exec --nv \
	    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python main.py --mode train --pooling al_seqnet --dataset nordland-sw --seqL 10 --w 5 --outDims 4096 --skip 10 --nEpoch 200 --expName "w5""