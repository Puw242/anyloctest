#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch

module purge

singularity exec --nv \
	    --overlay /scratch/yl9727/my_env/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python main.py --mode test --pooling seqnet --dataset nordland-sf --seqL 5 --split test --resume ./data/runs/seqnet_Mar11_02-02-13_l10_w5/ --outDims 4096" 