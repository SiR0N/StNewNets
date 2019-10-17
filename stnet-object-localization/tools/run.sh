#!/bin/bash
###SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH --time=07:30:00      # 4 hours
#SBATCH --mem-per-cpu=40000   # 4G of memory
#SBATCH --cpus-per-task=4

srun python stnet_object_localization.py --exe-mode bbox_eval --dataset-dir /scratch/work/paredej1/stnet-object-localization/dataset/
