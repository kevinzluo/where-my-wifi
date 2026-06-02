#!/bin/bash
#SBATCH --job-name=gp_grid
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=0

conda run -n jax-cuda12 python "gibbs_gs.py" $SLURM_ARRAY_TASK_ID
