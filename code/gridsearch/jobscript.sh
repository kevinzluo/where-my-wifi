#!/bin/bash
#SBATCH --job-name=gp_grid
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=138,175,229,313,343,394,437,456,517,519,530

conda run -n jax-cuda12 python "gibbs_gs.py" $SLURM_ARRAY_TASK_ID
