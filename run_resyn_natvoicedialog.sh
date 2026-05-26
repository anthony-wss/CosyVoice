#!/bin/bash
#SBATCH --job-name=resyn_dialogue
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

export SINGULARITYENV_WORKER_ID=$SLURM_JOB_ID

singularity exec --nv \
    /livingrooms/public/singularity‐images/default_20220325_1a.sif \
    /home/anthony/.local/bin/uv run resyn_natvoicedialog.py