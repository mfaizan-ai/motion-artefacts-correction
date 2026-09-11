#!/bin/bash
#SBATCH --job-name=denoise_runs
#SBATCH --output=logs/denoise_runs_%j.out
#SBATCH --error=logs/denoise_runs_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs

source ~/.bashrc
conda activate moco

# Whole-run correction for the 128-subject raw-baseline population, using
# st_v4_ddp_disc_temporal_roi's best_model.pt. Output mirrors the source
# directory structure so create_roi_timeseries.py/denoising_evaluation.py
# can be re-run against it (before/after comparison).
python denoise_runs.py
