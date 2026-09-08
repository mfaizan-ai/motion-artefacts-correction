#!/bin/bash
#SBATCH --job-name=roi_ts_videos
#SBATCH --output=logs/roi_ts_videos_%j.out
#SBATCH --error=logs/roi_ts_videos_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

mkdir -p logs

source ~/.bashrc
conda activate moco

# CPU-only preprocessing (ROI-timeseries extraction + cosine high-pass filter),
# no GPU needed -- ~2s/run x 431 video runs, well under the 1h walltime.
python motion_denoising_measure/connectivity/create_roi_timeseries.py
