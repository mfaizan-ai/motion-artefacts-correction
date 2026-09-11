#!/bin/bash
#SBATCH --job-name=denoise_eval_raw
#SBATCH --output=logs/denoise_eval_raw_%j.out
#SBATCH --error=logs/denoise_eval_raw_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

mkdir -p logs

source ~/.bashrc
conda activate moco

# CPU-only (QC-FC, QC-FC-DD, modularity, DVARS, tSNR), no GPU needed.
# Full 128-subject run on the RAW (pre-denoising) roi_timeseries_hfiltered_videos
# data -- timed at ~20s/subject (modularity dominates) -> ~42 min total, 2h buffer.
python motion_denoising_measure/denoising_evaluation.py \
    --output_dir /lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/roi_timeseries_hfiltered_videos/denoising_eval_2mo_firstrun_raw \
    --run_label raw \
    --plot
