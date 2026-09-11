#!/bin/bash
#SBATCH --job-name=test_st_v3
#SBATCH --output=logs/test_st_v3_%j.out
#SBATCH --error=logs/test_st_v3_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs

source ~/.bashrc
conda activate moco

# 1,996 held-out test-split chunks (all grades) through model.correct(),
# same metrics pipeline as training validation (train.py's compute_fmri_metrics).
# st_v3 checkpoint, for comparison against st_v4's test results.
python test.py \
    --checkpoint_path /lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction/runs/st_v3_ddp_with_roi_time_series_cycleGANS/best_model.pt \
    --output_dir /lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction/runs/st_v3_ddp_with_roi_time_series_cycleGANS/test
