#!/bin/bash
#SBATCH --job-name=isc_correct_v4
#SBATCH --output=../logs/isc_correct_v4_%j.out
#SBATCH --error=../logs/isc_correct_v4_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p ../logs

source ~/.bashrc
conda activate moco

python motion_correct.py \
    --order A \
    --checkpoint ../runs/fix_v4/best_model.pt \
    --window_size 5 \
    --out_dir_name motion_correction_with_residual
