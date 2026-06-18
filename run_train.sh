#!/bin/bash
#SBATCH --job-name=cyclegan_train
#SBATCH --output=logs/cyclegan_train_%j.out
#SBATCH --error=logs/cyclegan_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs

source ~/.bashrc
conda activate moco

export WANDB_MODE=offline

python -u train.py \
    --epochs 300 \
    --batch_size 4 \
    --num_workers 8 \
    --run_name fix_v1