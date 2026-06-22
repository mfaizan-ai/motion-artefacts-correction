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
    --epochs 50 \
    --batch_size 4 \
    --num_workers 8 \
    --run_name fix_v4 \
    --max_grad_norm 3.0 \
    --w_cyc 5.0 \
    --w_idt 2.0 \
    --w_content 0.3 \
    --w_art 0.1 \
    --d_update_every 1 \
    --label_smooth_real 1.0 \
    --label_smooth_fake 0.0 \
    --r1_weight 0.5 \
    --r1_every 8 \
    --no_disc_temporal_diffs \
    --residual
