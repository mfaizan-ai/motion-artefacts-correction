#!/bin/bash
#SBATCH --job-name=cyclegan_fc
#SBATCH --output=logs/cyclegan_fc_%j.out
#SBATCH --error=logs/cyclegan_fc_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs

source ~/.bashrc
conda activate moco

export WANDB_MODE=offline

DATA_ROOT=/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/cyclegans_chunk5_dataset
CKPT_ROOT=/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction/runs
VIDEO_DIR=${DATA_ROOT}/chunk5_subject_wise_order_video_data

python -u train.py \
    --data_root ${DATA_ROOT} \
    --in_timepoints 5 \
    --epochs 100 \
    --num_workers 8 \
    --run_name fc_v1 \
    --ckpt_root ${CKPT_ROOT} \
    --max_grad_norm 3.0 \
    --w_cyc 10.0 \
    --w_idt 5.0 \
    --d_update_every 1 \
    --label_smooth_real 1.0 \
    --label_smooth_fake 0.0 \
    --r1_weight 0.5 \
    --r1_every 8 \
    --num_disc_scales 2 \
    --residual \
    --use_sequences \
    --manifest_csv ${VIDEO_DIR}/video_sequence_manifest.csv \
    --chunk_metadata_csv ${VIDEO_DIR}/video_chunk_metadata_with_paths.csv \
    --w_temporal 0.0 \
    --w_fc 1.0 \
    --fc_mask_strategy threshold \
    --fc_threshold 0.3
