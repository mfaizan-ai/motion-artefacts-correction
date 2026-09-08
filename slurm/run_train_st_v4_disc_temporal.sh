#!/bin/bash
#SBATCH --job-name=st_v4_disc_temporal
#SBATCH --output=logs/st_v4_disc_temporal_%j.out
#SBATCH --error=logs/st_v4_disc_temporal_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

mkdir -p logs

source ~/.bashrc
conda activate moco

export WANDB_MODE=offline

CKPT_ROOT=/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction/runs
GRADE_CHUNK_METADATA_CSV=/lustre/disk/home/users/mfaizan/motion_correction/cycleGANS_on_2d_images/pytorch-CycleGAN-and-pix2pix/data_preprocessing/motion_grades_chunk_5_dataset_hfiltered/chunk_metadata.csv
GRADE_RUN_STATS_CSV=/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction/run_normalization_stats_hfiltered.csv

NPROC=4

# Identical config to st_v3_ddp_with_roi_time_series_cycleGANS, except the image
# discriminator (STMultiScaleDiscriminator) now has TemporalConv1D enabled on its
# last (coarsest, cheapest) FactorizedDiscBlock -- see models/st_model.py
# _STScaleCNN. D_A/D_B previously scored each timepoint fully independently and
# only averaged scores across T at the end; now the deepest block mixes
# neighbouring timepoints' features before that average.
torchrun \
    --standalone \
    --nproc_per_node=${NPROC} \
    train.py \
    --use_grade_dataset \
    --grade_chunk_metadata_csv ${GRADE_CHUNK_METADATA_CSV} \
    --grade_run_stats_csv ${GRADE_RUN_STATS_CSV} \
    --in_timepoints 5 \
    --epochs 100 \
    --batch_size 4 \
    --num_workers 8 \
    --run_name st_v4_ddp_disc_temporal_roi \
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
    --use_st_model \
    --use_roi_discriminator \
    --lambda_roi 0.5 \
    --w_roi_adv 1.0 \
    --w_roi_cycle 1.0
