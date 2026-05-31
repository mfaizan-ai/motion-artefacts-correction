#!/bin/bash
#SBATCH --job-name=build_cycleGAN_data
#SBATCH --output=logs/build_cycleGAN_data_%j.out
#SBATCH --error=logs/build_cycleGAN_data_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu

# set -euo pipefail

# Create log directory before SLURM writes runtime logs if running manually.
# Note: SLURM itself needs the logs/ directory to already exist before sbatch submission.
mkdir -p logs
mkdir -p dataset_build_logs

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "Working directory: $(pwd)"
echo "=========================================="

# Load shell configuration and activate conda environment
source ~/.bashrc
conda activate moco

# Input CSV files
CORRUPTED_CSV="dataset/preprocessed/motion_corrupted_chunk_dataset_with_preprocessed.csv"
MOTION_FREE_VIDEO_CSV="dataset/preprocessed/video_state_clean_chunk_dataset_with_preprocessed.csv"
MOTION_FREE_REST_CSV="dataset/preprocessed/resting_state_clean_chunk_dataset_with_preprocessed.csv"

# Output directory
OUTPUT_DIR="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/cyclegans_dataset"

# Optional: check that input files exist before running
for file in "$CORRUPTED_CSV" "$MOTION_FREE_VIDEO_CSV" "$MOTION_FREE_REST_CSV"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Input file not found: $file"
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

echo "Running CycleGAN dataset build script..."
echo "Corrupted CSV: $CORRUPTED_CSV"
echo "Motion-free video CSV: $MOTION_FREE_VIDEO_CSV"
echo "Motion-free rest CSV: $MOTION_FREE_REST_CSV"
echo "Output directory: $OUTPUT_DIR"

python -u build_cycle_gans_dataset.py \
    --corrupted_csv "$CORRUPTED_CSV" \
    --motion_free_video_csv "$MOTION_FREE_VIDEO_CSV" \
    --motion_free_rest_csv "$MOTION_FREE_REST_CSV" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \

echo "=========================================="
echo "Job finished on: $(date)"
echo "=========================================="