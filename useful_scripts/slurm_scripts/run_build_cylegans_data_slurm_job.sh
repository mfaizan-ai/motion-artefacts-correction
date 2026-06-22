#!/bin/bash
#SBATCH --job-name=qc_train_analysis
#SBATCH --output=logs/qc_train_analysis_%j.out
#SBATCH --error=logs/qc_train_analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

# set -euo pipefail

mkdir -p logs

echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "Working directory: $(pwd)"
echo "=========================================="

source ~/.bashrc
conda activate moco

OUTPUT_DIR="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/cyclegans_dataset"
QC_OUT_DIR="qc_results/train_qc"

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "ERROR: Dataset directory not found: $OUTPUT_DIR"
    exit 1
fi

if [[ ! -d "$OUTPUT_DIR/metadata" ]]; then
    echo "ERROR: Metadata directory not found: $OUTPUT_DIR/metadata"
    exit 1
fi

for f in "$OUTPUT_DIR/metadata/train_corrupted_all.csv" \
         "$OUTPUT_DIR/metadata/train_motion_free.csv"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Required metadata CSV not found: $f"
        exit 1
    fi
done

mkdir -p "$QC_OUT_DIR"

echo "Running QC chunk analysis for TRAIN split..."
echo "Dataset dir  : $OUTPUT_DIR"
echo "QC output    : $QC_OUT_DIR"
echo "=========================================="

python -u dataset_qc_check.py \
    --metadata_dir "$OUTPUT_DIR/metadata" \
    --dataset_dir  "$OUTPUT_DIR" \
    --split        train \
    --output_dir   "$QC_OUT_DIR" \
    --n_jobs       8

echo "=========================================="
echo "Job finished on: $(date)"
echo "=========================================="