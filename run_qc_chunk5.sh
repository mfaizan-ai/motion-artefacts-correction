#!/bin/bash
#SBATCH --job-name=qc_chunk5
#SBATCH --output=logs/qc_chunk5_%j.out
#SBATCH --error=logs/qc_chunk5_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

mkdir -p logs

source ~/.bashrc
conda activate moco

set -euo pipefail

echo "=========================================="
echo "  QC ANALYSIS — CHUNK-5 DATASET"
echo "  Job started: $(date)"
echo "  Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

PROJECT_DIR="/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction"
DATASET_DIR="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/cyclegans_chunk5_dataset"
METADATA_DIR="${DATASET_DIR}/metadata"
OUTPUT_BASE="${PROJECT_DIR}/dataset_quality_check/chunk5"

cd "${PROJECT_DIR}"

for SPLIT in train val test; do
    echo ""
    echo "--- Running QC for split: ${SPLIT} ---"
    python -u useful_scripts/dataset_qc_check.py \
        --metadata_dir "${METADATA_DIR}" \
        --dataset_dir  "${DATASET_DIR}" \
        --split        "${SPLIT}" \
        --output_dir   "${OUTPUT_BASE}/${SPLIT}_qc" \
        --n_jobs       8
done

echo ""
echo "=========================================="
echo "  DONE — $(date)"
echo "  Output: ${OUTPUT_BASE}"
echo "=========================================="
