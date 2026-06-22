#!/bin/bash
#SBATCH --job-name=build_chunk5
#SBATCH --output=logs/build_chunk5_%j.out
#SBATCH --error=logs/build_chunk5_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs

source ~/.bashrc
conda activate moco

set -euo pipefail

echo "=========================================="
echo "  BUILD CHUNK-5 DATASET"
echo "  Job started: $(date)"
echo "  Job ID: ${SLURM_JOB_ID:-N/A}"
echo "=========================================="

PROJECT_DIR="/lustre/disk/home/users/mfaizan/motion_correction/prototyping/motion-artefacts-correction"
NOTEBOOK_DIR="/lustre/disk/home/users/mfaizan/motion_correction/prototyping/notebooks"

VIDEO_MAPPING="${NOTEBOOK_DIR}/foundcog_video_motion_mapping.csv"
REST_MAPPING="${NOTEBOOK_DIR}/rest10_foundcog_motion_mapping.csv"

CHUNK_CSV_DIR="${PROJECT_DIR}/dataset/chunk-5"
PREPROC_CSV_DIR="${CHUNK_CSV_DIR}/preprocessed"

OUTPUT_DIR="/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/faizan_motion_correction_dataset/cyclegans_chunk5_dataset"

mkdir -p "${CHUNK_CSV_DIR}" "${PREPROC_CSV_DIR}"

cd "${PROJECT_DIR}"

# =========================================================
# STEP 1: Create chunk CSVs with chunk_size=5
# =========================================================

echo ""
echo "--- Step 1a: Motion-corrupted chunks (video task) ---"
python -u motion_corrupted_chunk_data_creation.py \
    --mapping    "${VIDEO_MAPPING}" \
    --output     "${CHUNK_CSV_DIR}/motion_corrupted_chunk_dataset.csv" \
    --chunk_size 5

echo ""
echo "--- Step 1b: Motion-free resting state chunks ---"
python -u resting_state_motion_free_chunk_data_creation.py \
    --mapping    "${REST_MAPPING}" \
    --output     "${CHUNK_CSV_DIR}/resting_state_clean_chunk_dataset.csv" \
    --chunk_size 5

echo ""
echo "--- Step 1c: Motion-free video state chunks ---"
python -u resting_state_motion_free_chunk_data_creation.py \
    --mapping    "${VIDEO_MAPPING}" \
    --output     "${CHUNK_CSV_DIR}/video_state_clean_chunk_dataset.csv" \
    --chunk_size 5

# =========================================================
# STEP 2: Add preprocessed BOLD paths
# =========================================================

echo ""
echo "--- Step 2: Adding preprocessed paths ---"

python -u useful_scripts/add_preprocessed_path_to_data_csv_files.py \
    --input  "${CHUNK_CSV_DIR}/motion_corrupted_chunk_dataset.csv" \
    --output "${PREPROC_CSV_DIR}/motion_corrupted_chunk_dataset_with_preprocessed.csv" \
    --verify

python -u useful_scripts/add_preprocessed_path_to_data_csv_files.py \
    --input  "${CHUNK_CSV_DIR}/resting_state_clean_chunk_dataset.csv" \
    --output "${PREPROC_CSV_DIR}/resting_state_clean_chunk_dataset_with_preprocessed.csv" \
    --verify

python -u useful_scripts/add_preprocessed_path_to_data_csv_files.py \
    --input  "${CHUNK_CSV_DIR}/video_state_clean_chunk_dataset.csv" \
    --output "${PREPROC_CSV_DIR}/video_state_clean_chunk_dataset_with_preprocessed.csv" \
    --verify

# =========================================================
# STEP 3: Build CycleGAN dataset (extract NIfTI chunks)
# =========================================================

echo ""
echo "--- Step 3: Extracting NIfTI chunks into ${OUTPUT_DIR} ---"

python -u build_cycle_gans_dataset.py \
    --corrupted_csv         "${PREPROC_CSV_DIR}/motion_corrupted_chunk_dataset_with_preprocessed.csv" \
    --motion_free_video_csv "${PREPROC_CSV_DIR}/video_state_clean_chunk_dataset_with_preprocessed.csv" \
    --motion_free_rest_csv  "${PREPROC_CSV_DIR}/resting_state_clean_chunk_dataset_with_preprocessed.csv" \
    --output_dir            "${OUTPUT_DIR}" \
    --splits                train val test \
    --n_jobs                8

echo ""
echo "=========================================="
echo "  DONE — $(date)"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="
