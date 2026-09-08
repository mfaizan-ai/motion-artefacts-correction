#!/bin/bash
#SBATCH --job-name=isc_chunk
#SBATCH --output=logs/isc_chunk_%A_%a.out
#SBATCH --error=logs/isc_chunk_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --array=0-4

mkdir -p logs

source ~/.bashrc
conda activate moco

ORDERS=(B C D E F)
ORDER=${ORDERS[$SLURM_ARRAY_TASK_ID]}

python ISC_analysis/build_isc_chunks.py --order ${ORDER} --workers 16
