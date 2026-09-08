#!/bin/bash
#SBATCH --job-name=isc_chunk
#SBATCH --output=logs/isc_chunk_%j.out
#SBATCH --error=logs/isc_chunk_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

mkdir -p logs

source ~/.bashrc
conda activate moco

python ISC_analysis/build_isc_chunks.py --order A --workers 16
