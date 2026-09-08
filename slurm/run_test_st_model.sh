#!/bin/bash
#SBATCH --job-name=cyclegan_st_test
#SBATCH --output=logs/cyclegan_st_test_%j.out
#SBATCH --error=logs/cyclegan_st_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

mkdir -p logs

source ~/.bashrc
conda activate moco

python -u -m tests.test_st_model
