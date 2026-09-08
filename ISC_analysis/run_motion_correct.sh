#!/bin/bash
#SBATCH --job-name=isc_correct
#SBATCH --output=../logs/isc_correct_%j.out
#SBATCH --error=../logs/isc_correct_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p ../logs

source ~/.bashrc
conda activate moco

python motion_correct.py --order A
