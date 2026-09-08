#!/bin/bash
#SBATCH --job-name=isc_tc
#SBATCH --output=../logs/isc_tc_%j.out
#SBATCH --error=../logs/isc_tc_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

mkdir -p ../logs

source ~/.bashrc
conda activate moco

python time_courses.py --order A --workers 16
