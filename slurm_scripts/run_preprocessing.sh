#!/bin/bash
#SBATCH --job-name=cyclegan_preprocess
#SBATCH --output=logs/cyclegan_preprocess_%j.out
#SBATCH --error=logs/cyclegan_preprocess_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

mkdir -p logs

export FSLDIR=/lustre/disk/home/users/mfaizan/fsl
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}
export FSLOUTPUTTYPE=NIFTI_GZ

source ~/.bashrc
conda activate moco

python -u preprocess_normalized_data_for_training.py --all