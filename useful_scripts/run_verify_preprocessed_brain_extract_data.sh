#!/bin/bash
#SBATCH --job-name=preprocessed_data_verification
#SBATCH --output=logs/preprocessed_data_verification_%j.out
#SBATCH --error=logs/preprocessed_data_verification_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu

mkdir -p logs

export FSLDIR=/lustre/disk/home/users/mfaizan/fsl
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}
export FSLOUTPUTTYPE=NIFTI_GZ

source ~/.bashrc
conda activate moco

python -u verify_preprocessed_brain_extract_data.py