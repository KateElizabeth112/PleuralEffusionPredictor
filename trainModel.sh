#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=15:mem=80gb:ngpus=1:gpu_type=RTX6000

# bash script to train the ResNet classifier on HPC
cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal

# run experiments
python preprocessCheXpert.py -r "/rds/general/user/kc2322/home/" -c "configHPC.toml"