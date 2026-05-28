#!/bin/bash
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=15:mem=80gb:ngpus=1:gpu_type=RTX6000

# bash script to run generalisation experiments on HPC
cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal

# install requirements
pip install -r requirements.txt

# run experiments
python preprocessCheXpert.py -d "/rds/general/user/kc2322/home/data" -i 320