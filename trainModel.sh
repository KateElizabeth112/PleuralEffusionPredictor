#!/bin/bash
#PBS -o /rds/general/user/kc2322/home/logs
#PBS -j oe
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=15:mem=80gb:ngpus=1:gpu_type=RTX6000

# bash script to train the ResNet classifier on HPC
cd ${PBS_O_WORKDIR}

# Load the python version we will use
module load Python/3.9.5-GCCcore-10.3.0   

# load the virtual environment
source .venv/bin/activateTRAI

# run experiments
python trainModel.py -r "/rds/general/user/kc2322/home/" -c "configHPC.toml" -t "$TRAIN_IDS_FILE"