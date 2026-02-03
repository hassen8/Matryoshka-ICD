#!/bin/bash

#SBATCH --job-name=icd
#SBATCH --output=icd_%j.out
#SBATCH --error=icd_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00

# activate venv
source /home/hsali/projects/icd/.venv/bin/activate

# load env variables
source .env

# run Python script
uv run main.py