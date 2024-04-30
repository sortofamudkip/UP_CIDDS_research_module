#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 8
#SBATCH -t 48:00:00
#SBATCH --gpus-per-node 1
 
module load anaconda3
 
source $ANACONDA3_ROOT/etc/profile.d/conda.sh
conda activate cidds-gan
 
python main.py
