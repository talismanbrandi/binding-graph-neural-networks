#!/bin/bash
#SBATCH --nodes=1                       # number of nodes requested
#SBATCH --ntasks=1                      # number of tasks (default: 1)
#SBATCH --ntasks-per-node=1             # number of tasks per node (default: whole node)
#SBATCH --partition=maxgpu              # partition to run in (all or maxwell)
#SBATCH --job-name=DNN-UUID             # job name
#SBATCH --output=DNN-TF-UUID-%N-%j.out  # output file name
#SBATCH --error=DNN-TF-UUID-%N-%j.err   # error file name
#SBATCH --time=96:00:00                 # runtime requested
#SBATCH --mail-user=ayan.paul@desy.de   # notification email
#SBATCH --mail-type=END,FAIL            # notification type
#SBATCH --export=ALL
#SBATCH --constraint=A100-PCIE-80GB
export LD_PRELOAD=""

# load module
module load python/3.10
module load cuda/11.4

# run the application:
source .venv/bin/activate
python3.10 torch-NN.py config-UUID.json
deactivate
rm config-UUID.json
