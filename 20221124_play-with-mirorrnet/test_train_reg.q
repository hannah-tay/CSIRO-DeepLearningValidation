#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=28
#SBATCH --mem=256GB
#SBATCH --time=02:00:00
#SBATCH --account=OD-221016
set -e

source /home/tay400/envs/mirorrnet/bin/activate
cd /datasets/work/hb-atlas/work/projects/2022-01-06_st-pet-atlas

train_registration --dataset SCHN \
  --registration-module-class NonrigidModule\
  --gpus 4 --num_workers 28  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
  --n_valid 2 --log_every_n_steps 5 --max_epochs 100 \
  --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5