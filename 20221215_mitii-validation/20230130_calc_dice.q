#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=14
#SBATCH --mem=128GB
#SBATCH --time=2-0:00:00
#SBATCH --account=OD-221016
set -e

source envs/mirorrnet/bin/activate

# using model trained from scratch - version 1 - TODO 30/01
python projects/calculate_dice.py \
  --dataset mitii_1 --method mirorrnet --num_classes 23 --version 16341847 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16341847.hdf5'

# registration with pre-trained OASIS model (no extra training - version 3) - TODO 30/01'
python projects/calculate_dice.py \
  --dataset mitii_3 --method mirorrnet --num_classes 23 --version 15835763 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_15835763.hdf5'