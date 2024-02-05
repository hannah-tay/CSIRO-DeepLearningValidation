#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=14
#SBATCH --mem=128GB
#SBATCH --time=10:00
#SBATCH --account=OD-221016
set -e

source /flush5/tay400/envs/mirorrnet/bin/activate

python /flush5/tay400/src/mirorrnet/src/mirorrnet/cli/plot_images.py \
  --file ../../datasets/work/hb-atlas/work/scratch/data/OASIS-1/dice_losses_15678938.hdf5 \
  --dataset OASIS-1 --moving OAS1_0002_MR1 --fixed OAS1_0005_MR1 \
  --plot_type nibabel