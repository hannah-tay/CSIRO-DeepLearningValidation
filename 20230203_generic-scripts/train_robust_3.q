#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=14
#SBATCH --mem=128GB
#SBATCH --time=3-00:00
#SBATCH --account=OD-221016
set -e

source envs/mirorrnet/bin/activate

PYTHONPATH=src/mirorrnet/src/ \
train_registration \
  --dataset "OASIS1" \
  --csv_file '../../datasets/work/hb-atlas/work/scratch/data/OASIS-1/participants_modified.csv' \
  --registration-module-class NonrigidModule \
  --gpus 2 --num_workers 14  --batch_size 2 --learning-rate 1e-3 --image-metric robustncc \
  --n_valid 5 --n_test 50 --log_every_n_steps 5 --max_epochs 100 \
  --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 \
  --lambda-param-grad 1e-1 --val_check_interval 50 \
  --robust_ncc_quantile 0.95 --robust_ncc_scale 0.75