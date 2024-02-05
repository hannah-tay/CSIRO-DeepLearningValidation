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

PYTHONPATH=src/mirorrnet/src/ \
train_registration \
  --csv_image_key 'img' --csv_extra_image_keys 'seg' \
  --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
  --ckpt-path 'runs/vxm/lightning_logs/version_15835763/checkpoints/last.ckpt' \
  --registration-module-class NonrigidModule \
  --gpus 2 --num_workers 14  --batch_size 2 --learning-rate 1e-3 --image-metric ncc \
  --n_valid 5 --n_test 10 --log_every_n_steps 5 --max_epochs 50 \
  --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 \
  --lambda-param-grad 1e-1 --val_check_interval 50