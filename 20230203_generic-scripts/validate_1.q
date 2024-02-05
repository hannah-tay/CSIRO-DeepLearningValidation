#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=256GB
#SBATCH --time=12:00:00
#SBATCH --account=OD-221016

set -e

source envs/mirorrnet/bin/activate

# VALIDATE MIRORRNET ================================================
# using model trained from scratch - version 1
PYTHONPATH=src/mirorrnet/src/ \
warp_validation_set \
  --model_path 'runs/vxm/lightning_logs/version_16548053/checkpoints/last.ckpt' \
  --dataset mitii_1 \
  --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
  --csv_image_key 'img' --csv_extra_image_keys 'seg' \
  --subject_id 'id' --registration-module-class NonrigidModule \
  --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
  --n_valid 40 --n_test 50 --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 
