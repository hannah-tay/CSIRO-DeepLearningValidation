#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=14
#SBATCH --mem=128GB
#SBATCH --time=2:00:00
#SBATCH --account=OD-221016

set -e

# VALIDATE ANTS =====================================================
source envs/mirorrnet/bin/activate
# ANTs full MITII dataset
python projects/20230203_generic-scripts/calculate_dice.py \
  --dataset mitii --method ants --num_classes 23 --version ANTs \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_ANTs.hdf5'


# VALIDATE MIRORRNET ================================================
# # using model trained from scratch - version 1
# PYTHONPATH=src/mirorrnet/src/ \
# warp_validation_set \
#   --model_path 'runs/vxm/lightning_logs/version_16548053/checkpoints/last.ckpt' \
#   --dataset mitii_1 \
#   --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
#   --csv_image_key 'img' --csv_extra_image_keys 'seg' \
#   --subject_id 'id' --registration-module-class NonrigidModule \
#   --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
#   --n_valid 40 --n_test 50 --registration-module-param-res-reduce 2 \
#   --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 

# # using model trained from OASIS model - version 2
# PYTHONPATH=src/mirorrnet/src/ \
# warp_validation_set \
#   --model_path 'runs/vxm/lightning_logs/version_16547705/checkpoints/last.ckpt' \
#   --dataset mitii_2 \
#   --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
#   --csv_image_key 'img' --csv_extra_image_keys 'seg' \
#   --subject_id 'id' --registration-module-class NonrigidModule \
#   --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
#   --n_valid 40 --registration-module-param-res-reduce 2 \
#   --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5;

# # registration with pre-trained OASIS model (no extra training - version 3) - COMPLETE
# PYTHONPATH=src/mirorrnet/src/ \
# warp_validation_set \
#   --model_path 'runs/vxm/lightning_logs/version_15835763/checkpoints/last.ckpt' \
#   --dataset mitii_3 \
#   --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
#   --csv_image_key 'img' --csv_extra_image_keys 'seg' \
#   --subject_id 'id' --registration-module-class NonrigidModule \
#   --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
#   --n_valid 40 --registration-module-param-res-reduce 2 \
#   --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5;

# using model trained from scratch - version 1
python projects/20230203_generic-scripts/calculate_dice.py \
  --dataset mitii_1 --method mirorrnet --num_classes 23 --version 16548053 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16548053.hdf5'

# registration with pre-trained OASIS model (no extra training - version 3) 
python projects/20230203_generic-scripts/calculate_dice.py \
  --dataset mitii_3 --method mirorrnet --num_classes 23 --version 15835763 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_15835763_modified.hdf5'

# using model trained from OASIS model - version 2 
python projects/20230203_generic-scripts/calculate_dice.py \
  --dataset mitii_2 --method mirorrnet --num_classes 23 --version 16547705 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16547705.hdf5'
