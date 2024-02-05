#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=28
#SBATCH --mem=256GB
#SBATCH --time=02:00:00
#SBATCH --account=OD-221016
set -e

source envs/mirorrnet/bin/activate

# MITII registration
# using model trained from scratch - version 1 - COMPLETE
PYTHONPATH=src/mirorrnet/src/ \
warp_validation_set \
  --model_path 'runs/vxm/lightning_logs/version_16548053/checkpoints/last.ckpt' \
  --dataset mitii_1_modified \
  --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
  --csv_image_key 'img' --csv_extra_image_keys 'seg' \
  --subject_id 'id' --registration-module-class NonrigidModule \
  --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
  --n_valid 40 --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 

# # version 1 tissue segmentations - COMPLETE
PYTHONPATH=src/mirorrnet/src/ \
warp_validation_set \
  --model_path 'runs/vxm/lightning_logs/version_16548053/checkpoints/last.ckpt' \
  --dataset mitii_1_tissue_modified \
  --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
  --csv_image_key 'img' --csv_extra_image_keys 'seg_tissue' \
  --seg_id 'seg_tissue' \
  --subject_id 'id' --registration-module-class NonrigidModule \
  --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
  --n_valid 40 --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 

# using model trained from OASIS model - version 2 - COMPLETE
PYTHONPATH=src/mirorrnet/src/ \
warp_validation_set \
  --model_path 'runs/vxm/lightning_logs/version_16547705/checkpoints/last.ckpt' \
  --dataset mitii_2_modified \
  --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
  --csv_image_key 'img' --csv_extra_image_keys 'seg' \
  --subject_id 'id' --registration-module-class NonrigidModule \
  --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
  --n_valid 40 --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5;

# # version 2 tissue segmentations - COMPLETE
PYTHONPATH=src/mirorrnet/src/ \
warp_validation_set \
  --model_path 'runs/vxm/lightning_logs/version_16547705/checkpoints/last.ckpt' \
  --dataset mitii_2_tissue_modified \
  --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
  --csv_image_key 'img' --csv_extra_image_keys 'seg_tissue' \
  --seg_id 'seg_tissue' \
  --subject_id 'id' --registration-module-class NonrigidModule \
  --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
  --n_valid 40 --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 

# registration with pre-trained OASIS model (no extra training - version 3) - COMPLETE
PYTHONPATH=src/mirorrnet/src/ \
warp_validation_set \
  --model_path 'runs/vxm/lightning_logs/version_15835763/checkpoints/last.ckpt' \
  --dataset mitii_3_modified \
  --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
  --csv_image_key 'img' --csv_extra_image_keys 'seg' \
  --subject_id 'id' --registration-module-class NonrigidModule \
  --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
  --n_valid 40 --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5;

# # version 3 tissue segmentations - COMPLETE
PYTHONPATH=src/mirorrnet/src/ \
warp_validation_set \
  --model_path 'runs/vxm/lightning_logs/version_15835763/checkpoints/last.ckpt' \
  --dataset mitii_3_tissue_modified \
  --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
  --csv_image_key 'img' --csv_extra_image_keys 'seg_tissue' \
  --seg_id 'seg_tissue' \
  --subject_id 'id' --registration-module-class NonrigidModule \
  --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
  --n_valid 40 --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 

# using model trained from scratch - version 1 - COMPLETE
python projects/calculate_dice.py \
  --dataset mitii_modified --method mirorrnet --num_classes 23 --version 16548053 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16548053.hdf5'

# # version 1 tissue segmentations - COMPLETE
python projects/calculate_dice.py \
  --dataset mitii_1_tissue_modified --method mirorrnet --num_classes 4 --version 16548053 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16548053_tissue.hdf5'

# registration with pre-trained OASIS model (no extra training - version 3) - COMPLETE
python projects/calculate_dice.py \
  --dataset mitii_3_modified --method mirorrnet --num_classes 23 --version 15835763 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_15835763_modified.hdf5'

# # version 3 tissue segmentations - COMPLETE
python projects/calculate_dice.py \
  --dataset mitii_3_tissue_modified --method mirorrnet --num_classes 4 --version 15835763 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_15835763_tissue_modified.hdf5'

# using model trained from OASIS model - version 2 - COMPLETE
python projects/calculate_dice.py \
  --dataset mitii_2_modified --method mirorrnet --num_classes 23 --version 16547705 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16547705.hdf5'

# # version 2 tissue segmentations - COMPLETE
python projects/calculate_dice.py \
  --dataset mitii_2_tissue_modified --method mirorrnet --num_classes 4 --version 16547705 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16547705_tissue.hdf5'
