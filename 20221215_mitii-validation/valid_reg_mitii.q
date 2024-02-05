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
# # using model trained from scratch - version 1 - COMPLETE
# PYTHONPATH=src/mirorrnet/src/ \
# warp_validation_set \
#   --model_path 'runs/vxm/lightning_logs/version_16672676/checkpoints/last.ckpt' \
#   --dataset mitii_1 \
#   --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
#   --csv_image_key 'img' --csv_extra_image_keys 'seg' \
#   --subject_id 'id' --registration-module-class NonrigidModule \
#   --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
#   --n_valid 40 --registration-module-param-res-reduce 2 \
#   --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 

# # version 1 tissue segmentations - COMPLETE
# PYTHONPATH=src/mirorrnet/src/ \
# warp_validation_set \
#   --model_path 'runs/vxm/lightning_logs/version_16672676/checkpoints/last.ckpt' \
#   --dataset mitii_1_tissue \
#   --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
#   --csv_image_key 'img' --csv_extra_image_keys 'seg_tissue' \
#   --seg_id 'seg_tissue' \
#   --subject_id 'id' --registration-module-class NonrigidModule \
#   --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
#   --n_valid 40 --registration-module-param-res-reduce 2 \
#   --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 


# # using model trained from OASIS model - version 2 - COMPLETE
# PYTHONPATH=src/mirorrnet/src/ \
# warp_validation_set \
#   --model_path 'runs/vxm/lightning_logs/version_16672717/checkpoints/last.ckpt' \
#   --dataset mitii_2 \
#   --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
#   --csv_image_key 'img' --csv_extra_image_keys 'seg' \
#   --subject_id 'id' --registration-module-class NonrigidModule \
#   --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
#   --n_valid 40 --registration-module-param-res-reduce 2 \
#   --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5;

# # version 2 tissue segmentations - COMPLETE
# PYTHONPATH=src/mirorrnet/src/ \
# warp_validation_set \
#   --model_path 'runs/vxm/lightning_logs/version_16672717/checkpoints/last.ckpt' \
#   --dataset mitii_2_tissue \
#   --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
#   --csv_image_key 'img' --csv_extra_image_keys 'seg_tissue' \
#   --seg_id 'seg_tissue' \
#   --subject_id 'id' --registration-module-class NonrigidModule \
#   --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
#   --n_valid 40 --registration-module-param-res-reduce 2 \
#   --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 


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

# # version 3 tissue segmentations - COMPLETE
PYTHONPATH=src/mirorrnet/src/ \
warp_validation_set \
  --model_path 'runs/vxm/lightning_logs/version_15835763/checkpoints/last.ckpt' \
  --dataset mitii_3_tissue \
  --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
  --csv_image_key 'img' --csv_extra_image_keys 'seg_tissue,antoher_key' \
  --seg_id 'seg_tissue' \
  --subject_id 'id' --registration-module-class NonrigidModule \
  --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
  --n_valid 40 --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 


# # using model trained from scratch - version 1 - COMPLETE
# python projects/20230203_generic-scripts/calculate_dice.py \
#   --dataset mitii_1 --method mirorrnet --num_classes 23 --version 16672676 \
#   --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16672676.hdf5'

# # version 1 tissue segmentations - COMPLETE
# python projects/20230203_generic-scripts/calculate_dice.py \
#   --dataset mitii_1_tissue --method mirorrnet --num_classes 4 --version 16672676 \
#   --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16672676_tissue.hdf5'

# # registration with pre-trained OASIS model (no extra training - version 3) - COMPLETE
# python projects/20230203_generic-scripts/calculate_dice.py \
#   --dataset mitii_3 --method mirorrnet --num_classes 23 --version 15835763 \
#   --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_15835763.hdf5'

# # version 3 tissue segmentations
# python projects/20230203_generic-scripts/calculate_dice.py \
#   --dataset mitii_3_tissue --method mirorrnet --num_classes 4 --version 15835763 \
#   --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_15835763_tissue.hdf5'

# # using model trained from OASIS model - version 2 - COMPLETE
# python projects/20230203_generic-scripts/calculate_dice.py \
#   --dataset mitii_2 --method mirorrnet --num_classes 23 --version 16672717 \
#   --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16672717.hdf5'

# # version 2 tissue segmentations - COMPLETE
# python projects/20230203_generic-scripts/calculate_dice.py \
#   --dataset mitii_2_tissue --method mirorrnet --num_classes 4 --version 16672717 \
#   --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16672717_tissue.hdf5'

