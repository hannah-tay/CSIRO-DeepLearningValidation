#!/bin/bash -l

# # version 1 tissue segmentations 
# PYTHONPATH=src/mirorrnet/src/ \
# warp_validation_set \
#   --model_path 'runs/vxm/lightning_logs/version_16548053/checkpoints/last.ckpt' \
#   --dataset mitii_1_tissue \
#   --data_table '../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
#   --csv_image_key 'img' --csv_extra_image_keys 'seg_tissue' \
#   --seg_id 'seg_tissue' \
#   --subject_id 'id' --registration-module-class NonrigidModule \
#   --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
#   --n_valid 40 --registration-module-param-res-reduce 2 \
#   --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 

# version 1 tissue segmentations
python projects/calculate_dice.py \
  --dataset mitii_1_tissue --method mirorrnet --num_classes 4 --version 16548053 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16548053_tissue.hdf5'