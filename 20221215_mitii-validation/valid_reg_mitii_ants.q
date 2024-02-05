#!/bin/bash -l

set -e
source envs/mirorrnet/bin/activate

# ANTs full MITII dataset - COMPLETE
python projects/calculate_dice.py \
  --dataset mitii --method ants --num_classes 23 --version ANTs \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_ANTs.hdf5'

# ANTs partial MITII dataset - COMPLETE
python projects/calculate_dice.py \
  --dataset mitii_modified --method ants --num_classes 23 --version ANTs \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_ANTs_modified.hdf5'

# ANTs full MITII dataset (tissue segmentations)
python projects/calculate_dice.py \
  --dataset mitii_tissue --method ants --num_classes 4 --version ANTs \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_ANTs_tissue.hdf5'

# ANTs partial MITII dataset (tissue segmentations)
python projects/calculate_dice.py \
  --dataset mitii_tissue_modified --method ants --num_classes 4 --version ANTs \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_ANTs_tissue_modified.hdf5'