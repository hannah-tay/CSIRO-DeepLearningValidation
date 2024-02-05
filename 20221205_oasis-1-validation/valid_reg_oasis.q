#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=14
#SBATCH --mem=128GB
#SBATCH --time=30:00
#SBATCH --account=OD-221016
set -e

source envs/mirorrnet/bin/activate

# # mirorrnet-registered OASIS - COMPLETE
# PYTHONPATH=src/mirorrnet/src/ \
# warp_validation_set \
#   --model_path 'runs/vxm/lightning_logs/version_15835763/checkpoints/last.ckpt' \
#   --dataset oasis \
#   --csv_file '../../datasets/work/hb-atlas/work/scratch/data/OASIS-1/participants_modified.csv' \
#   --subject_id 'ID' --registration-module-class NonrigidModule \
#   --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
#   --n_valid 40 --registration-module-param-res-reduce 2 \
#   --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5;

# freesurfer OASIS - COMPLETE
# PYTHONPATH=src/mirorrnet/src/ \
# warp_validation_set \
#   --model_path 'runs/vxm/lightning_logs/version_16935173/checkpoints/last.ckpt' \
#   --dataset oasis \
#   --csv_file '../../datasets/work/hb-atlas/work/scratch/data/OASIS-1/participants_modified_fs.csv' \
#   --subject_id 'id' --registration-module-class NonrigidModule \
#   --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
#   --n_valid 40 --registration-module-param-res-reduce 2 \
#   --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 --freesurfer True

# # mirorrnet-registered OASIS - COMPLETE
# python projects/calculate_dice.py \
#   --dataset oasis --method mirorrnet --num_classes 4 --version 15835763 \
#   --output_dir '/datasets/work/hb-atlas/work/scratch/data/OASIS-1/dice_losses_15835763.hdf5'

# # ANTs-registered OASIS - COMPLETE
# python projects/calculate_dice.py \
#   --dataset oasis --method ants --num_classes 4 --version ANTs \
#   --output_dir '/datasets/work/hb-atlas/work/scratch/data/OASIS-1/dice_losses_ANTs.hdf5'

# # calculate dice for warped freesurfer segmentations (ANTs) - COMPLETE
# python projects/20230203_generic-scripts/calculate_dice.py \
#   --dataset oasis_freesurfer --method ants --num_classes 86 --version ANTs \
#   --output_dir '/datasets/work/hb-atlas/work/scratch/data/OASIS-1/dice_losses_ANTs_freesurfer.hdf5'

# calculate dice for warped freesurfer segmentations (vxm parameters) (ANTs)
python projects/20230203_generic-scripts/calculate_dice.py \
  --dataset oasis_freesurfer_vxm --method ants --num_classes 86 --version ANTs \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/OASIS-1/dice_losses_ANTs_freesurfer_vxm.hdf5'

# calculate dice for warped freesurfer segmentations - COMPLETE
# python projects/20230203_generic-scripts/calculate_dice.py \
#   --dataset oasis_freesurfer --method mirorrnet --num_classes 86 --version 16935173  \
#   --output_dir '/datasets/work/hb-atlas/work/scratch/data/OASIS-1/dice_losses_16935173.hdf5'