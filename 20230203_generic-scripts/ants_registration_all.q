#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=30:00
#SBATCH --account=OD-221016

set -e

source envs/ants/bin/activate
# ANTs register partial dataset
python /flush5/tay400/projects/20230112_ants-pipeline/ants_registration.py \
  --data_table /datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv \
  --image_key img --seg_key seg --id_key id --n_valid 40 --dataset mitii \
  --max_severity 2

# ANTs register full dataset
python /flush5/tay400/projects/20230112_ants-pipeline/ants_registration.py \
  --data_table /datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv \
  --image_key img --seg_key seg --id_key id --n_valid 40 --dataset mitii \

# ANTs register partial dataset (tissue segmentations)
python /flush5/tay400/projects/20230112_ants-pipeline/ants_registration.py \
  --data_table /datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv \
  --image_key img --seg_key seg_tissue --id_key id --n_valid 40 --dataset mitii \
  --max_severity 2

# ANTs register full dataset (tissue segmentations)
python /flush5/tay400/projects/20230112_ants-pipeline/ants_registration.py \
  --data_table /datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv \
  --image_key img --seg_key seg_tissue --id_key id --n_valid 40 --dataset mitii \
