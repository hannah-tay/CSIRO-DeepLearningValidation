#!/bin/bash -l

set -e

source envs/ants/bin/activate

# run from flush5/tay400/

python /flush5/tay400/projects/20230112_ants-pipeline/ants_registration.py \
  --data_table /datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv \
  --image_key img --seg_key seg --id_key id --n_valid 40 --n_test 50 --dataset mitii 


# python /flush5/tay400/projects/20230112_ants-pipeline/ants_registration.py \
#   --data_table /datasets/work/hb-atlas/work/scratch/data/OASIS-1/participants_modified.csv \
#   --image_key img_masked --seg_key seg --id_key ID --n_valid 40 --dataset oasis \


python /flush5/tay400/projects/20230112_ants-pipeline/ants_registration_fs.py \
  --data_table /datasets/work/hb-atlas/work/scratch/data/OASIS-1/participants_modified_fs.csv \
  --image_key img_masked --seg_key seg --id_key id --n_valid 40 --dataset oasis \
  --freesurfer True


# python /flush5/tay400/projects/20230112_ants-pipeline/ants_registration.py \
#   --data_table /datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv \
#   --image_key img --seg_key seg_tissue --id_key id --n_valid 40 --dataset mitii_tissue \
#   --max_severity 2

# python /flush5/tay400/projects/20230112_ants-pipeline/ants_registration.py \
#   --data_table /datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv \
#   --image_key img --seg_key seg_tissue --id_key id --n_valid 40 --dataset mitii_tissue 