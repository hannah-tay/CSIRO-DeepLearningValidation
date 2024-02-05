#!/bin/bash -l

#SBATCH --account=OD-221016

# run from /OASIS-1/disc*_nii/OAS_*_MR1/
module load freesurfer
source /apps/freesurfer/7.1.1/SetUpFreeSurfer.sh

mri_label2vol --seg ../../disc1/OAS1_0001_MR1/mri/aseg.mgz \
  --temp PROCESSED/MPRAGE/T88_111/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.nii \
  --o OAS1_0001_MR1_freesurfer.mgz --regheader ../../disc1/OAS1_0001_MR1/mri/aseg.mgz