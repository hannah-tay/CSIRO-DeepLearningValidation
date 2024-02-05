# CSIRO Vacation Studentship 
Hannah Tay Nov 2022-Feb 2023

## Version Information
The current changes function in the `oasis_validation` branch. Some scripts will need to be modified when merging with the most recent version. 


## New Files
The [projects](/projects/) folder contains most of the new files, including the [ANTs registration script](/projects/20230112_ants-pipeline/ants_registration.py), [dice calculation script](/projects/20230203_generic-scripts/calculate_dice.py), and [boxplot script](/projects/20230203_generic-scripts/boxplot.py). 

The [script for warping images](/src/mirorrnet/src/mirorrnet/cli/warp_validation_set.py) is located in the Mirorrnet source code. The [script for plotting images](/src/mirorrnet/src/mirorrnet/cli/plot_images.py) is deprecated as the warping script saves images out in job_data, but can still be used.

Figures and progress reports can be found in OneNote.



## Dataset Integration
#### **OASIS-1**
The OASIS-1 images were converted from .hdr to .nii files using [convert.py](/projects/20221205_oasis-1-validation/convert.py). This script needs to be run for each disc folder (e.g. disc1) in the [OASIS-1 folder](/datasets/work/hb-atlas/work/scratch/data/OASIS-1/), but can be easily modified to loop over all disc folders. The .csv file used is also in this same folder. 
The .csv file contains several different options for images. The column `img_masked` was used for all registration (brain-masked images). 

Once converted, the images are loaded into Mirorrnet using [oasis1.py](/src/mirorrnet/src/mirorrnet/datasets/oasis1.py) which is a reimplementation of `generic.py` with a modified `load_transform`. When loading FreeSurfer segmentations rather than FSL, comment out the `SqueezeDimD` command in the `load_transform` (line 50). 

#### **Mitii CP**
The Mitii images are loaded into Mirorrnet using `generic.py` (`ImageandPhenotypeLDataModule`). The images and .csv file are located in the [Mitii_CP folder](/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/).



## Registration Outputs
The [job_data](/job_data/) folder contains images/segmentations (fixed, moving, warped, etc.) warp fields, and dice scores for all of the registration experiments performed. Each output folder within job_data is named according to the registration method and dataset used (in the form [{method}_registration_{dataset}]). The [sbatch_out](/job_data/sbatch_out/) folder contains the slurm outputs from ANTs registration. 

Methods:
- ants: ANTs SyN using outputs from ants_registration.py
- mirorrnet: Voxelmorph using Mirorrnet source code

Datasets (folder suffixes):
- mitii: Mitii CP dataset _(ANTs only)_
- mitii_extra: Single image pair used to compare ANTs with Mirorrnet in final presentation _(ANTs only)_
- *_modified: Mitii CP dataset with maximum severity threshold of 2 (maximum 3)
- *_tissue: Mitii CP dataset using tissue segmentations
- mitii_1: Mitii CP dataset; model trained from scratch _(Mirorrnet only)_
- mitii_2: Mitii CP dataset; model trained on OASIS and Mitii data _(Mirorrnet only)_
- mitii_3: Mitii CP dataset; model only trained on OASIS _(Mirorrnet only)_
- oasis: OASIS-1 dataset
- oasis_freesurfer: OASIS-1 dataset with FreeSurfer segmentations _(ANTs only)_
- oasis_freesurfer_vxm: Oasis-1 FreeSurfer segmentations; registered with parameters from Voxelmorph paper _(ANTs only)_

job_data also contains the [animations](/job_data/animations/) and [plots](/job_data/plots/) that were generated. 

The [registration_losses](/job_data/registration_losses/) folder contains .csv files of training and validation losses from the relevant Mirorrnet registrations. The version numbers can be used to identify the runs:
| Version # | Information |
| --- | --- |
| `15835763` | Best OASIS model |
| `16548053` | Trained on MITII from scratch (maximum severity of 2) |
| `16547705` | Trained on MITII from pre-trained OASIS model (maximum severity of 2) |
| `16608275` | Trained on MITII from scratch (lower `param-grad`, maximum severity of 2) |
| `16672676` | Trained on MITII from scratch |
| `16672717` | Trained on MITII from pre-trained OASIS model |



## ANTs Pipeline
#### **[ants_registration.py](/projects/20230112_ants-pipeline/ants_registration.py)**
Arguments:
- `data_table` (str): path to .csv file with dataset information
- `image_key` (str): image key (`img_masked` for OASIS-1, `img` for Mitii)
- `seg_key` (str): segmentation key (`seg` for most cases)
- `id_key` (str): subject ID key (`ID` for OASIS-1, `id` for Mitii)
- `n_valid` (int): validation set size (40 was used)
- `n_test` (int): test set size (50 was used)
- `dataset` (str): dataset name for output folder name (see above)

Example call:
```
python /flush5/tay400/projects/20230112_ants-pipeline/ants_registration.py \
  --data_table /datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv \
  --image_key img --seg_key seg --id_key id --n_valid 40 --n_test 50 --dataset mitii 
```
Creates folder for each image pair (ants_registration_*) within output folder (ants_registration_{`dataset`}). Creates ants_registration_*.q and resample_ants_registration_*.q for each image pair as well as saves moving and fixed image IDs. 

`ants_registration_*.q`: Performs registration and saves images, segmentations for one image pair.

`resample_ants_registration_*.q`: Produces and saves warped segmentations for one image pair.

#### **Performing ANTs registration**
```
cd /job_data/*  # desired folder (e.g. ants_registration_mitii_modified)
find . -maxdepth 2 -name 'ants_registration_*.q' -exec sbatch '{}' \;   # perform registration
# many jobs will be submitted via sbatch
# once everything is complete,
find . -maxdepth 2 -name 'resample_ants_registration_*.q' -exec sbatch '{}' \; # warp segmentations
```
Dice score calculation/other analysis can now be run.



## Mirorrnet Scripts
#### **[warp_validation_set.py](/src/mirorrnet/src/mirorrnet/cli/warp_validation_set.py)**
Can be run directly from the command line (`warp_validation_set`) (Note: edits were made to [setup.py](/src/mirorrnet/setup.py) to allow this).

Arguments:
Same as `train_registration`, with some additions:
- `dataset` (str): dataset name for output folder name (see above)
- `model_path` (str): path to checkpoint for registration model
- `seg_id` (str): segmentation key (defaults to `seg`)
- `subject_id` (str): subject ID key 
- `freesurfer` (bool): indicates whether FreeSurfer segmentations are used

Example call:
```
PYTHONPATH=src/mirorrnet/src/ \
warp_validation_set \
  --model_path 'runs/vxm/lightning_logs/version_16548053/checkpoints/last.ckpt' \
  --dataset mitii_1 \
  --data_table '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv' \
  --csv_image_key 'img' --csv_extra_image_keys 'seg' \
  --subject_id 'id' --registration-module-class NonrigidModule \
  --num_workers 0  --batch_size 2 --learning-rate 1e-2 --image-metric ncc \
  --n_valid 40 --n_test 50 --registration-module-param-res-reduce 2 \
  --lambda-im-loss-forward 0.5 --lambda-im-loss-backward 0.5 
```
Creates folder for each image pair (mirorrnet_registration_*) within output folder (mirorrnet_registration_{`dataset`}). Saves images (moving, fixed, warped, segmentations, warp fields) for each pair. 
Once finished, dice score calculation/other analysis can be run.


#### **[calculate_dice.py](/projects/20230203_generic-scripts/calculate_dice.py)**
Arguments:
- `dataset` (str): dataset name to locate correct output folder
- `method` (str): ants or mirorrnet; used to locate correct output folder
- `num_classes` (int): number of segmentation classes (`4` for OASIS-1, `23` for Mitii)
- `version` (str): version number for .hdf5 file output (not necessary if not using `plot_images.py` with output)
- `output_dir` (str): path to desired output .hdf5 file (optional)

Example call:
```
python projects/20230203_generic-scripts/calculate_dice.py \
  --dataset mitii_1 --method mirorrnet --num_classes 23 --version 16548053 \
  --output_dir '/datasets/work/hb-atlas/work/scratch/data/Mitii_CP/dice_losses_16548053.hdf5'
```
Calculates dice scores for each image pair in validation dataset and saves all outputs into .csv file (saved in registration folder). Prints number of failures and average dice scores (for each region) for quick reference. If .hdf5 output file is specified, also saves images and dice scores to in .hdf5 format. 


#### **[boxplot.py](/projects/20230203_generic-scripts/boxplot.py)**
Arguments:
- `csv_file` (str): path to .csv file with relevant dice scores (need to be manually combined before running this script)
- `output_dir` (str): path to desired output .png file (generally in /job_data/plots/)
- `plot_title` (str): desired plot title
- `widen` (bool): whether to edit font size etc. for a wider plot; generally only used when plotting all classes in Mitii (defaults to False)

Example call:
```
python projects/20230203_generic-scripts/boxplot.py --csv_file job_data/ants_vs_dl_oasis.csv \
--output_dir job_data/plots/boxplot_dl_ants.png --plot_title 'ANTs vs DL (OASIS)'
```
Creates and saves boxplot (x = region, y = dice score, hue = method) for given .csv file using Seaborn. 
Expected column order: blank (row numbers) --> info --> brain regions --> method (see [example file](/job_data/ants_registration_mitii/ants_mitii.csv)).


#### **[generalise.py](/projects/20230203_generic-scripts/generalise.py)**
Arguments:
- `dataset_csv` (str): path to .csv file with dataset information
- `plot_csv` (str): path to .csv with data for plotting (see `boxplot.py --csv_file`)

Example call:
```
python projects/20230203_generic-scripts/generalise.py --dataset_csv /datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants_modified.csv --plot_csv job_data/generalisation_mitii.csv 
```
Assigns a severity label (e.g. `1_to_1`, `1_to_3`) for every pair validated in `plot_csv` (expected column order same as for `boxplot.py`).
Generates a boxplot for every brain region present in `plot_csv` with x = rating, y = dice score, hue = method. Saves all plots in new folder [generalisation_mitii/](/job_data/plots/generalisation_mitii/).

In progress: calculates "generalisation gap" (difference in dice scores between methods) and produces boxplot of the gap for three brain regions (`generalisation.png`). 



## Useful Links
1.	Registration paper: https://www.sciencedirect.com/science/article/pii/S1361841507000606?via%3Dihub 
2.	Voxelmorph: https://arxiv.org/pdf/1809.05231.pdf
3.	Registration methods comparison (reference): https://arxiv.org/pdf/1810.08315.pdf
4.	ANTs registration code: https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images,-pointsets-and-Jacobians
5.	Convert from FreeSurfer to native space: https://surfer.nmr.mgh.harvard.edu/fswiki/FsAnat-to-NativeAnat
6.	Convert .mgz to .nii: https://neurostars.org/t/freesurfer-mgz-to-nifti/21647
