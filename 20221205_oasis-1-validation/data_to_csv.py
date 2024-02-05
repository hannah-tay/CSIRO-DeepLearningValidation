import os
import pandas as pd


DATA_DIR = '../../datasets/work/hb-atlas/work/scratch/data/OASIS-1/'
CSV_DIR = '../../datasets/work/hb-atlas/work/scratch/data/OASIS-1/participants.csv'
s = '../../datasets/work/hb-atlas/work/scratch/data/OASIS-1/'


images_raw = []
images_averaged = []
images_brain_masked = []
images_atlas_registered = []
segmentations = []

for FOLDER in os.listdir(DATA_DIR):
    if '_nii' in FOLDER:
        FOLDER_DIR = os.path.join(DATA_DIR, FOLDER)
        for id in sorted(os.listdir(FOLDER_DIR)): # OAS1_XXXX_MRI
            id_dir = os.path.join(FOLDER_DIR, id)

            for folder in os.listdir(id_dir): # OAS1_XXXX_MRI/DATATYPE
                # image filename
                if folder == 'PROCESSED':
                    folder_dir = os.path.join(id_dir, folder)
                    folder_dir = os.path.join(folder_dir, 'MPRAGE/T88_111/')
                    images_atlas_registered.append(os.path.join(folder_dir, os.listdir(folder_dir)[0]).replace(s, '')) # gain-field corrected atlas registered image
                    images_brain_masked.append(os.path.join(folder_dir, os.listdir(folder_dir)[1]).replace(s, '')) # brain-masked atlas registered image

                    folder_dir = os.path.join(id_dir, folder)
                    folder_dir = os.path.join(folder_dir, 'MPRAGE/SUBJ_111/')
                    images_averaged.append(os.path.join(folder_dir, os.listdir(folder_dir)[0]).replace(s, '')) # average across all scans
                
                elif folder == 'FSL_SEG':
                    folder_dir = os.path.join(id_dir, folder)
                    segmentations.append(os.path.join(folder_dir, os.listdir(folder_dir)[0]).replace(s, '')) # brain tissue segmentation
                
                elif folder == 'RAW':
                    folder_dir = os.path.join(id_dir, folder)
                    images_raw.append(os.path.join(folder_dir, os.listdir(folder_dir)[0]).replace(s, '')) # individual scan

# print(images_raw[-1], images_averaged[-1], images_brain_masked[-1], images_atlas_registered[-1], segmentations[-1])


data = pd.read_csv(CSV_DIR)
data['img_raw'] = images_raw
data['img_avg'] = images_averaged
data['img_masked'] = images_brain_masked
data['img_atlas_reg'] = images_atlas_registered
data['seg'] = segmentations

# print(len(data['seg']))
# print(len(data['img_atlas_reg']))

data.to_csv('../../datasets/work/hb-atlas/work/scratch/data/OASIS-1/participants_modified.csv')