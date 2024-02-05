import os
import pandas as pd
# TODO: download dics 10-12 and rerun script for full .csv

DATA_DIR = '/datasets/work/hb-atlas/work/scratch/data/OASIS-1/'
CSV_DIR = '/datasets/work/hb-atlas/work/scratch/data/OASIS-1/participants.csv'
s = '/datasets/work/hb-atlas/work/scratch/data/OASIS-1/'

# IMG: /mri/nu.mgz
# SEG: /mri/aseg.mgz

data = {
    'id': [],
    'img_masked': [],
    'seg': []
}

for FOLDER in os.listdir(DATA_DIR):
    FOLDER_DIR = os.path.join(DATA_DIR, FOLDER)
    if '_nii' not in FOLDER and os.path.isdir(FOLDER_DIR):
        for id in sorted(os.listdir(FOLDER_DIR)): # OAS1_XXXX_MRI
            id_dir = os.path.join(FOLDER_DIR, id)

            for folder in os.listdir(id_dir): # OAS1_XXXX_MRI/DATATYPE
                # image filename
                if folder == 'mri':
                    folder_dir = os.path.join(id_dir, folder)
                    data['id'].append(id)
                    data['img_masked'].append(os.path.join(folder_dir, 'nu.mgz').replace(s, '')) # brain-masked atlas registered image
                    data['seg'].append(os.path.join(folder_dir, 'aseg.mgz').replace(s, '')) # brain tissue segmentation

# print(data['id'][-1], data['img_masked'][-1], data['seg'][-1])

data = pd.DataFrame.from_dict(data)

# print(len(data['seg']))
# print(len(data['img_atlas_reg']))

data.to_csv('../../datasets/work/hb-atlas/work/scratch/data/OASIS-1/participants_modified_fs.csv')