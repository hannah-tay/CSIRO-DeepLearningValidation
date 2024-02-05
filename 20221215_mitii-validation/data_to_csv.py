import os
import pandas as pd

DATA_DIR = '../../../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/Mitii_CP'
CSV_DIR = '../../../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/participants.csv'
s = '../../../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/'

# image locations
# IMAGE: DATA_DIR/SUBJECT/.../T13D/{image_filename} + 'masked'
# SEGMENTATION: DATA_DIR/SUBJECT/.../masks/{image_filename} + 'seg_final'

# note: some subject folders have mutliple folders (sets of scans) to go through 
# just one set (atlas-registered images and segmentations) for each subject will be taken 

output = {
    'id': [],
    'img': [],
    'seg': [],
    'seg_tissue': []
}
img = seg = seg_tissue = None

for subject in os.listdir(DATA_DIR):
    subject_dir = os.path.join(DATA_DIR, subject)

    for folder in os.listdir(subject_dir):
        if '20' in folder:
            subject_dir = os.path.join(subject_dir, os.listdir(subject_dir)[0])
            break
    # subject_dir: DATA_DIR/SUBJECT/.../
    
    for datatype in os.listdir(subject_dir):
        datatype_dir = os.path.join(subject_dir, datatype)
        # datatype_dir: DATA_DIR/SUBJECT/.../T13D/
        #            or DATA_DIR/SUBJECT/.../masks/
        if datatype == 'T13D':
            for file_img in os.listdir(datatype_dir):
                if 'masked' in file_img:
                    img = os.path.join(datatype_dir, file_img).replace(s, '')
                    break

        if datatype == 'masks':
            for file_seg in os.listdir(datatype_dir):
                if 'seg_final' in file_seg:
                    seg = os.path.join(datatype_dir, file_seg).replace(s, '')
                    continue
                if 'tissues' in file_seg:
                    seg_tissue = os.path.join(datatype_dir, file_seg).replace(s, '')
                    continue


    if img and seg and seg_tissue: 
    # only save subject data if both images and segmentations are available
        output['id'].append(subject)
        output['img'].append(img)
        output['seg'].append(seg)
        output['seg_tissue'].append(seg_tissue)
        img = seg = seg_tissue = None # reset for the next subject

data = pd.DataFrame(output)
data.to_csv(CSV_DIR)
    
