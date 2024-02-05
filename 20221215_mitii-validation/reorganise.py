import os
import shutil

DATA_DIR = '../../../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/Mitii_CP'
OUTPUT_DIR = '../../../../datasets/work/hb-atlas/work/scratch/data/Mitii_CP/Mitii_CP_Reorganised/'

# image locations
# IMAGE: DATA_DIR/SUBJECT/.../T13D/{image_filename} + 'masked'
# SEGMENTATION: DATA_DIR/SUBJECT/.../masks/{image_filename} + 'seg_final'

# new image locations
# IMAGE: OUTPUT_DIR/IMAGES
# SEGMENTATION: OUTPUT_DIR/SEGMENTATIONS


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
                    img = os.path.join(datatype_dir, file_img)
                    output_path = OUTPUT_DIR + 'IMAGES/'
                    os.makedirs(output_path, exist_ok=True)
                    shutil.copy(img, output_path)

        if datatype == 'masks':
            for file_seg in os.listdir(datatype_dir):
                if 'seg_final' in file_seg:
                    seg = os.path.join(datatype_dir, file_seg)
                    output_path = OUTPUT_DIR + 'SEGMENTATIONS/'
                    os.makedirs(output_path, exist_ok=True)
                    shutil.copy(seg, output_path)