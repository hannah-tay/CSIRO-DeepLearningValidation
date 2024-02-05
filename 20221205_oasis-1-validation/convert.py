import os
import nibabel

INPUT_DIR = '../../datasets/work/hb-atlas/work/scratch/data/OASIS-1/disc1/'
OUTPUT_DIR = '../../datasets/work/hb-atlas/work/scratch/data/OASIS-1/disc1_nii/'

for subject_dir in os.listdir(INPUT_DIR): # e.g. INPUT_DIR/OAS1_0001_MRI/
    sub_dir_full = os.path.join(INPUT_DIR, subject_dir)

    for datatype in os.listdir(sub_dir_full): # e.g. INPUT_DIR/OAS1_0001_MRI/FSL_SEG/
        data_dir_full = os.path.join(sub_dir_full, datatype)

        if os.path.isdir(data_dir_full):
            if datatype == "PROCESSED": # e.g. INPUT_DIR/OAS1_0001_MR1/PROCESSED/
                data_dir_full = os.path.join(data_dir_full, 'MPRAGE/')

                for subfolder in os.listdir(data_dir_full): # e.g. INPUT_DIR/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/
                    sub_data_dir_full = os.path.join(data_dir_full, subfolder)

                    if os.path.isdir(sub_data_dir_full):
                        for filename in os.listdir(sub_data_dir_full): # e.g. INPUT_DIR/OAS1_0001_MR1/PROCESSED/MPRAGE/SUBJ_111/OAS1_0001_MR1_mpr_n4_anon_sbj_111.hdr
                            if filename.endswith('.hdr'):
                                file_path = os.path.join(sub_data_dir_full, filename)
                                file_path_nii = file_path.replace(INPUT_DIR, OUTPUT_DIR).replace('.hdr', '.nii')
                                folder_path_nii = sub_data_dir_full.replace(INPUT_DIR, OUTPUT_DIR)

                                os.makedirs(folder_path_nii, exist_ok=True)

                                image = nibabel.load(os.path.join(sub_data_dir_full, filename))
                                nii_image = nibabel.Nifti1Image(image.get_fdata(), image.affine, image.header)
                                nii_image.set_qform(image.affine)
                                nii_image.set_sform(image.affine)
                                nii_image.to_filename(file_path_nii)

            else: # e.g. INPUT_DIR/OAS1_0001_MR1/FSL_SEG/
                for filename in os.listdir(data_dir_full): # e.g. INPUT_DIR/OAS1_0001_MRI/FSL_SEG/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr
                    if filename.endswith('.hdr'):
                        file_path = os.path.join(data_dir_full, filename)
                        file_path_nii = file_path.replace(INPUT_DIR, OUTPUT_DIR).replace('.hdr', '.nii')
                        folder_path_nii = data_dir_full.replace(INPUT_DIR, OUTPUT_DIR)

                        os.makedirs(folder_path_nii, exist_ok=True)

                        image = nibabel.load(os.path.join(data_dir_full, filename))
                        nii_image = nibabel.Nifti1Image(image.get_fdata(), image.affine, image.header)
                        nii_image.set_qform(image.affine)
                        nii_image.set_sform(image.affine)
                        nii_image.to_filename(file_path_nii)