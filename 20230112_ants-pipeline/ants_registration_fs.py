# creates job scripts to run antsRegistrationSyN.sh on each pair of images in a dataset

from argparse import ArgumentParser
import pandas
import monai
from utils import PairedDataset
# import h5py
# from monai.losses.dice import DiceLoss
# import torch
from monai.networks import one_hot
import os
# import numpy as np
import nibabel
import ants
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--data_table", type=str) # path to .csv file with images
    parser.add_argument("--image_key", type=str)
    parser.add_argument("--seg_key", type=str)
    parser.add_argument("--id_key", type=str) 
    parser.add_argument("--n_valid", type=int)
    parser.add_argument("--dataset", type=str)

    # optional:
    parser.add_argument("--freesurfer", type=bool, default=False) # whether to convert from FS .mgz
    parser.add_argument("--max_severity", type=int, default=None) # maximum injury severity 
    
    args, _ = parser.parse_known_args()

    data_frame: pandas.DataFrame = pandas.read_csv(args.data_table)
    data_path = os.path.dirname(args.data_table)
    dataset_name = args.dataset
    
    # limit injury severity of data if required
    if args.max_severity and 'severity' in data_frame.keys():
        if args.max_severity < max(data_frame['severity']):
            dataset_name = args.dataset + '_modified'

        data_frame = data_frame[data_frame['severity'] <= args.max_severity]

    if args.freesurfer:
        dataset_name = args.dataset + '_freesurfer_vxm'

    # ensure path
    data_frame_copy = data_frame.copy()
    keys = [args.image_key, args.seg_key]
    for key in keys:
        for i, image in enumerate(data_frame[key]):
            image = os.path.join(data_path, image)
            data_frame_copy[key][i] = image
    data_frame = data_frame_copy

    # create dataset and dataloader
    dataset = monai.data.CSVDataset(
        src=data_frame,
        #transform=load_transform
    )
    dataset = PairedDataset(orig_dataset=dataset)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset=dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     shuffle=True
    # )
    # completed_pairs = []

    for idx, datum in enumerate(dataset):
        if idx == args.n_valid: break
        
        image_1_fn = datum[f'{args.image_key}_1']
        image_2_fn = datum[f'{args.image_key}_2']
        seg_1_fn = datum[f'{args.seg_key}_1']
        seg_2_fn = datum[f'{args.seg_key}_2']
        id_1 = datum[f'{args.id_key}_1']
        id_2 = datum[f'{args.id_key}_2']
        # sorted_images = tuple(sorted([image_1_fn, image_2_fn]))

        if image_1_fn == image_2_fn:
            continue
        # if sorted_images in completed_pairs:
        #     continue
        # completed_pairs.append(sorted_images)
        
        job_name = f'ants_registration_{idx}'
        os.makedirs(f'job_data/ants_registration_{dataset_name}/{job_name}', exist_ok=True)
        job_file_name = f'job_data/ants_registration_{dataset_name}/{job_name}/{job_name}.q'
        job_file_resample_name = f'job_data/ants_registration_{dataset_name}/{job_name}/resample_{job_name}.q'
        job_info_file = f'job_data/ants_registration_{dataset_name}/{job_name}/{job_name}.txt'

        output_prefix = 'OUT'
        # output_forward_prefix = f'{output_prefix}FORWARD/'
        # output_backward_prefix = f'{output_prefix}BACKWARD/'

        job_file_contents = f"""\
#!/bin/bash -l

#SBATCH --account=OD-221016
#SBATCH --time=6:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --output=/flush5/tay400/job_data/sbatch_out/%j.out

source /flush5/tay400/envs/ants/bin/activate
export ANTSPATH=/flush5/tay400/envs/ants/ANTs/bin/
export PATH=/flush5/tay400/envs/ants/ANTs/bin/:$PATH
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$SLURM_NTASKS_PER_NODE

cd {job_name}
cp {image_1_fn} moving.mgz
cp {image_2_fn} fixed.mgz
cp {seg_1_fn} moving_seg.mgz
cp {seg_2_fn} fixed_seg.mgz

# default call
# antsRegistrationSyN.sh -t so -d 3 -m moving.mgz -f fixed.mgz -o {output_prefix} 

# test call
/flush5/tay400/envs/ants/ANTs/bin//antsRegistration --verbose 1 --dimensionality 3 --float 0 \
    --collapse-output-transforms 1 --output [ OUT,OUTWarped.nii.gz,OUTInverseWarped.nii.gz ] \
    --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995 ] \
    --initial-moving-transform [ fixed.mgz,moving.mgz,1 ] --transform SyN[ 0.25,3,0.2] \
    --metric CC[ fixed.mgz,moving.mgz,1,4 ] --convergence [ 201x201x201x201,1e-6,10 ] \
    --shrink-factors 8x4x2x1 --smoothing-sigmas 2x1x.5x0vox

# from Voxelmorph issue on Github
# /path/to/ANTS 3 -m CC[/path/to/atlas.nii.gz,/path/to/subject.nii.gz,1,4] -t Syn[0.25] 
# -o /path/to/output_prefix --number-of-affine-iterations 0x0 -i 201x201x201 -r Gauss[9,0.2]
"""
        job_file_resample_contents = f"""\
#!/bin/bash -l

#SBATCH --account=OD-221016
#SBATCH --time=0:10:00
#SBATCH --ntasks-per-node=2
#SBATCH --output=/flush5/tay400/job_data/sbatch_out/%j.out

source /flush5/tay400/envs/ants/bin/activate
export ANTSPATH=/flush5/tay400/envs/ants/ANTs/bin/
export PATH=/flush5/tay400/envs/ants/ANTs/bin/:$PATH
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$SLURM_NTASKS_PER_NODE

cd {job_name}

antsApplyTransforms -d 3 -i moving_seg.mgz -r fixed_seg.mgz -n NearestNeighbor -t {output_prefix}1Warp.nii.gz -o {output_prefix}WarpedSeg.nii.gz
antsApplyTransforms -d 3 -i fixed_seg.mgz -r moving_seg.mgz -n NearestNeighbor -t [{output_prefix}1Warp.nii.gz, 1] -o {output_prefix}InverseWarpedSeg.nii.gz

"""

        with open(job_file_resample_name, 'w+') as fp:
            fp.write(job_file_resample_contents)

        with open(job_file_name, 'w+') as fp:
            fp.write(job_file_contents)

        with open(job_info_file, 'w+') as fp:
            fp.write(f'MOVING_{id_1}_FIXED_{id_2}')



if __name__ == "__main__":
    main()
