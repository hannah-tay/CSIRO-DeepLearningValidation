import os
from argparse import ArgumentParser
# from monai.losses.dice import DiceLoss
import monai
from monai.metrics.meandice import compute_dice
# from monai.networks import one_hot
from torch.nn.functional import one_hot
import h5py
import torch
import numpy as np
import nibabel
# import tensorflow as tf
import pandas as pd

# NiftiSaver is deprecated - may need to use transforms.SaveImage in future
def save_nifti(data, meta_data, filename, idx):
    monai.data.write_nifti(
        data=data, 
        file_name=filename,
        affine=meta_data.get("affine", None)[idx],
        target_affine=meta_data.get("original_affine", None)[idx],
        resample=False,
        output_spatial_shape=meta_data.get("spatial_shape", None)[idx],
    )

def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--method", type=str)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--version", type=str)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    INPUT_DIR = f'job_data/{args.method}_registration_{args.dataset}/'
    # print(INPUT_DIR)
    # print(list(os.walk(INPUT_DIR)))
    output = {}
    CSV_DIR = INPUT_DIR + f'{args.method}_{args.dataset}.csv'
    # loss_fn = DiceLoss(softmax=True, reduction="none", include_background=False)
    if args.output_dir: file = h5py.File(args.output_dir, 'w')
    
    tmp = np.zeros((1, args.num_classes-1))
    num_failures = 0

    for i in range(1, len(list(os.walk(INPUT_DIR)))):
    # for i in range(1, 110):
        IMAGE_DIR = INPUT_DIR + f'{args.method}_registration_{i}/'
        # print(IMAGE_DIR)

        if not os.path.isdir(IMAGE_DIR): continue  # TODO: why did a folder get skipped

        with open(IMAGE_DIR + f'{args.method}_registration_{i}.txt', 'r') as fp:
            job_info = fp.read()

        try: moving_nib = nibabel.load(IMAGE_DIR + 'moving.nii.gz')
        except: moving_nib = nibabel.load(IMAGE_DIR + 'moving.mgz')
        moving = np.array(moving_nib.dataobj).squeeze()

        try: fixed = nibabel.load(IMAGE_DIR + 'fixed.nii.gz')
        except: fixed = nibabel.load(IMAGE_DIR + 'fixed.mgz')
        fixed = np.array(fixed.dataobj).squeeze()

        try: fixed_seg = nibabel.load(IMAGE_DIR + 'fixed_seg.nii.gz')
        except: fixed_seg = nibabel.load(IMAGE_DIR + 'fixed_seg.mgz')
        fixed_seg = fixed_seg.get_fdata().squeeze()

        try: warped_moving = nibabel.load(IMAGE_DIR + 'OUTWarped.nii.gz')
        except: warped_moving = nibabel.load(IMAGE_DIR + 'OUTWarped.nii')
        warped_moving = warped_moving.get_fdata().squeeze()

        try: warped_seg_moving = nibabel.load(IMAGE_DIR + 'OUTWarpedSeg.nii.gz')
        except: warped_seg_moving = nibabel.load(IMAGE_DIR + 'OUTWarpedSeg.nii')
        warped_seg_moving = warped_seg_moving.get_fdata().squeeze()

        try: 
            warped_grid = nibabel.load(IMAGE_DIR + 'OUTWarpedGrid.nii')
            warped_grid = warped_grid.get_fdata().squeeze() 
        except: warped_grid = [] # TODO: warp grid using ANTs registration output
    

        # calculate dice loss for each pair
        fixed_seg_hard = torch.round(torch.Tensor(fixed_seg))
        warped_seg_moving_hard = torch.round(torch.Tensor(warped_seg_moving))

        fixed_seg_hot = torch.movedim(one_hot(fixed_seg_hard.long()), -1, 0) 
        warped_seg_moving_hot = torch.movedim(one_hot(warped_seg_moving_hard.long()), -1, 0)
        # print(fixed_seg_hot.shape, warped_seg_moving_hot.shape)

        try: 
            intersection_seg = fixed_seg_hot * warped_seg_moving_hot
            intersection_seg = np.argmax(intersection_seg, axis=0) 
            # print(intersection_seg.shape)
            # print(intersection_seg.max(), intersection_seg.min())
            intersection_seg_nii = type(moving_nib)(intersection_seg.numpy(), moving_nib.affine, moving_nib.header)
            intersection_seg_nii.to_filename(IMAGE_DIR + 'OUTIntersection.nii.gz')
        
            inverse_intersection = 1 - (fixed_seg_hot * warped_seg_moving_hot)
            inverse_intersection = torch.prod(inverse_intersection, dim=0)
            intersection_inv_nii = type(moving_nib)(inverse_intersection.numpy(), moving_nib.affine, moving_nib.header)
            intersection_inv_nii.to_filename(IMAGE_DIR + 'OUTInverseIntersection.nii.gz')
            # print(f'intersection seg {(intersection_seg > 0).sum()}')
            # print(f'intersection inverse {(inverse_intersection > 0).sum()}')
            # print(f'fixed seg {(fixed_seg_hot[1:] > 0).sum()}')
            # print(f'warped moving seg {(warped_seg_moving_hot[1:] > 0).sum()}') 
        except: intersection_seg = []

        # batch_loss = loss_fn(warped_seg_moving_hot.float(), fixed_seg_hot.float())
        try: 
            batch_loss = compute_dice(warped_seg_moving_hot.unsqueeze(0), fixed_seg_hot.unsqueeze(0), include_background=False)[0]
            # batch_loss = compute_dice(fixed_seg_hot.unsqueeze(0), fixed_seg_hot.unsqueeze(0), include_background=False)
            # print(f'batch {i}: ', batch_loss)
            if 'info' not in output:
                output['info'] = []
            output['info'].append(job_info)

            for j in range(0, len(batch_loss)):
                if f'dice_{j}' not in output:
                    output[f'dice_{j}'] = []
                output[f'dice_{j}'].append(batch_loss[j].numpy())
        except: 
            num_failures = num_failures + 1 # number of segmentations wrong - MITII injury classes
            # print(f'batch {i} failed due to incorrect segmentations.')
        # print(f'batch {i} average loss: ', torch.mean(batch_loss))

        # if np.isnan(batch_loss).any():
            # print(f'batch {i} failed (contains NaN).')
        
        tmp = np.vstack((tmp, batch_loss.numpy()))

        # create group for each pair in batch
        if file:
            grp = file.create_group(job_info)
            grp['img_moving'] = moving
            grp['img_moving_warped'] = warped_moving
            grp['img_fixed'] = fixed
            # grp['warp_field'] = displacement_field.detach()
            grp['warped_grid'] = warped_grid
            grp['losses'] = batch_loss
            grp['avg_loss'] = torch.mean(batch_loss)
            grp['version'] = args.version
        # print(job_info, '\n')

    file.close()
    # print(tmp.shape)
    print(INPUT_DIR)
    print(f'Validation dataset evaluated. Average dice score is {np.nanmean(tmp, axis=0)}.')
    print(f'Number of failures: {num_failures}.')

    data = pd.DataFrame(output)
    data.to_csv(CSV_DIR)



if __name__ == "__main__":
    main()