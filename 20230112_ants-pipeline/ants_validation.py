# for each folder in job_data/ants_registration/ 
# look for moving, fixed, warped, warp, segmentations (moving, fixed, warped)
# calculate dice loss 
# save outputs to .hdf5 file in plot_images.py-friendly format

import os
from argparse import ArgumentParser
from monai.losses.dice import DiceLoss
from monai.networks import one_hot
import h5py
import torch


def main():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    INPUT_DIR = 'job_data/ants_registration_oasis/'

    file = h5py.File(args.output_dir, 'w')
    completed_pairs = []
    tmp = 0

    for i in range(1, len(next(os.walk(INPUT_DIR))[1])+1):
        IMAGE_DIR = INPUT_DIR + f'ants_registration_{i}/'
        group = (x[1] for x in os.walk(INPUT_DIR + f'ants_registration_{i}/'))
        group = list(group)[0][0]
        RESULTS_DIR = IMAGE_DIR + group + '/'
        # print(IMAGE_DIR)

        id_m = group[7:20]
        id_f = group[-13:-1] + '1' # :(

        moving = IMAGE_DIR + 'moving.nii'
        fixed = IMAGE_DIR + 'fixed.nii'
        fixed_seg = IMAGE_DIR + 'fixed_seg.nii'
        warped_moving = RESULTS_DIR + 'Warped.nii.gz'
        warped_seg_moving = RESULTS_DIR + 'WarpedSeg.nii'

        # calculate dice
        loss_fn = DiceLoss(softmax=True, reduction="none", include_background=False)
        
        warped_seg_moving = one_hot(labels=warped_seg_moving, num_classes=args.num_classes, dim=1)
        fixed_seg = one_hot(labels=fixed_seg, num_classes=args.num_classes, dim=1)
        
        batch_loss = loss_fn(warped_seg_moving, fixed_seg)
        tmp = tmp + torch.mean(batch_loss)

        # validity check
        if id_m == id_f: continue
        if sorted([id_m, id_f]) in completed_pairs: continue
        completed_pairs.append(tuple(sorted([id_m, id_f])))

        # create group for each pair in batch
        grp = file.create_group(group)
        grp['img_moving'] = moving
        grp['img_moving_warped'] = warped_moving
        grp['img_fixed'] = fixed
        grp['warp_field'] = None
        grp['warped_grid'] = None #TODO
        grp['losses'] = batch_loss
        grp['avg_loss'] = torch.mean(batch_loss)
        grp['version'] = 'ANTs_OASIS'
        print(group)


    file.close()
    print(tmp/i)

    print('Validation and saving complete. Output is stored in ', args.output_dir)





if __name__ == "__main__":
    main()