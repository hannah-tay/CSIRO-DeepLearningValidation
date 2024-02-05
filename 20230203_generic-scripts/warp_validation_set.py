from argparse import ArgumentParser
from monai.losses.dice import DiceLoss
from typing import Dict
import monai
import torch
import os
import time

from mirorrnet.datasets import (
    ImageAndPhenotypeCSVLDataModule,
    PairedImageLDataModule,
    ABIDEIIPairedLDataModule,
    SCHNPaediatricFDGPairedLDataModule,
)

from mirorrnet.datasets.oasis1 import OASIS1PairedLDataModule
from mirorrnet.lightningmodules.registration import NonrigidRegistrationLModule


DATASETS: Dict[str, PairedImageLDataModule] = {
    "": ImageAndPhenotypeCSVLDataModule,
    "SCHN": SCHNPaediatricFDGPairedLDataModule,
    "ABIDEII": ABIDEIIPairedLDataModule,
    "OASIS": OASIS1PairedLDataModule,
}


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
    parser.add_argument("--debug", action="count", default=0)
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--subject_id", type=str)
    parser.add_argument("--seg_id", type=str, default='seg')
    parser.add_argument("--freesurfer", type=bool, default=False)

    args, _ = parser.parse_known_args()

    if args.dataset != 'oasis': dataset_name = ""
    else: dataset_name = args.dataset

    DATASETS[dataset_name.upper()].add_argparse_args(parser)

    NonrigidRegistrationLModule.add_model_specific_args(parser)

    # custom defaults 
    parser = ArgumentParser(
        description="Validate Registration model", parents=[parser]
    )
    parser.set_defaults(
        default_root_dir="runs/vxm",
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_false",
    )
    args = parser.parse_args()

    datamodule = DATASETS[dataset_name.upper()](**vars(args))

    if not dataset_name:  # default dataset isn't paired
        datamodule = PairedImageLDataModule(datamodule, **vars(args))

    datamodule.setup()
    dataloader_valid = datamodule.val_dataloader()

    registrator = NonrigidRegistrationLModule.load_from_checkpoint(
        args.model_path, **NonrigidRegistrationLModule.parse_argparse_args(args)
    )

    print(f'Warping {args.n_valid} batches ...')
    # tmp = 0
    if args.freesurfer: dataset_name = dataset_name + '_freesurfer'

    OUTPUT_DIR = f'job_data/mirorrnet_registration_{dataset_name}/'
    count = 1

    # run registration on each pair
    for i, batch in enumerate(dataloader_valid):
        # t = time.localtime()
        # current_time = time.strftime("%H:%M:%S", t)
        # print(current_time)
        
        # if count == args.n_valid: break

        id_m, id_f = (
            batch[j] for j in [f'{args.subject_id}_1', f'{args.subject_id}_2']
        ) # moving, fixed
        image_m, image_f, seg_m, seg_f = (
            batch[k] for k in ['image_1', 'image_2', f'{args.seg_id}_1', f'{args.seg_id}_2']
        )
        meta_m, meta_f = (
            batch[q] for q in ['image_meta_dict_1', 'image_meta_dict_2']
        )

        _, _, reg_params = registrator(image_m, image_f)

        # apply warp to each segmentation pair
        displacement_field = registrator.get_forward_displacement(reg_params)

        warped_seg_m = registrator.apply_displacement_field(
            seg_m,
            displacement_field, 
            mode='nearest'
        )

        image_m_warped = registrator.apply_displacement_field(
            image_m,
            displacement_field
        )

        # t = time.localtime()
        # current_time = time.strftime("%H:%M:%S", t)
        # print(current_time)  

        # calculate a grid for plotting warp
        line_every_n = 5
        xyz = torch.meshgrid(
            *(
                torch.arange(0, image_m.shape[i])
                for i in range(-3, 0)
            )
        )
        grid = torch.max(
            torch.stack([(x % line_every_n) == 0 for x in xyz]), dim=0
        ).values.float()
        grid = grid.view(1, 1, *image_m.shape[-3:])  # add batch and ch dims
        grid = torch.tile(grid, (args.batch_size, 1, 1, 1, 1))  # repeat on batch dim
        warped_grid = registrator.apply_displacement_field(
            grid, displacement_field
        )
        warped_grid = warped_grid - warped_grid.min()
        warped_grid /= warped_grid.max()

        for j in range(len(id_m)):
            if id_m[j] == id_f[j]: continue

            IMAGE_DIR = OUTPUT_DIR + f'mirorrnet_registration_{count}/'
            os.makedirs(IMAGE_DIR, exist_ok=True)

            save_nifti(image_m[j][0], meta_m, IMAGE_DIR + 'moving.nii.gz', j)
            save_nifti(image_f[j][0], meta_f, IMAGE_DIR + 'fixed.nii.gz', j)
            save_nifti(seg_m[j][0], meta_m, IMAGE_DIR + 'moving_seg.nii.gz', j)
            save_nifti(seg_f[j][0], meta_f, IMAGE_DIR + 'fixed_seg.nii.gz', j)
            save_nifti(image_m_warped[j][0], meta_f, IMAGE_DIR + 'OUTWarped.nii.gz', j)
            save_nifti(warped_seg_m[j][0], meta_f, IMAGE_DIR + 'OUTWarpedSeg.nii.gz', j)
            save_nifti(warped_grid[j][0], meta_f, IMAGE_DIR + 'OUTWarpedGrid.nii.gz', j)

            print(f'saved batch {count}')
            # print(f'MOVING_{id_m[j]}_FIXED_{id_f[j]}')

            with open(IMAGE_DIR + f'mirorrnet_registration_{count}.txt', 'w+') as fp:
                fp.write(f'MOVING_{id_m[j]}_FIXED_{id_f[j]}')
            count = count + 1
            
    print(f'Validation dataset warped. Images are stored in {OUTPUT_DIR}.')
    
    
if __name__ == "__main__":
    main()