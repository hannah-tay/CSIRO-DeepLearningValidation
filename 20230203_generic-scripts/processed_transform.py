import SimpleITK as sitk
import numpy as np

def main():
    # input_fn = '/datasets/work/hb-atlas/work/scratch/data/OASIS-1/disc1/OAS1_0001_MR1/mri/aseg.nii.gz'
    input_fn = '/datasets/work/hb-atlas/work/scratch/data/OASIS-1/disc1_nii/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.nii'
    # output_fn = '/datasets/work/hb-atlas/work/scratch/data/OASIS-1/disc1/OAS1_0001_MR1/mri/aseg_transformed.nii.gz'
    output_fn = '/datasets/work/hb-atlas/work/scratch/data/OASIS-1/disc1_nii/OAS1_0001_MR1/RAW/transformed_OAS1_0001_MR1_mpr-1_anon.nii'

    matrix = np.array([
        [-0.032628, -0.904287, 0.257911], 
        [-0.004039, 0.268793, 0.828523], 
        [-0.935167, -0.001362, -0.019341]
    ])
    # matrix[:, 0] *= -1
    # matrix[:, 1] *= -1
    matrix = list(matrix.flat) # a_xx, a_xy, ...
    # matrix = list(matrix.T.flat) # a_xx, a_yx, ...

    translation = [-0.9747, 5.4644, 2.2831]
    # scale = 0.910581

    reader = sitk.ImageFileReader()
    reader.SetFileName(input_fn)
    image = reader.Execute()

    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix)
    transform.SetTranslation(translation)
    transform = transform.GetInverse()
    # sitk.WriteTransform(transform, '')

    transformed_image = sitk.Resample(image, transform)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_fn)
    writer.Execute(transformed_image)


if __name__ == "__main__":
    main()