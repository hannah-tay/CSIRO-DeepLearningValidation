import glob
import nibabel as nib
import numpy as np

x_min = 9e9
x_max = 0


for f in glob('*/path/*/aparc+aseg.mgz'):
    im = nib.load(f)
    voxels = im.get_fdata() > 0
    x_idxs, y_idxs, z_idxs = np.nonzero(voxels)
    if x_min > x_idxs.min():
        x_min = x_idxs.min()
    if x_max < x_idxs.max():
        x_max = x_idxs.max()


print('x', x_min, x_max)