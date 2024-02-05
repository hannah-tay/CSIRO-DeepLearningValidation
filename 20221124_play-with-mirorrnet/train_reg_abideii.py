# imports
import monai
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

# sys.path.append('src/mirorrnet/src')
from mirorrnet.datasets.abideii import ABIDEIIPairedLDataModule

from utils import (
    preview_image, preview_3D_vector_field, preview_3D_deformation,
    jacobian_determinant, plot_against_epoch_numbers
)

# monai.config.print_config()

# Set deterministic training for reproducibility
monai.utils.set_determinism(seed=2938649572)


# data
data_module = ABIDEIIPairedLDataModule(
    csv_file="/../../datasets/work/hb-atlas/work/scratch/data/ABIDEII/registered/participants.csv",
    n_valid=50
)
data_module.setup()

dataloader_train = data_module.train_dataloader()
dataloader_valid = data_module.val_dataloader()


# setup
resize = 96
device = torch.device("cuda:0")
num_segmentation_classes = 4


# registration network
reg_net = monai.networks.nets.UNet(
    3,  # spatial dims
    1,  # input channels (one for fixed image and one for moving image)
    3,  # output channels (to represent 3D displacement vector field)
    (16, 32, 32, 32, 32),  # channel sequence
    (1, 2, 2, 2),  # convolutional strides
    dropout=0.2,
    norm="batch"
)

# segmentation network for use in anatomy loss
seg_net = monai.networks.nets.UNet(
    3,  # spatial dims
    1,  # input channels
    num_segmentation_classes,  # output channels
    (8, 16, 16, 32, 32, 64, 64),  # channel sequence
    (1, 2, 1, 2, 1, 2),  # convolutional strides
    dropout=0.2,
    norm='batch'
)


# image warping
data_item = next(iter(dataloader_train))

reg_net_example_input = data_item['image_1']

reg_net_example_output = reg_net(reg_net_example_input)
image_scale = reg_net_example_input.shape[-1]

warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="border")
warp_nearest = monai.networks.blocks.Warp(mode="nearest", padding_mode="border")

# loss functions
dice_loss = monai.losses.DiceLoss(
    include_background=True,
    to_onehot_y=True,  # Our seg labels are single channel images indicating class index, rather than one-hot
    softmax=True,  # Note that our segmentation network is missing the softmax at the end
    reduction="mean"
)
# A version of the dice loss with to_onehot_y=False and softmax=False;
# This will be handy for anatomy loss, for which we often compare two outputs of seg_net
dice_loss2 = monai.losses.DiceLoss(
    include_background=True,
    to_onehot_y=False,
    softmax=False,
    reduction="mean"
)

lncc_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(
    spatial_dims=3,
    kernel_size=3,
    kernel_type='rectangular',
    reduction="mean"
)

bending_loss = monai.losses.BendingEnergyLoss()

regularization_loss = bending_loss

def similarity_loss(displacement_field, image_pair):
    """ Accepts a batch of displacement fields, shape (B,3,H,W,D),
        and a batch of image pairs, shape (B,2,H,W,D). """
    warped_img2 = warp(image_pair[:, [1], :, :, :], displacement_field)
    return lncc_loss(
        warped_img2,  # prediction
        image_pair[:, [0], :, :, :]  # target
    )

def anatomy_loss(displacement_field, image_pair, seg_net, gt_seg1=None, gt_seg2=None):
    """
    Accepts a batch of displacement fields, shape (B,3,H,W,D),
    and a batch of image pairs, shape (B,2,H,W,D).
    seg_net is the model used to segment an image,
      mapping (B,1,H,W,D) to (B,C,H,W,D) where C is the number of segmentation classes.
    gt_seg1 and gt_seg2 are ground truth segmentations for the images in image_pair, if ground truth is available;
      if unavailable then they can be None.
      gt_seg1 and gt_seg2 are expected to be in the form of class labels, with shape (B,1,H,W,D).
    """
    if gt_seg1 is not None:
        # ground truth seg of target image
        seg1 = monai.networks.one_hot(
            gt_seg1, num_segmentation_classes
        )
    else:
        # seg_net on target image, "noisy ground truth"
        seg1 = seg_net(image_pair[:, [0], :, :, :]).softmax(dim=1)

    if gt_seg2 is not None:
        # ground truth seg of moving image
        seg2 = monai.networks.one_hot(
            gt_seg2, num_segmentation_classes
        )
    else:
        # seg_net on moving image, "noisy ground truth"
        seg2 = seg_net(image_pair[:, [1], :, :, :]).softmax(dim=1)

    # seg1 and seg2 are now in the form of class probabilities at each voxel
    # The trilinear interpolation of the function `warp` is then safe to use;
    # it will preserve the probabilistic interpretation of seg2.

    return dice_loss2(
        warp(seg2, displacement_field),  # warp of moving image segmentation
        seg1  # target image segmentation
    )

def reg_losses(batch):
    img12 = batch['image_1'].to(device)

    displacement_field12 = reg_net(img12)

    loss_sim = similarity_loss(displacement_field12, img12)
    loss_reg = regularization_loss(displacement_field12)
    loss_ana = anatomy_loss(displacement_field12, img12, seg_net, gt_seg1=None, gt_seg2=None)

    return loss_sim, loss_reg, loss_ana


# batch generator
def create_batch_generator(dataloader_subdivided):
    """
    Create a batch generator that samples data pairs.

    Arguments:
        dataloader_subdivided : a mapping from the labels in seg_availabilities to dataloaders
        weights : a list of probabilities, one for each label in seg_availabilities;
                  if not provided then we weight by the number of data items of each type,
                  effectively sampling uniformly over the union of the datasets

    Returns: batch_generator
        A function that accepts a number of batches to sample and that returns a generator.
        The generator will weighted-randomly pick one of the seg_availabilities and
        yield the next batch from the corresponding dataloader.
    """

    def batch_generator(num_batches_to_sample):
        for _ in range(num_batches_to_sample):
            yield next(iter(dataloader_subdivided))
    return batch_generator

batch_generator_train_reg = create_batch_generator(dataloader_train)
batch_generator_valid_reg = create_batch_generator(dataloader_valid)


# training 
reg_net.to(device)

learning_rate_reg = 5e-4
optimizer_reg = torch.optim.Adam(reg_net.parameters(), learning_rate_reg)

lambda_a = 2.0  # anatomy loss weight
lambda_sp = 3.0  # supervised segmentation loss weight

# regularization loss weight
# This often requires some careful tuning. Here we suggest a value, which unfortunately needs to
# depend on image scale. This is because the bending energy loss is not scale-invariant.
# 7.5 worked well with the above hyperparameters for images of size 128x128x128.
lambda_r = 7.5 * (image_scale / 128)**2

max_epochs = 120
reg_phase_training_batches_per_epoch = 40
reg_phase_num_validation_batches_to_use = 40
val_interval = 5

training_losses_reg = []
validation_losses_reg = []

best_reg_validation_loss = float('inf')

for epoch_number in range(max_epochs):

    print(f"Epoch {epoch_number+1}/{max_epochs}:")

    losses = []
    for batch in batch_generator_train_reg(reg_phase_training_batches_per_epoch):
        optimizer_reg.zero_grad()
        loss_sim, loss_reg, loss_ana = reg_losses(batch)
        loss = loss_sim + lambda_r * loss_reg + lambda_a * loss_ana
        loss.backward()
        optimizer_reg.step()
        losses.append(loss.item())

    training_loss = np.mean(losses)
    print(f"\treg training loss: {training_loss}")
    training_losses_reg.append([epoch_number, training_loss])

    if epoch_number % val_interval == 0:
        reg_net.eval()
        losses = []
        with torch.no_grad():
            for batch in batch_generator_valid_reg(reg_phase_num_validation_batches_to_use):
                loss_sim, loss_reg, loss_ana = reg_losses(batch)
                loss = loss_sim + lambda_r * loss_reg + lambda_a * loss_ana
                losses.append(loss.item())

        validation_loss = np.mean(losses)
        print(f"\treg validation loss: {validation_loss}")
        validation_losses_reg.append([epoch_number, validation_loss])

        if validation_loss < best_reg_validation_loss:
            best_reg_validation_loss = validation_loss
            torch.save(reg_net.state_dict(), 'reg_net_best.pth')

    # Free up memory
    del loss, loss_sim, loss_reg, loss_ana
    torch.cuda.empty_cache()

print(f"\n\nBest reg_net validation loss: {best_reg_validation_loss}")

# save plots
plot_against_epoch_numbers(training_losses_reg, label="training")
plot_against_epoch_numbers(validation_losses_reg, label="validation")
plt.legend()
plt.ylabel('loss')
plt.title('Alternating training: registration loss')
plt.savefig('reg_net_losses.png')
plt.show()

# save weights
torch.save(reg_net.state_dict(), 'reg_net.pth')