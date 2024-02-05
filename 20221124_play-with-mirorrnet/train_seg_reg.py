import monai
import torch
import numpy as np
import glob
import os.path
import random
import matplotlib.pyplot as plt
from utils import plot_against_epoch_numbers


num_segmentation_classes = 4  # background, CSF, white matter, gray matter
device = torch.device("cuda:0")
resize = 96


# networks
seg_net = monai.networks.nets.UNet(
    3,  # spatial dims
    1,  # input channels
    num_segmentation_classes,  # output channels
    (8, 16, 16, 32, 32, 64, 64),  # channel sequence
    (1, 2, 1, 2, 1, 2),  # convolutional strides
    dropout=0.2,
    norm='batch'
)
reg_net = monai.networks.nets.UNet(
    3,  # spatial dims
    2,  # input channels (one for fixed image and one for moving image)
    3,  # output channels (to represent 3D displacement vector field)
    (16, 32, 32, 32, 32),  # channel sequence
    (1, 2, 2, 2),  # convolutional strides
    dropout=0.2,
    norm="batch"
)


# warp
warp = monai.networks.blocks.Warp(mode="bilinear", padding_mode="border")
warp_nearest = monai.networks.blocks.Warp(mode="nearest", padding_mode="border")


# loss functions
bending_loss = monai.losses.BendingEnergyLoss()

def swap_training(network_to_train, network_to_not_train):
    """
        Switch out of training one network and into training another
    """

    for param in network_to_not_train.parameters():
        param.requires_grad = False

    for param in network_to_train.parameters():
        param.requires_grad = True

    network_to_not_train.eval()
    network_to_train.train()

    regularization_loss = bending_loss

lncc_loss = monai.losses.LocalNormalizedCrossCorrelationLoss(
    spatial_dims=3,
    kernel_size=3,
    kernel_type='rectangular',
    reduction="mean"
)

def similarity_loss(displacement_field, image_pair):
    """ Accepts a batch of displacement fields, shape (B,3,H,W,D),
        and a batch of image pairs, shape (B,2,H,W,D). """
    warped_img2 = warp(image_pair[:, [1], :, :, :], displacement_field)
    return lncc_loss(
        warped_img2,  # prediction
        image_pair[:, [0], :, :, :]  # target
    )

dice_loss2 = monai.losses.DiceLoss(
    include_background=True,
    to_onehot_y=False,
    softmax=False,
    reduction="mean"
)
dice_loss = monai.losses.DiceLoss(
    include_background=True,
    to_onehot_y=True,  # Our seg labels are single channel images indicating class index, rather than one-hot
    softmax=True,  # Note that our segmentation network is missing the softmax at the end
    reduction="mean"
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

regularization_loss = bending_loss

def reg_losses(batch):
    img12 = batch['img12'].to(device)

    displacement_field12 = reg_net(img12)

    loss_sim = similarity_loss(displacement_field12, img12)

    loss_reg = regularization_loss(displacement_field12)

    gt_seg1 = batch['seg1'].to(device) if 'seg1' in batch.keys() else None
    gt_seg2 = batch['seg2'].to(device) if 'seg2' in batch.keys() else None
    loss_ana = anatomy_loss(displacement_field12, img12, seg_net, gt_seg1, gt_seg2)

    return loss_sim, loss_reg, loss_ana


# transforms for data
transform_pair = monai.transforms.Compose(
    transforms=[
        monai.transforms.LoadImageD(keys=['img1', 'seg1', 'img2', 'seg2'], image_only=True, allow_missing_keys=True),
        monai.transforms.TransposeD(keys=['img1', 'seg1', 'img2', 'seg2'], indices=(2, 1, 0), allow_missing_keys=True),
        monai.transforms.EnsureChannelFirstD(keys=['img1', 'seg1', 'img2', 'seg2'], allow_missing_keys=True),
        monai.transforms.ConcatItemsD(keys=['img1', 'img2'], name='img12', dim=0),
        monai.transforms.DeleteItemsD(keys=['img1', 'img2']),
        monai.transforms.ResizeD(
            keys=['img12', 'seg1', 'seg2'],
            spatial_size=(resize, resize, resize),
            mode=['trilinear', 'nearest', 'nearest'],
            allow_missing_keys=True,
            align_corners=[False, None, None]
        ) if resize is not None else monai.transforms.Identity()
    ]
)


# download and extract data
data_dir = "OASIS-1"

resource = "https://download.nrg.wustl.edu/data/oasis_cross-sectional_disc1.tar.gz"
md5 = "c83e216ef8654a7cc9e2a30a4cdbe0cc"

compressed_file = "oasis_cross-sectional_disc1.tar.gz"
if not os.path.exists(data_dir):
    monai.apps.utils.download_and_extract(resource, compressed_file, data_dir, md5)

num_segmentation_classes = 4  # background, CSF, white matter, gray matter

image_path_expression = "PROCESSED/MPRAGE/T88_111/OAS1_*_MR*_mpr_n*_anon_111_t88_masked_gfc.img"
segmentation_path_expression = "FSL_SEG/OAS1_*_MR*_mpr_n*_anon_111_t88_masked_gfc_fseg.img"

image_paths = glob.glob(os.path.join(data_dir, '*', image_path_expression))
image_paths += glob.glob(os.path.join(data_dir, '*/*', image_path_expression))
segmentation_paths = glob.glob(os.path.join(data_dir, '*', segmentation_path_expression))
segmentation_paths += glob.glob(os.path.join(data_dir, '*/*', segmentation_path_expression))

num_segs_to_select = 10
np.random.shuffle(segmentation_paths)
segmentation_paths = segmentation_paths[:num_segs_to_select]


# organise data for datasets
def path_to_id(path):
    return os.path.basename(path).strip('OAS1_')[:8]

seg_ids = list(map(path_to_id, segmentation_paths))
img_ids = map(path_to_id, image_paths)
data = []
for img_index, img_id in enumerate(img_ids):
    data_item = {'img': image_paths[img_index]}
    if img_id in seg_ids:
        data_item['seg'] = segmentation_paths[seg_ids.index(img_id)]
    data.append(data_item)

data_seg_available = list(filter(lambda d: 'seg' in d.keys(), data))
data_seg_unavailable = list(filter(lambda d: 'seg' not in d.keys(), data))

data_seg_available_train, data_seg_available_valid = \
    monai.data.utils.partition_dataset(data_seg_available, ratios=(8, 2))

data_without_seg_valid = data_seg_unavailable + data_seg_available_train  # Note the order

data_valid, data_train = monai.data.utils.partition_dataset(
    data_without_seg_valid,  # Note the order
    ratios=(2, 8),  # Note the order
    shuffle=False
)

def take_data_pairs(data, symmetric=True):
    """Given a list of dicts that have keys for an image and maybe a segmentation,
    return a list of dicts corresponding to *pairs* of images and maybe segmentations.
    Pairs consisting of a repeated image are not included.
    If symmetric is set to True, then for each pair that is included, its reverse is also included"""
    data_pairs = []
    for i in range(len(data)):
        j_limit = len(data) if symmetric else i
        for j in range(j_limit):
            if j == i:
                continue
            d1 = data[i]
            d2 = data[j]
            pair = {
                'img1': d1['img'],
                'img2': d2['img']
            }
            if 'seg' in d1.keys():
                pair['seg1'] = d1['seg']
            if 'seg' in d2.keys():
                pair['seg2'] = d2['seg']
            data_pairs.append(pair)
    return data_pairs

data_pairs_valid = take_data_pairs(data_valid)
data_pairs_train = take_data_pairs(data_train)


# datasets and dataloaders
def subdivide_list_of_data_pairs(data_pairs_list):
    out_dict = {'00': [], '01': [], '10': [], '11': []}
    for d in data_pairs_list:
        if 'seg1' in d.keys() and 'seg2' in d.keys():
            out_dict['11'].append(d)
        elif 'seg1' in d.keys():
            out_dict['10'].append(d)
        elif 'seg2' in d.keys():
            out_dict['01'].append(d)
        else:
            out_dict['00'].append(d)
    return out_dict

data_pairs_valid_subdivided = subdivide_list_of_data_pairs(data_pairs_valid)
data_pairs_train_subdivided = subdivide_list_of_data_pairs(data_pairs_train)

dataset_pairs_train_subdivided = {
    seg_availability: monai.data.CacheDataset(
        data=data_list,
        transform=transform_pair,
        cache_num=32
    )
    for seg_availability, data_list in data_pairs_train_subdivided.items()
}

dataset_pairs_valid_subdivided = {
    seg_availability: monai.data.CacheDataset(
        data=data_list,
        transform=transform_pair,
        cache_num=32
    )
    for seg_availability, data_list in data_pairs_valid_subdivided.items()
}

dataloader_pairs_train_subdivided = {
    seg_availability: monai.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=4,
        shuffle=True
    )
    if len(dataset) > 0 else []  # empty dataloaders are not a thing-- put an empty list if needed
    for seg_availability, dataset in dataset_pairs_train_subdivided.items()
}

dataloader_pairs_valid_subdivided = {
    seg_availability: monai.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=4,
        shuffle=True  # Shuffle validation data because we will only take a sample for validation each time
    )
    if len(dataset) > 0 else []  # empty dataloaders are not a thing-- put an empty list if needed
    for seg_availability, dataset in dataset_pairs_valid_subdivided.items()
}

# validation data
transform_seg_available = monai.transforms.Compose(
    # transforms=[
        monai.transforms.LoadImageD(keys=['img', 'seg'], ensure_channel_first=True, image_only=True),
        # monai.transforms.EnsureChannelFirstD(keys=['img', 'seg']),
        monai.transforms.TransposeD(keys=['img', 'seg'], indices=(2, 1, 0)),
        monai.transforms.ResizeD(
            keys=['img', 'seg'],
            spatial_size=(resize, resize, resize),
            mode=['trilinear', 'nearest'],
            align_corners=[False, None]
        ) if resize is not None else monai.transforms.Identity()
    # ]
)
dataset_seg_available_valid = monai.data.CacheDataset(
    data=data_seg_available_valid,
    transform=transform_seg_available,
    cache_num=16
)
dataloader_seg_available_valid = monai.data.DataLoader(
    dataset_seg_available_valid,
    batch_size=16,
    num_workers=4,
    shuffle=False
)


# batch generator
seg_availabilities = ['00', '01', '10', '11']

def create_batch_generator(dataloader_subdivided, weights=None):
    """
    Create a batch generator that samples data pairs with various segmentation availabilities.

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
    if weights is None:
        weights = np.array([len(dataloader_subdivided[s]) for s in seg_availabilities])
    weights = np.array(weights)
    weights = weights / weights.sum()
    dataloader_subdivided_as_iterators = {s: iter(d) for s, d in dataloader_subdivided.items()}

    def batch_generator(num_batches_to_sample):
        for _ in range(num_batches_to_sample):
            seg_availability = np.random.choice(seg_availabilities, p=weights)
            try:
                yield next(dataloader_subdivided_as_iterators[seg_availability])
            except StopIteration:  # If dataloader runs out, restart it
                dataloader_subdivided_as_iterators[seg_availability] =\
                    iter(dataloader_subdivided[seg_availability])
                yield next(dataloader_subdivided_as_iterators[seg_availability])
    return batch_generator

batch_generator_train_reg = create_batch_generator(dataloader_pairs_train_subdivided)
batch_generator_valid_reg = create_batch_generator(dataloader_pairs_valid_subdivided)

# When training seg_net alone, we only consider data pairs for which at least one ground truth seg is available
seg_train_sampling_weights = [0] + [len(dataloader_pairs_train_subdivided[s]) for s in seg_availabilities[1:]]
print(f"""When training seg_net alone, segmentation availabilities {seg_availabilities}
will be sampled with respective weights {seg_train_sampling_weights}""")
batch_generator_train_seg = create_batch_generator(dataloader_pairs_train_subdivided, seg_train_sampling_weights)

# forward pass
def take_random_from_subdivided_dataset(dataset_subdivided):
    """Given a dict mapping segmentation availability labels to datasets, return a random data item"""
    datasets = list(dataset_subdivided.values())
    datasets_combined = sum(datasets[1:], datasets[0])
    return random.choice(datasets_combined)


data_item = take_random_from_subdivided_dataset(dataset_pairs_train_subdivided)
reg_net_example_input = data_item['img12'].unsqueeze(0)
reg_net_example_output = reg_net(reg_net_example_input)
print(f"Shape of reg_net input: {reg_net_example_input.shape}")
print(f"Shape of reg_net output: {reg_net_example_output.shape}")
image_scale = reg_net_example_input.shape[-1]


# training
seg_net.to(device)
reg_net.to(device)

learning_rate_reg = 5e-4
optimizer_reg = torch.optim.Adam(reg_net.parameters(), learning_rate_reg)

learning_rate_seg = 1e-3
optimizer_seg = torch.optim.Adam(seg_net.parameters(), learning_rate_seg)

lambda_a = 2.0  # anatomy loss weight
lambda_sp = 3.0  # supervised segmentation loss weight

# regularization loss weight
# This often requires some careful tuning. Here we suggest a value, which unfortunately needs to
# depend on image scale. This is because the bending energy loss is not scale-invariant.
# 7.5 worked well with the above hyperparameters for images of size 128x128x128.
lambda_r = 7.5 * (image_scale / 128)**2

max_epochs = 120
reg_phase_training_batches_per_epoch = 40
seg_phase_training_batches_per_epoch = 5  # Fewer batches needed, because seg_net converges more quickly
reg_phase_num_validation_batches_to_use = 40
val_interval = 5

training_losses_reg = []
validation_losses_reg = []
training_losses_seg = []
validation_losses_seg = []

best_seg_validation_loss = float('inf')
best_reg_validation_loss = float('inf')

for epoch_number in range(max_epochs):

    print(f"Epoch {epoch_number+1}/{max_epochs}:")

    # ------------------------------------------------
    #         reg_net training, with seg_net frozen
    # ------------------------------------------------

    # Keep computational graph in memory for reg_net, but not for seg_net, and do reg_net.train()
    swap_training(reg_net, seg_net)

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

    # ------------------------------------------------
    #         seg_net training, with reg_net frozen
    # ------------------------------------------------

    # Keep computational graph in memory for seg_net, but not for reg_net, and do seg_net.train()
    swap_training(seg_net, reg_net)

    losses = []
    for batch in batch_generator_train_seg(seg_phase_training_batches_per_epoch):
        optimizer_seg.zero_grad()

        img12 = batch['img12'].to(device)

        displacement_fields = reg_net(img12)
        seg1_predicted = seg_net(img12[:, [0], :, :, :]).softmax(dim=1)
        seg2_predicted = seg_net(img12[:, [1], :, :, :]).softmax(dim=1)

        # Below we compute the following:
        # loss_supervised: supervised segmentation loss; compares ground truth seg with predicted seg
        # loss_anatomy: anatomy loss; compares warped seg of moving image to seg of target image
        # loss_metric: a single supervised seg loss, as a metric to track the progress of training

        if 'seg1' in batch.keys() and 'seg2' in batch.keys():
            seg1 = monai.networks.one_hot(batch['seg1'].to(device), num_segmentation_classes)
            seg2 = monai.networks.one_hot(batch['seg2'].to(device), num_segmentation_classes)
            loss_metric = dice_loss2(seg2_predicted, seg2)
            loss_supervised = dice_loss2(seg1_predicted, seg1) + loss_metric
            # The above supervised loss looks a bit different from the one in the paper
            # in that it includes predictions for both images in the current image pair;
            # we might as well do this, since we have gone to the trouble of loading
            # both segmentations into memory.

        elif 'seg1' in batch.keys():  # seg1 available, but no seg2
            seg1 = monai.networks.one_hot(batch['seg1'].to(device), num_segmentation_classes)
            loss_metric = dice_loss2(seg1_predicted, seg1)
            loss_supervised = loss_metric
            seg2 = seg2_predicted  # Use this in anatomy loss

        else:  # seg2 available, but no seg1
            assert('seg2' in batch.keys())
            seg2 = monai.networks.one_hot(batch['seg2'].to(device), num_segmentation_classes)
            loss_metric = dice_loss2(seg2_predicted, seg2)
            loss_supervised = loss_metric
            seg1 = seg1_predicted  # Use this in anatomy loss

        # seg1 and seg2 should now be in the form of one-hot class probabilities

        loss_anatomy = dice_loss2(warp_nearest(seg2, displacement_fields), seg1)\
            if 'seg1' in batch.keys() or 'seg2' in batch.keys()\
            else 0.  # It wouldn't really be 0, but it would not contribute to training seg_net

        # (If you want to refactor this code for *joint* training of reg_net and seg_net,
        #  then use the definition of anatomy loss given in the function anatomy_loss above,
        #  where differentiable warping is used and reg net can be trained with it.)

        loss = lambda_a * loss_anatomy + lambda_sp * loss_supervised
        loss.backward()
        optimizer_seg.step()

        losses.append(loss_metric.item())

    training_loss = np.mean(losses)
    print(f"\tseg training loss: {training_loss}")
    training_losses_seg.append([epoch_number, training_loss])

    if epoch_number % val_interval == 0:
        # The following validation loop would not do anything in the case
        # where there is just one segmentation available,
        # because data_seg_available_valid would be empty.
        seg_net.eval()
        losses = []
        with torch.no_grad():
            for batch in dataloader_seg_available_valid:
                imgs = batch['img'].to(device)
                true_segs = batch['seg'].to(device)
                predicted_segs = seg_net(imgs)
                loss = dice_loss(predicted_segs, true_segs)
                losses.append(loss.item())

        validation_loss = np.mean(losses)
        print(f"\tseg validation loss: {validation_loss}")
        validation_losses_seg.append([epoch_number, validation_loss])

        if validation_loss < best_seg_validation_loss:
            best_seg_validation_loss = validation_loss
            torch.save(seg_net.state_dict(), 'seg_net_best.pth')

    # Free up memory
    del loss, seg1, seg2, displacement_fields, img12, loss_supervised, loss_anatomy, loss_metric,\
        seg1_predicted, seg2_predicted
    torch.cuda.empty_cache()

print(f"\n\nBest reg_net validation loss: {best_reg_validation_loss}")
print(f"Best seg_net validation loss: {best_seg_validation_loss}")


# save plots
plot_against_epoch_numbers(training_losses_reg, label="training")
plot_against_epoch_numbers(validation_losses_reg, label="validation")
plt.legend()
plt.ylabel('loss')
plt.title('Alternating training: registration loss')
plt.savefig('reg_net_losses.png')
plt.show()

plot_against_epoch_numbers(training_losses_seg, label="training")
plt.ylabel('training loss')
plt.title('Alternating training: segmentation loss (training)')
plt.savefig('seg_net_training_losses.png')
plt.show()

plot_against_epoch_numbers(validation_losses_seg, label="validation", color='orange')
plt.ylabel('validation loss')
plt.title('Alternating training: segmentation loss (validation)')
plt.savefig('seg_net_validation_losses.png')
plt.show()


# save weights
torch.save(seg_net.state_dict(), 'seg_net.pth')
torch.save(reg_net.state_dict(), 'reg_net.pth')