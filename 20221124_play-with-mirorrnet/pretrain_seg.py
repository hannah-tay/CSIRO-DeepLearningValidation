import monai
import torch
import numpy as np
import glob
import os.path
import matplotlib.pyplot as plt
from utils import plot_against_epoch_numbers

num_segmentation_classes = 4  # background, CSF, white matter, gray matter
device = torch.device("cuda:0")
resize = 96


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


# transforms applied to data
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


# datasets and dataloaders
dataset_seg_available_train = monai.data.CacheDataset(
    data=data_seg_available_train,
    transform=transform_seg_available,
    cache_num=16
)
dataset_seg_available_valid = monai.data.CacheDataset(
    data=data_seg_available_valid,
    transform=transform_seg_available,
    cache_num=16
)

dataloader_seg_available_train = monai.data.DataLoader(
    dataset_seg_available_train,
    batch_size=8,
    num_workers=4,
    shuffle=True
)
dataloader_seg_available_valid = monai.data.DataLoader(
    dataset_seg_available_valid,
    batch_size=16,
    num_workers=4,
    shuffle=False
)


# segmentation network
seg_net = monai.networks.nets.UNet(
    3,  # spatial dims
    1,  # input channels
    num_segmentation_classes,  # output channels
    (8, 16, 16, 32, 32, 64, 64),  # channel sequence
    (1, 2, 1, 2, 1, 2),  # convolutional strides
    dropout=0.2,
    norm='batch'
)


# loss function
dice_loss = monai.losses.DiceLoss(
    include_background=True,
    to_onehot_y=True,  # Our seg labels are single channel images indicating class index, rather than one-hot
    softmax=True,  # Note that our segmentation network is missing the softmax at the end
    reduction="mean"
)


# training
seg_net.to(device)

learning_rate = 1e-3
optimizer = torch.optim.Adam(seg_net.parameters(), learning_rate)

max_epochs = 30 #60
training_losses = []
validation_losses = []
val_interval = 5

for epoch_number in range(max_epochs):

    print(f"Epoch {epoch_number+1}/{max_epochs}:")

    seg_net.train()
    losses = []
    for batch in dataloader_seg_available_train:
        imgs = batch['img'].to(device)
        true_segs = batch['seg'].to(device)

        optimizer.zero_grad()
        predicted_segs = seg_net(imgs)
        loss = dice_loss(predicted_segs, true_segs)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    training_loss = np.mean(losses)
    print(f"\ttraining loss: {training_loss}")
    training_losses.append([epoch_number, training_loss])

    if epoch_number % val_interval == 0:
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
        print(f"\tvalidation loss: {validation_loss}")
        validation_losses.append([epoch_number, validation_loss])

# Free up some memory
del loss, predicted_segs, true_segs, imgs
torch.cuda.empty_cache()

# save plots
plot_against_epoch_numbers(training_losses, label="training")
plot_against_epoch_numbers(validation_losses, label="validation")
plt.legend()
plt.ylabel('mean dice loss')
plt.title('seg_net pretraining')
plt.savefig('seg_net_pretrained_losses.png')
plt.show()

# save weights
torch.save(seg_net.state_dict(), 'seg_net_pretrained.pth')