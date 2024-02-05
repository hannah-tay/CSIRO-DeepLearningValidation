import h5py

file = h5py.File('src/data/OASIS-1/dice_losses_20221216.hdf5', 'r')
print(list(file.keys()))
print(list(file['BATCH_1'].keys()))
print(file['BATCH_1']['avg_loss'][()])
print(file['BATCH_1']['ids'][()])