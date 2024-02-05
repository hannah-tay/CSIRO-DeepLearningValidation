# import and print module names
import pandas as pd
import argparse
import monai
import h5py
import torch
import numpy as np
import nibabel
import tensorflow as tf
import seaborn as sns
import matplotlib

try: import ants 
except: pass
# need to be environment with ANTs installed
# guide used: https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS 

print('Module Versions')
print('===============')

print('pandas: ', pd.__version__)
print('argparse: ', argparse.__version__)
print('monai: ', monai.__version__)
print('h5py: ', h5py.__version__)
print('torch: ', torch.__version__)
print('numpy: ', np.__version__)
print('nibabel: ', nibabel.__version__)
print('tensorflow: ', tf.__version__)
print('seaborn: ', sns.__version__)
print('matplotlib: ', matplotlib.__version__)

try: print('ants: ', ants.__version__)
except: print('Note: no ANTs module found.')