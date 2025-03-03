import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

################################# weight binarize #####################

# Load model
torchdict = torch.load('model.pth')

# binarized weight dictionary
numpydict = {}

# weight binarization and convert to uint8
def binarize_and_convert_to_uint8(tensor):
    binary_tensor = torch.sign(tensor)  # -1 & 1
    binary_tensor = (binary_tensor + 1) // 2  # mapping to 0 & 1
    return binary_tensor.to(dtype=torch.int8)  # uint8

# binarized conv1 weight
conv1_weight = torch.tensor(torchdict['conv1.weight'])
binary_conv1_weight = binarize_and_convert_to_uint8(conv1_weight)
numpydict['conv1w'] = binary_conv1_weight.numpy()

# binarized conv2 weight
conv2_weight = torch.tensor(torchdict['conv2.weight'])
binary_conv2_weight = binarize_and_convert_to_uint8(conv2_weight)
numpydict['conv2w'] = binary_conv2_weight.numpy()

# binarized fc weight
fc_weight = torch.tensor(torchdict['fc.weight'])
binary_fc_weight = binarize_and_convert_to_uint8(fc_weight)
numpydict['fc3w'] = binary_fc_weight.numpy()

# binarized weights saved as .npy
np.save('binary_model_uint8_01.npy', numpydict, allow_pickle=True)