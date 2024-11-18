# Description: This file contains the code to create a custom dataloader for HDF5 files.

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class HDF5Dataset(Dataset):
    def __init__(self, h5_file, transform=None):
        self.h5_file = h5_file # Path to the HDF5 file
        self.transform = transform # Image transformations (augmentation)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as hf:
            return len(hf['images'])

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as hf:
            image = hf['images'][idx] # Access the image by index
            label = hf['labels'][idx] # Access the corresponding label
            image_paths =hf['image_paths'][idx]
        
        if label in [0,1]:
            label =0
        else:
            label =1

        label = torch.tensor(label, dtype=torch.long) # Convert label to a PyTorch tensor
        # long indicates long integer 64 bit beacuse functions in pytorch deal with indices of this type
        # this data type is used for discrete values. loss functions like cross entrpy loss use these to index into target tensors.
        if self.transform:
            image = Image.fromarray(image) # Convert the numpy array to a PIL image
            image = self.transform(image) # Apply transformations

        return image, label

def get_hdf5_dataloader(h5_file, batch_size=32, num_workers=4, transforms=None):
    dataset = HDF5Dataset(h5_file=h5_file, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader