# Description: This file contains the code to create a custom dataloader for HDF5 files.

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class HDF5Dataset(Dataset):
    def __init__(self, h5_file, transform=None):
        """
        Initializes the dataset by opening the HDF5 file and accessing datasets.

        Args:
            h5_file (str): Path to the HDF5 file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.h5_file = h5_file
        self.transform = transform
        self.file = h5py.File(self.h5_file, 'r')  # Open the HDF5 file once
        self.images = self.file['images']
        self.labels = self.file['labels']
        self.image_paths = self.file['image_paths']

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is a transformed image tensor and label is a tensor.
        """
        image = self.images[idx]            # Access the image by index
        label = self.labels[idx]            # Access the corresponding label
        image_path = self.image_paths[idx]  # Access the image path if needed

        label = torch.tensor(label, dtype=torch.long)  # Convert label to a PyTorch tensor

        if self.transform:
            image = Image.fromarray(image)   # Convert the numpy array to a PIL image
            image = self.transform(image)    # Apply transformations

        return image, label

    def __del__(self):
        """
        Ensures that the HDF5 file is properly closed when the dataset object is destroyed.
        """
        self.file.close()


def get_hdf5_dataloader(h5_file, batch_size=32, num_workers=4, transforms=None, return_dataset=False):
    """
    Creates a DataLoader or returns the Dataset based on the parameter.

    Args:
        h5_file (str): Path to the HDF5 file.
        batch_size (int, optional): Number of samples per batch.
        num_workers (int, optional): Number of subprocesses to use for data loading.
        transforms (callable, optional): Optional transform to be applied on a sample.
        return_dataset (bool, optional): If True, returns the Dataset instead of DataLoader.

    Returns:
        DataLoader or Dataset: Depending on the return_dataset flag.
    """
    dataset = HDF5Dataset(h5_file=h5_file, transform=transforms)
    if return_dataset:
        return dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader
