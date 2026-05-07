# contains our own CheXpertDataset class that extends the PyTorch Dataset class
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np


class CheXpertDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform

        # Do some checks to make sure we have the right files
        if split == 'train':
            self.labels_path = os.path.join(root_dir, 'train_reduced.csv')
            self.data_path = os.path.join(root_dir, 'train_npy')
        elif split == 'valid':
            self.labels_path = os.path.join(root_dir, 'valid.csv')
            self.data_path = os.path.join(root_dir, 'valid_npy')
        else:
            raise ValueError('split must be either "train" or "valid"')
        
        # check that we have a csv file with the labels
        if not os.path.exists(self.labels_path):
            raise ValueError(f'Labels path {self.labels_path} does not exist')
        
        # check that we have a directory with the data
        if not os.path.exists(self.data_path):
            raise ValueError(f'Data path {self.data_path} does not exist')
        
        # check that the directory contains .npy files
        files = os.listdir(self.data_path)
        files = [f for f in files if f[-4::] == '.npy']
        if len(files) == 0:
            raise ValueError(f'Data path {self.data_path} does not contain any .npy files')
        
        self.file_names = files

        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        # check that the index is within the range of the dataset
        if idx >= len(self):
            raise IndexError(f'Index {idx} is out of range for dataset of length {len(self)}')
        
        img_name = os.path.join(self.data_path, f"img_{format(idx, '05d')}.npy")
        image = np.load(img_name)

        if self.transform:
            image = self.transform(image)

        # get the label from the csv file
        label = 0
        
        return image, label
    

# Test the CheXpertDataset class
root_dir = '/Users/katephd/Documents/data/CheXpertSmall'
dataset = CheXpertDataset(root_dir, split='train', transform=transforms.ToTensor())
print(len(dataset))
print(dataset[0][0].shape)
print(dataset[0][1])

