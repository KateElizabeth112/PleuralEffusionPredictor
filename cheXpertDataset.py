# contains our own CheXpertDataset class that extends the PyTorch Dataset class
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd


class CheXpertDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', resized=True):
        self.root_dir = root_dir
        self.transform = transform

        # Do some checks to make sure we have the right files
        if split == 'train':
            self.labels_path = os.path.join(root_dir, 'train_reduced.csv')
            if resized:
                self.data_path = os.path.join(root_dir, 'train_resized_npy')
            else:
                self.data_path = os.path.join(root_dir, 'train_npy')
            
        elif split == 'valid':
            self.labels_path = os.path.join(root_dir, 'valid.csv')
            if resized:
                self.data_path = os.path.join(root_dir, 'valid_resized_npy')
            else:
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

        # extract the labels from the csv file using pandas and store them as a numpy array alongside image ids
        df = pd.read_csv(self.labels_path)
        self.labels = df['Pleural Effusion'].values
        self.ids = df['image_id'].values

        
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

        # get the pleural effusion label for this image first using the idx to get the image id, then using the image id to get the label
        label_idx = np.where(self.ids == idx)[0]
        label = self.labels[label_idx]

        # check that we have a label for this image
        if len(label) == 0:
            raise ValueError(f'No label found for image with id {idx}')
        
        return image, label
    

def main():
    # Test the CheXpertDataset class
    root_dir = '/Users/katephd/Documents/data/CheXpertSmall'
    dataset = CheXpertDataset(root_dir, split='train', transform=transforms.ToTensor())
    print(len(dataset))

    # print the label for the first 20 images in the dataset
    for i in range(20):
        print(dataset[i][1])

if __name__ == "__main__":
    main()